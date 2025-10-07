"""
PDF Indexer for RAG Pipeline - Step 1: Text Extraction and Vector Storage
This module handles loading PDFs, extracting text by page, generating embeddings,
and storing chunks in FAISS for semantic search.
"""

import os
import fitz  # PyMuPDF
import pickle
from typing import List, Dict
import numpy as np
import faiss
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class PDFIndexer:
    """
    Main class for indexing PDF documents into a vector database.
    Handles text extraction, embedding generation, and storage.
    """
    
    def __init__(self, index_path: str = "faiss_index"):
        """
        Initialize the PDFIndexer with OpenAI client and FAISS index.
        
        Args:
            index_path: Path to save/load the FAISS index and metadata
        """
        # Initialize OpenAI client for embeddings
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.embedding_model = "text-embedding-3-large"
        self.embedding_dimension = 1536  # Dimension for text-embedding-3-small
        
        # Initialize FAISS index
        self.index_path = index_path
        self.index = faiss.IndexFlatL2(self.embedding_dimension)
        
        # Store metadata for each indexed chunk
        self.metadata = []  # List of dicts with document_id, page_number, text
        
        # Try to load existing index
        self._load_index()
        
        print(f"✓ Initialized PDFIndexer with FAISS index at: {index_path}")
    
    def _load_index(self):
        """Load existing FAISS index and metadata if available."""
        if os.path.exists(f"{self.index_path}.index") and os.path.exists(f"{self.index_path}.metadata"):
            try:
                self.index = faiss.read_index(f"{self.index_path}.index")
                with open(f"{self.index_path}.metadata", "rb") as f:
                    self.metadata = pickle.load(f)
                print(f"✓ Loaded existing index with {len(self.metadata)} documents")
            except Exception as e:
                print(f"⚠ Could not load existing index: {e}")
    
    def _save_index(self):
        """Save FAISS index and metadata to disk."""
        faiss.write_index(self.index, f"{self.index_path}.index")
        with open(f"{self.index_path}.metadata", "wb") as f:
            pickle.dump(self.metadata, f)
        print(f"✓ Saved index to {self.index_path}")
    
    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict[str, any]]:
        """
        Extract text from each page of a PDF document.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of dictionaries containing page text and metadata
        """
        pages_data = []
        
        # Open the PDF
        pdf_document = fitz.open(pdf_path)
        document_id = os.path.basename(pdf_path)
        
        print(f"📄 Processing PDF: {document_id}")
        print(f"   Total pages: {len(pdf_document)}")
        
        # Iterate through each page
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            
            # Extract text from the page
            text = page.get_text("text")
            
            # Store page data with metadata
            page_data = {
                "document_id": document_id,
                "page_number": page_num + 1,  # 1-indexed for readability
                "text": text.strip()
            }
            
            pages_data.append(page_data)
        
        pdf_document.close()
        print(f"✓ Extracted text from {len(pages_data)} pages")
        
        return pages_data
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a given text using OpenAI's embedding model.
        
        Args:
            text: Text to embed
            
        Returns:
            NumPy array representing the embedding vector
        """
        # Handle empty text
        if not text or text.isspace():
            text = "[Empty page]"
        
        # Generate embedding using OpenAI API
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=text,
            dimensions=self.embedding_dimension
        )
        
        embedding = np.array(response.data[0].embedding, dtype='float32')
        return embedding
    
    def index_pdf(self, pdf_path: str) -> int:
        """
        Complete indexing pipeline: extract text, generate embeddings, and store in FAISS.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Number of pages indexed
        """
        # Step 1: Extract text from all pages
        pages_data = self.extract_text_from_pdf(pdf_path)
        
        # Step 2: Generate embeddings and add to FAISS
        print("🔄 Generating embeddings...")
        embeddings_list = []
        
        for page_data in pages_data:
            # Generate embedding for this page's text
            embedding = self.generate_embedding(page_data["text"])
            embeddings_list.append(embedding)
            
            # Store metadata
            self.metadata.append(page_data)
        
        # Convert list of embeddings to numpy array
        embeddings_array = np.array(embeddings_list, dtype='float32')
        
        # Step 3: Add to FAISS index
        print("💾 Storing in vector database...")
        self.index.add(embeddings_array)
        
        # Save index to disk
        self._save_index()
        
        return len(pages_data)
    
    def query(self, query_text: str, top_k: int = 3) -> List[Dict[str, any]]:
        """
        Search the vector database for pages most relevant to the query.
        
        Args:
            query_text: Search query
            top_k: Number of top results to return
            
        Returns:
            List of dictionaries containing matched pages and metadata
        """
        print(f"\n Searching for: '{query_text}'")
        
        if self.index.ntotal == 0:
            print("⚠ Index is empty. Please index some documents first.")
            return []
        
        # Generate embedding for the query
        query_embedding = self.generate_embedding(query_text)
        query_embedding = query_embedding.reshape(1, -1)  # FAISS expects 2D array
        
        # Search in FAISS
        distances, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))
        
        # Format results
        matched_pages = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx >= 0 and idx < len(self.metadata):  # Valid index
                metadata = self.metadata[idx]
                matched_page = {
                    "rank": i + 1,
                    "document_id": metadata["document_id"],
                    "page_number": metadata["page_number"],
                    "text": metadata["text"],
                    "distance": float(distance)  # L2 distance (lower is better)
                }
                matched_pages.append(matched_page)
        
        return matched_pages
    
    def reset_index(self):
        """
        Clear all data from the current index.
        Useful for re-indexing or testing.
        """
        self.index = faiss.IndexFlatL2(self.embedding_dimension)
        self.metadata = []
        
        # Remove saved files if they exist
        if os.path.exists(f"{self.index_path}.index"):
            os.remove(f"{self.index_path}.index")
        if os.path.exists(f"{self.index_path}.metadata"):
            os.remove(f"{self.index_path}.metadata")
        
        print("✓ Index reset")

    def extract_page_as_image(self, document_id: str, page_number: int, output_dir: str = "output_images", zoom: int = 2) -> str:
        """
        Extracts a specific page from a PDF as a high-resolution image.

        Args:
            document_id: The filename of the PDF document.
            page_number: The 1-indexed page number to extract.
            output_dir: The directory to save the image in.
            zoom: The zoom factor for rendering (higher zoom = higher resolution).

        Returns:
            The path to the saved image file, or an empty string on failure.
        """
        pdf_path = os.path.join("pdfs", document_id)
        if not os.path.exists(pdf_path):
            print(f"❌ Error: PDF file not found at {pdf_path}")
            return ""

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        try:
            pdf_document = fitz.open(pdf_path)
            # page_number is 1-indexed, but PyMuPDF is 0-indexed
            page = pdf_document.load_page(page_number - 1)

            # Render page to a pixmap using a zoom factor for higher resolution
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)

            # Define output path
            output_filename = f"{os.path.splitext(document_id)[0]}_page_{page_number}.png"
            output_path = os.path.join(output_dir, output_filename)
            
            # Save the image
            pix.save(output_path)
            
            pdf_document.close()
            return output_path

        except Exception as e:
            print(f"❌ Error extracting image from page {page_number}: {e}")
            return ""