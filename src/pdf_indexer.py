"""
PDF Indexer for RAG Pipeline - Text Extraction and Vector Storage
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
from tqdm import tqdm
import tiktoken
import config

# Load environment variables
load_dotenv()


class PDFIndexer:
    """
    Main class for indexing PDF documents into a vector database.
    Handles text extraction, embedding generation, and storage.
    """
    
    def __init__(self, index_path: str = "faiss_index/index"):
        """
        Initialize the PDFIndexer with OpenAI client and FAISS index.
        
        Args:
            index_path: Path to save/load the FAISS index and metadata
        """
        # Initialize OpenAI client for embeddings
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            timeout=30.0,  # Set a 30-second timeout for API requests
            max_retries=2, # Retry up to 2 times on failure
        )
        self.embedding_model = config.EMBEDDING_MODEL
        self.embedding_dimension = config.EMBEDDING_DIMENSION
        self.chunk_size = config.CHUNK_SIZE
        self.chunk_overlap = config.CHUNK_OVERLAP
        self.embedding_batch_size = config.EMBEDDING_BATCH_SIZE
        self.tokenizer = tiktoken.encoding_for_model(config.EMBEDDING_MODEL)
        
        # Initialize FAISS index
        self.index_path = index_path
        self.index = faiss.IndexFlatL2(self.embedding_dimension)
        
        # Store metadata for each indexed chunk
        self.metadata = []  # List of dicts with document_id, page_number, text
        
        # Try to load existing index
        self._load_index()
    
    def _load_index(self):
        """Load existing FAISS index and metadata if available."""
        if os.path.exists(f"{self.index_path}.index") and os.path.exists(f"{self.index_path}.metadata"):
            try:
                self.index = faiss.read_index(f"{self.index_path}.index")
                with open(f"{self.index_path}.metadata", "rb") as f:
                    self.metadata = pickle.load(f)
                print(f"\nLoaded existing index with {len(self.metadata)} documents")
            except Exception as e:
                print(f"\nCould not load existing index: {e}")
    
    def _save_index(self):
        """Save FAISS index and metadata to disk."""
        index_dir = os.path.dirname(self.index_path)
        if index_dir:
            os.makedirs(index_dir, exist_ok=True)
            
        faiss.write_index(self.index, f"{self.index_path}.index")
        with open(f"{self.index_path}.metadata", "wb") as f:
            pickle.dump(self.metadata, f)
        print(f"\nSaved index for {self.index_path}")
    
    def _chunk_text(self, text: str) -> List[str]:
        """Splits a long text into smaller, overlapping chunks based on tokens."""
        if not text:
            return []
        
        tokens = self.tokenizer.encode(text)

        chunks = []
        start_index = 0
        while start_index < len(tokens):
            end_index = start_index + self.chunk_size
            token_chunk = tokens[start_index:end_index]
            chunk_text = self.tokenizer.decode(token_chunk)
            chunks.append(chunk_text)
            start_index += self.chunk_size - self.chunk_overlap
            
        return chunks

    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict[str, any]]:
        """
        Extract text from each page of a PDF document and split it into chunks.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of dictionaries containing chunk text and metadata
        """
        all_chunks_data = []
        
        # Open the PDF
        pdf_document = fitz.open(pdf_path)
        document_id = os.path.basename(pdf_path)
        
        # --- Store the absolute path for robustness ---
        absolute_pdf_path = os.path.abspath(pdf_path)
        
        print(f"\nProcessing PDF: {document_id}")
        
        # Iterate through each page
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            
            # Extract text from the page
            text = page.get_text("text").strip()

            if not text:
                continue

            # Split text into chunks
            chunks = self._chunk_text(text)

            # Store each chunk with its metadata
            for i, chunk_text in enumerate(chunks):
                chunk_data = {
                    "document_id": document_id,
                    "pdf_path": absolute_pdf_path,
                    "page_number": page_num + 1,  # 1-indexed for readability
                    "chunk_number": i + 1,
                    "text": chunk_text,
                }
                all_chunks_data.append(chunk_data)
        
        pdf_document.close()
        print(f"\nExtracted {len(all_chunks_data)} text chunks")
        
        return all_chunks_data
    
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
        # If the index is not empty, it implies a re-indexing operation.
        # We clear it to ensure a fresh start.
        if self.index.ntotal > 0:
            print(f"\nIndex already contains data. Clearing for re-indexing...")
            self.reset_index()

        # Step 1: Extract text from all pages
        chunks_data = self.extract_text_from_pdf(pdf_path)
        
        # Step 2: Generate embeddings in batches and add to FAISS
        embeddings_list = []
        
        for i in tqdm(range(0, len(chunks_data), self.embedding_batch_size), desc="Generating embeddings", unit="batch"):
            batch_chunks = chunks_data[i:i + self.embedding_batch_size]
            batch_texts = [chunk['text'] for chunk in batch_chunks]

            try:
                response = self.client.embeddings.create(
                    model=self.embedding_model,
                    input=batch_texts,
                    dimensions=self.embedding_dimension
                )
                batch_embeddings = [np.array(embedding.embedding, dtype='float32') for embedding in response.data]
                
                embeddings_list.extend(batch_embeddings)
                self.metadata.extend(batch_chunks)

            except Exception as e:
                print(f"\n\nError generating embeddings for batch starting at chunk {i}: {e}")
                print("Aborting indexing process. No changes were saved.")
                return 0
        
        # Convert list of embeddings to numpy array
        if not embeddings_list:
            print("\nNo embeddings were generated. Aborting.")
            return 0
            
        embeddings_array = np.array(embeddings_list, dtype='float32')
        
        # Step 3: Add to FAISS index
        print("\nStoring in vector database...")
        self.index.add(embeddings_array)
        
        # Save index to disk
        self._save_index()
        
        return len(chunks_data)
    
    def query(self, query_text: str, top_k: int = 3) -> List[Dict[str, any]]:
        """
        Search the vector database for pages most relevant to the query.
        
        Args:
            query_text: Search query
            top_k: Number of top results to return
            
        Returns:
            List of dictionaries containing matched pages and metadata
        """
        
        if self.index.ntotal == 0:
            print("\nIndex is empty. Please index some documents first.")
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
                    "pdf_path": metadata.get("pdf_path", ""), # Add pdf_path for backward compatibility
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
        
        print("\nIndex reset")

    def extract_page_as_image(self, pdf_path: str, page_number: int, output_dir: str = config.OUTPUT_IMAGE_DIR, zoom: int = 2) -> str:
        """
        Extracts a specific page from a PDF as a high-resolution image.

        Args:
            pdf_path: The path to the PDF document.
            page_number: The 1-indexed page number to extract.
            output_dir: The directory to save the image in.
            zoom: The zoom factor for rendering (higher zoom = higher resolution).

        Returns:
            The path to the saved image file, or an empty string on failure.
        """
        if not os.path.exists(pdf_path):
            print(f"\nError: PDF file not found at {pdf_path}")
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
            document_id = os.path.basename(pdf_path)
            output_filename = f"{os.path.splitext(document_id)[0]}_page_{page_number}.png"
            output_path = os.path.join(output_dir, output_filename)
            
            # Save the image
            pix.save(output_path)
            
            pdf_document.close()
            return output_path

        except Exception as e:
            print(f"\nError extracting image from page {page_number}: {e}")
            return ""