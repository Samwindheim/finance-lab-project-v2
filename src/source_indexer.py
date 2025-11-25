"""
The SourceIndexer class is designed for an "event-centric" approach to data extraction.

It handles:
- Finding all documents (PDFs and HTML) associated with a single issue_id.
- Extracting text from all these sources and combining them into a single corpus.
- Building a unified FAISS vector index from the combined text.
- Storing rich metadata for each text chunk, preserving the original source document.
- Querying this unified index to find the most relevant context for an extraction query.
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
from html_processor import extract_text_from_html

# Load environment variables
load_dotenv()

class SourceIndexer:
    """
    Class for creating and querying a unified index for all documents of a single issue.
    """
    
    def __init__(self, issue_id: str):
        """
        Initialize the SourceIndexer.
        
        Args:
            issue_id: The unique identifier for the financial event.
        """
        self.issue_id = issue_id
        self.index_path = os.path.join(config.FAISS_INDEX_DIR, f"{self.issue_id}")
        
        # Initialize OpenAI client for embeddings
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            timeout=30.0,
            max_retries=2,
        )
        self.embedding_model = config.EMBEDDING_MODEL
        self.embedding_dimension = config.EMBEDDING_DIMENSION
        self.chunk_size = config.CHUNK_SIZE
        self.chunk_overlap = config.CHUNK_OVERLAP
        self.embedding_batch_size = config.EMBEDDING_BATCH_SIZE
        self.tokenizer = tiktoken.encoding_for_model(config.EMBEDDING_MODEL)
        
        # Initialize FAISS index
        self.index = faiss.IndexFlatL2(self.embedding_dimension)
        self.metadata = []
        
        self._load_index()

    def _load_index(self):
        """Load existing FAISS index and metadata if available."""
        if os.path.exists(f"{self.index_path}.index") and os.path.exists(f"{self.index_path}.metadata"):
            try:
                self.index = faiss.read_index(f"{self.index_path}.index")
                with open(f"{self.index_path}.metadata", "rb") as f:
                    self.metadata = pickle.load(f)
                print(f"\nLoaded existing index for issue {self.issue_id} with {len(self.metadata)} documents")
            except Exception as e:
                print(f"\nCould not load existing index for issue {self.issue_id}: {e}")

    def _save_index(self):
        """Save FAISS index and metadata to disk."""
        os.makedirs(config.FAISS_INDEX_DIR, exist_ok=True)
        faiss.write_index(self.index, f"{self.index_path}.index")
        with open(f"{self.index_path}.metadata", "wb") as f:
            pickle.dump(self.metadata, f)
        print(f"\nSaved index for issue {self.issue_id}")

    def _chunk_text(self, text: str, source_document: str, source_type: str, page_number: int = None) -> List[Dict]:
        """Splits a long text into smaller, overlapping chunks with rich metadata."""
        if not text:
            return []
        
        tokens = self.tokenizer.encode(text)
        chunks = []
        start_index = 0
        while start_index < len(tokens):
            end_index = start_index + self.chunk_size
            token_chunk = tokens[start_index:end_index]
            chunk_text = self.tokenizer.decode(token_chunk)
            
            chunk_data = {
                "issue_id": self.issue_id,
                "source_document": source_document,
                "source_type": source_type,
                "page_number": page_number, # Can be None for HTML
                "text": chunk_text,
            }
            chunks.append(chunk_data)
            start_index += self.chunk_size - self.chunk_overlap
            
        return chunks

    def index_issue(self, pdf_sources: List[Dict], html_sources: List[Dict]):
        """
        Builds a unified index for all documents related to the issue_id.
        """
        print(f"\nBuilding unified index for issue: {self.issue_id}")
        
        all_chunks = []
        base_dir = config.BASE_DIR

        # --- Process PDFs ---
        for pdf_source in pdf_sources:
            pdf_filename = pdf_source.get("source_url")
            pdf_path = os.path.join(base_dir, 'pdfs', pdf_filename)
            if not os.path.exists(pdf_path):
                print(f"Warning: PDF file not found at {pdf_path}. Skipping.")
                continue
            
            print(f"  - Processing PDF: {pdf_filename}")
            pdf_document = fitz.open(pdf_path)
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                page_text = page.get_text("text").strip()
                if page_text:
                    all_chunks.extend(self._chunk_text(page_text, pdf_filename, "PDF", page_number=page_num + 1))
            pdf_document.close()
            
            all_chunks.append(self._chunk_text(f"\n--- END OF DOCUMENT: {pdf_filename} ---\n", pdf_filename, "SEPARATOR")[0])

        # --- Process HTMLs ---
        for html_source in html_sources:
            html_url = html_source.get("source_url")
            print(f"  - Processing HTML: {html_url}")
            text = extract_text_from_html(html_url, preserve_tables=True)
            
            if text:
                # Create ONE single embedding for the HTML document
                # Truncate to 8191 tokens (limit for text-embedding-3-large)
                tokens = self.tokenizer.encode(text)
                if len(tokens) > 8191:
                    print(f"    - Truncating HTML text to 8191 tokens (original: {len(tokens)})")
                    text = self.tokenizer.decode(tokens[:8191])
                
                chunk_data = {
                    "issue_id": self.issue_id,
                    "source_document": html_url,
                    "source_type": "HTML",
                    "page_number": None,
                    "text": text,
                }
                all_chunks.append(chunk_data)

        if not all_chunks:
            print("No text extracted from any source. Aborting index creation.")
            return

        # --- Generate embeddings in batches ---
        embeddings_list = []
        for i in tqdm(range(0, len(all_chunks), self.embedding_batch_size), desc="Generating embeddings", unit="batch"):
            batch_chunks = all_chunks[i:i + self.embedding_batch_size]
            batch_texts = [chunk['text'] for chunk in batch_chunks]
            
            try:
                response = self.client.embeddings.create(
                    model=self.embedding_model,
                    input=batch_texts,
                    dimensions=self.embedding_dimension
                )
                batch_embeddings = [np.array(e.embedding, dtype='float32') for e in response.data]
                embeddings_list.extend(batch_embeddings)
                self.metadata.extend(batch_chunks)
            except Exception as e:
                print(f"\nError generating embeddings for batch: {e}")
                return

        # --- Add to FAISS and save ---
        if embeddings_list:
            embeddings_array = np.array(embeddings_list, dtype='float32')
            self.index.add(embeddings_array)
            self._save_index()

    def query(self, query_text: str, top_k: int = 5) -> List[Dict]:
        """Queries the unified index."""
        if self.index.ntotal == 0:
            return []
            
        query_embedding = np.array(self.client.embeddings.create(model=self.embedding_model, input=query_text, dimensions=self.embedding_dimension).data[0].embedding, dtype='float32').reshape(1, -1)
        distances, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0:
                meta = self.metadata[idx]
                results.append({
                    "distance": float(dist),
                    "source_document": meta.get("source_document"),
                    "source_type": meta.get("source_type"),
                    "page_number": meta.get("page_number"),
                    "text": meta.get("text")
                })
        return results

    def extract_page_as_image(self, pdf_filename: str, page_number: int, zoom: int = 2) -> str | None:
        """
        Extracts a specific page from a PDF as a high-resolution image.
        This is a utility function that can be called after a query.
        """
        pdf_path = os.path.join(config.BASE_DIR, 'pdfs', pdf_filename)
        if not os.path.exists(pdf_path):
            print(f"\nWarning: PDF file not found at {pdf_path} for image extraction.")
            return None

        output_dir = config.OUTPUT_IMAGE_DIR
        os.makedirs(output_dir, exist_ok=True)

        try:
            pdf_document = fitz.open(pdf_path)
            # page_number is 1-indexed from our metadata
            page = pdf_document.load_page(page_number - 1)

            # Render page to a pixmap using a zoom factor for higher resolution
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)

            # Define output path
            output_filename = f"{os.path.splitext(pdf_filename)[0]}_page_{page_number}.png"
            output_path = os.path.join(output_dir, output_filename)
            
            pix.save(output_path)
            pdf_document.close()
            return output_path
        except Exception as e:
            print(f"\nError extracting image from page {page_number} of {pdf_filename}: {e}")
            return None
