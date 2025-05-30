"""
Module for indexing documents.
"""
import os
import time
import logging
import hashlib
import pickle
import concurrent.futures
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

from src.retrieval.vector_store import VectorStore
import config

# Configure logging
logger = logging.getLogger(__name__)

# Cache for document hashes to detect changes
DOCUMENT_HASHES = {}

def compute_document_hash(file_path: str) -> str:
    """Compute a hash of a document file.
    
    Args:
        file_path: Path to the document file.
        
    Returns:
        Hash of the document file.
    """
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def load_document_hashes() -> Dict[str, str]:
    """Load document hashes from disk.
    
    Returns:
        Dictionary of document paths to hashes.
    """
    hash_file = Path(config.VECTOR_DB_DIR) / "document_hashes.pkl"
    if hash_file.exists():
        try:
            with open(hash_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"Could not load document hashes: {e}")
    return {}

def save_document_hashes(hashes: Dict[str, str]) -> None:
    """Save document hashes to disk.
    
    Args:
        hashes: Dictionary of document paths to hashes.
    """
    hash_file = Path(config.VECTOR_DB_DIR) / "document_hashes.pkl"
    try:
        # Create parent directory if it doesn't exist
        hash_file.parent.mkdir(parents=True, exist_ok=True)
        with open(hash_file, 'wb') as f:
            pickle.dump(hashes, f)
    except Exception as e:
        logger.warning(f"Could not save document hashes: {e}")

def process_pdf(file_path: str) -> List[Dict[str, Any]]:
    """Process a PDF file and split it into chunks.
    
    Args:
        file_path: Path to the PDF file.
        
    Returns:
        List of document chunks.
    """
    try:
        logger.info(f"Processing PDF: {file_path}")
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        # Add file path to metadata
        for doc in documents:
            doc.metadata['source'] = file_path
            
        return documents
    except Exception as e:
        logger.error(f"Error processing PDF {file_path}: {e}")
        return []

def index_documents(force_reindex: bool = False) -> Optional[VectorStore]:
    """Index documents in the documents directory.
    
    Args:
        force_reindex: Whether to force reindexing of all documents.
        
    Returns:
        Vector store with indexed documents, or None if indexing failed.
    """
    try:
        start_time = time.time()
        logger.info("Starting document indexing...")
        
        # Create vector store directory if it doesn't exist
        os.makedirs(config.VECTOR_DB_DIR, exist_ok=True)
        
        # Check if we need to reindex
        if not force_reindex and Path(config.VECTOR_DB_DIR).exists() and any(Path(config.VECTOR_DB_DIR).glob("*")):
            # Load document hashes
            global DOCUMENT_HASHES
            DOCUMENT_HASHES = load_document_hashes()
            
            # Check if any documents have changed
            docs_dir = Path(config.DOCS_DIR)
            if docs_dir.exists():
                changed = False
                for pdf_file in docs_dir.glob("**/*.pdf"):
                    file_path = str(pdf_file)
                    current_hash = compute_document_hash(file_path)
                    if file_path not in DOCUMENT_HASHES or DOCUMENT_HASHES[file_path] != current_hash:
                        changed = True
                        break
                
                if not changed:
                    logger.info("No documents have changed, loading existing vector store...")
                    vector_store = VectorStore()
                    if vector_store.load_vector_store():
                        elapsed_time = time.time() - start_time
                        logger.info(f"Loaded existing vector store in {elapsed_time:.2f} seconds")
                        return vector_store
            
        # Initialize vector store
        vector_store = VectorStore()
        
        # Load documents in parallel
        all_documents = []
        docs_dir = Path(config.DOCS_DIR)
        
        if not docs_dir.exists():
            logger.warning(f"Documents directory {docs_dir} does not exist.")
            return None
        
        # Get all PDF files
        pdf_files = list(docs_dir.glob("**/*.pdf"))
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {docs_dir}")
            return None
            
        logger.info(f"Found {len(pdf_files)} PDF files")
        
        # Process PDFs in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(os.cpu_count() or 1, 4)) as executor:
            future_to_file = {executor.submit(process_pdf, str(pdf_file)): str(pdf_file) for pdf_file in pdf_files}
            for future in concurrent.futures.as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    documents = future.result()
                    all_documents.extend(documents)
                    # Update document hash
                    DOCUMENT_HASHES[file_path] = compute_document_hash(file_path)
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
        
        # Save document hashes
        save_document_hashes(DOCUMENT_HASHES)
        
        if not all_documents:
            logger.warning("No documents loaded.")
            return None
            
        logger.info(f"Loaded {len(all_documents)} document pages")
        
        # Split documents into chunks
        logger.info("Splitting documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""],
            length_function=len
        )
        
        # Process in batches to avoid memory issues
        batch_size = 100
        all_chunks = []
        
        for i in range(0, len(all_documents), batch_size):
            batch = all_documents[i:min(i+batch_size, len(all_documents))]
            chunks = text_splitter.split_documents(batch)
            all_chunks.extend(chunks)
            logger.info(f"Processed batch {i//batch_size + 1}/{(len(all_documents)-1)//batch_size + 1}, created {len(chunks)} chunks")
        
        logger.info(f"Created {len(all_chunks)} chunks from {len(all_documents)} documents")
        
        # Create vector store
        vector_store.create_vector_store(all_chunks)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Document indexing completed in {elapsed_time:.2f} seconds")
        
        return vector_store
        
    except Exception as e:
        logger.error(f"Error indexing documents: {e}", exc_info=True)
        return None 