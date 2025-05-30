"""
Module for loading and processing PDF documents.
"""
import os
from typing import List, Dict, Any
from pathlib import Path
from tqdm import tqdm

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

import config

class DocumentProcessor:
    """Class for loading and processing documents."""
    
    def __init__(self, docs_dir: Path = config.DOCS_DIR):
        """Initialize the document processor.
        
        Args:
            docs_dir: Directory containing the documents to process.
        """
        self.docs_dir = docs_dir
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            length_function=len
        )
    
    def get_pdf_files(self) -> List[Path]:
        """Get all PDF files in the docs directory.
        
        Returns:
            List of paths to PDF files.
        """
        return [f for f in self.docs_dir.glob("*.pdf")]
    
    def load_document(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load and process a single document.
        
        Args:
            file_path: Path to the document.
            
        Returns:
            List of document chunks with metadata.
        """
        try:
            loader = PyPDFLoader(str(file_path))
            documents = loader.load()
            
            # Add metadata
            for doc in documents:
                doc.metadata["source"] = file_path.name
                doc.metadata["title"] = file_path.stem
            
            # Split documents into chunks
            chunks = self.text_splitter.split_documents(documents)
            
            print(f"Processed {file_path.name}: {len(documents)} pages, {len(chunks)} chunks")
            return chunks
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return []
    
    def load_all_documents(self) -> List[Dict[str, Any]]:
        """Load and process all PDF documents in the docs directory.
        
        Returns:
            List of all document chunks with metadata.
        """
        pdf_files = self.get_pdf_files()
        all_chunks = []
        
        print(f"Found {len(pdf_files)} PDF files to process")
        for pdf_file in tqdm(pdf_files, desc="Processing documents"):
            chunks = self.load_document(pdf_file)
            all_chunks.extend(chunks)
        
        print(f"Total chunks created: {len(all_chunks)}")
        return all_chunks 