"""
Module for creating and managing vector stores for document retrieval.
"""
import os
import time
import logging
import hashlib
import pickle
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

import config

# Configure logging
logger = logging.getLogger(__name__)

# Cache for query embeddings to avoid recomputing
EMBEDDING_CACHE = {}
# Cache for search results to speed up repeated queries
SEARCH_CACHE = {}
# Maximum cache size
MAX_CACHE_SIZE = 500

class VectorStore:
    """Class for managing document vector stores."""
    
    def __init__(
        self,
        embedding_model_name: str = config.EMBEDDING_MODEL,  # Use the config value
        persist_directory: Path = config.VECTOR_DB_DIR
    ):
        """Initialize the vector store.
        
        Args:
            embedding_model_name: Name of the embedding model to use.
            persist_directory: Directory to persist the vector store.
        """
        self.embedding_model_name = embedding_model_name
        self.persist_directory = persist_directory
        
        # Initialize embeddings with optimized settings
        start_time = time.time()
        logger.info(f"Initializing embeddings model: {embedding_model_name}")
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={"device": "cpu"},  # Keep on CPU for embeddings
            encode_kwargs={"normalize_embeddings": True, "batch_size": 32}  # Batch processing for speed
        )
        
        elapsed_time = time.time() - start_time
        logger.info(f"Embeddings model initialized in {elapsed_time:.2f} seconds")
        
        self.vector_store = None
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Load embedding cache if available
        self._load_embedding_cache()
    
    def _load_embedding_cache(self):
        """Load embedding cache from disk if available."""
        cache_path = Path(self.persist_directory) / "embedding_cache.pkl"
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    global EMBEDDING_CACHE
                    EMBEDDING_CACHE = pickle.load(f)
                logger.info(f"Loaded {len(EMBEDDING_CACHE)} cached embeddings")
            except Exception as e:
                logger.warning(f"Could not load embedding cache: {e}")
    
    def _save_embedding_cache(self):
        """Save embedding cache to disk."""
        # Only save if we have a reasonable number of items
        if len(EMBEDDING_CACHE) > 5:
            cache_path = Path(self.persist_directory) / "embedding_cache.pkl"
            try:
                # Create parent directory if it doesn't exist
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(cache_path, 'wb') as f:
                    pickle.dump(EMBEDDING_CACHE, f)
                logger.info(f"Saved {len(EMBEDDING_CACHE)} cached embeddings")
            except Exception as e:
                logger.warning(f"Could not save embedding cache: {e}")
    
    def _get_cached_embedding(self, query: str) -> Optional[List[float]]:
        """Get cached embedding for a query."""
        # Create a hash of the query for the cache key
        query_hash = hashlib.md5(query.encode()).hexdigest()
        return EMBEDDING_CACHE.get(query_hash)
    
    def _cache_embedding(self, query: str, embedding: List[float]):
        """Cache embedding for a query."""
        # Create a hash of the query for the cache key
        query_hash = hashlib.md5(query.encode()).hexdigest()
        
        # Manage cache size
        if len(EMBEDDING_CACHE) >= MAX_CACHE_SIZE:
            # Remove a random item (simple strategy)
            EMBEDDING_CACHE.pop(next(iter(EMBEDDING_CACHE)))
            
        EMBEDDING_CACHE[query_hash] = embedding
    
    def _get_cached_search_results(self, query: str, k: int) -> Optional[List]:
        """Get cached search results for a query."""
        # Create a hash of the query and k for the cache key
        cache_key = hashlib.md5(f"{query}:{k}".encode()).hexdigest()
        
        if cache_key in SEARCH_CACHE:
            self.cache_hits += 1
            return SEARCH_CACHE[cache_key]
        
        self.cache_misses += 1
        return None
    
    def _cache_search_results(self, query: str, k: int, results: List):
        """Cache search results for a query."""
        # Create a hash of the query and k for the cache key
        cache_key = hashlib.md5(f"{query}:{k}".encode()).hexdigest()
        
        # Manage cache size
        if len(SEARCH_CACHE) >= MAX_CACHE_SIZE:
            # Remove a random item (simple strategy)
            SEARCH_CACHE.pop(next(iter(SEARCH_CACHE)))
            
        SEARCH_CACHE[cache_key] = results
    
    def create_vector_store(self, documents: List[Dict[str, Any]]) -> None:
        """Create a vector store from documents.
        
        Args:
            documents: List of document chunks to index.
        """
        start_time = time.time()
        logger.info(f"Creating vector store with {len(documents)} documents...")
        
        # Process documents in batches for better memory efficiency
        batch_size = 100
        if len(documents) > batch_size:
            # Create with first batch
            first_batch = documents[:batch_size]
            self.vector_store = Chroma.from_documents(
                documents=first_batch,
                embedding=self.embeddings,
                persist_directory=str(self.persist_directory)
            )
            # Add remaining batches
            for i in range(batch_size, len(documents), batch_size):
                batch = documents[i:min(i+batch_size, len(documents))]
                logger.info(f"Adding batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1} ({len(batch)} documents)")
                self.vector_store.add_documents(batch)
                # Persist after each batch to avoid memory issues
                self.vector_store.persist()
        else:
            # Small enough to create in one go
            self.vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=str(self.persist_directory)
            )
        
        # Final persistence
        self.vector_store.persist()
        
        elapsed_time = time.time() - start_time
        logger.info(f"Vector store created and persisted in {elapsed_time:.2f} seconds")
    
    def load_vector_store(self) -> bool:
        """Load an existing vector store.
        
        Returns:
            True if the vector store was loaded successfully, False otherwise.
        """
        try:
            start_time = time.time()
            logger.info(f"Loading vector store from {self.persist_directory}")
            
            self.vector_store = Chroma(
                persist_directory=str(self.persist_directory),
                embedding_function=self.embeddings
            )
            
            elapsed_time = time.time() - start_time
            logger.info(f"Vector store loaded in {elapsed_time:.2f} seconds")
            return True
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            return False
    
    def similarity_search(self, query: str, k: int = config.TOP_K_RETRIEVAL) -> List:
        """Perform similarity search on the vector store.
        
        Args:
            query: Query string.
            k: Number of results to return.
            
        Returns:
            List of document chunks with similarity scores.
        """
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Call create_vector_store or load_vector_store first.")
        
        start_time = time.time()
        
        # Check cache first
        cached_results = self._get_cached_search_results(query, k)
        if cached_results is not None:
            logger.info(f"Using cached search results for query: {query[:50]}...")
            return cached_results
        
        # Get cached embedding or compute new one
        cached_embedding = self._get_cached_embedding(query)
        
        if cached_embedding is not None:
            # Use cached embedding for search
            logger.info("Using cached embedding for search")
            results = self.vector_store.similarity_search_by_vector_with_relevance_scores(
                cached_embedding,
                k=k
            )
        else:
            # Compute new embedding and search
            results = self.vector_store.similarity_search_with_relevance_scores(query, k=k)
            
            # Cache the embedding for future use
            try:
                # Get the embedding that was just computed
                if hasattr(self.embeddings, "_embeddings_cache"):
                    # Some embedding models have a cache
                    embedding = self.embeddings._embeddings_cache.get(query)
                    if embedding is not None:
                        self._cache_embedding(query, embedding)
            except Exception as e:
                logger.warning(f"Could not cache embedding: {e}")
        
        # Sort by score descending (most similar first)
        sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
        # Always return top K, even if scores are negative
        top_results = sorted_results[:k]
        
        # Cache the search results
        self._cache_search_results(query, k, top_results)
        
        # Periodically save the embedding cache
        if len(EMBEDDING_CACHE) % 10 == 0:
            self._save_embedding_cache()
        
        elapsed_time = time.time() - start_time
        logger.info(f"Search completed in {elapsed_time:.2f} seconds, found {len(top_results)} documents")
        
        # Log similarity scores for debugging
        logger.debug("Similarity scores: %s", [score for _, score in top_results])
        
        return top_results