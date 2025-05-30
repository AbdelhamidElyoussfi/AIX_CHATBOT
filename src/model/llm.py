"""
Module for working with the Granite 3.3 2B Instruct LLM.
"""
import torch
import hashlib
import functools
from typing import Dict, List, Any, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
import re
import os
from pathlib import Path
import time
import logging
import json

import config

logger = logging.getLogger(__name__)

# Create a simple in-memory cache with LRU behavior
class LRUCache:
    def __init__(self, capacity=100):
        self.capacity = capacity
        self.cache = {}
        self.order = []
        
    def get(self, key):
        if key in self.cache:
            # Move to the end to mark as recently used
            self.order.remove(key)
            self.order.append(key)
            return self.cache[key]
        return None
        
    def put(self, key, value):
        if key in self.cache:
            self.order.remove(key)
        elif len(self.cache) >= self.capacity:
            # Remove least recently used
            oldest = self.order.pop(0)
            del self.cache[oldest]
            
        self.cache[key] = value
        self.order.append(key)

class GenerationCancelled(Exception):
    """Exception raised when generation is cancelled."""
    pass

class LLMModel:
    """Class for interacting with the Granite 3.3 2B Instruct LLM."""
    
    def __init__(
        self,
        model_dir: str = config.MODEL_DIR,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_new_tokens: int = config.MAX_NEW_TOKENS,
        temperature: float = config.TEMPERATURE,
        top_p: float = config.TOP_P,
        repetition_penalty: float = config.REPETITION_PENALTY
    ):
        """Initialize the LLM.
        
        Args:
            model_dir: Path to the model directory from config.
            device: Device to run the model on.
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Temperature for generation.
            top_p: Top-p for nucleus sampling.
            repetition_penalty: Penalty for repeating tokens.
        """
        self.model_dir = model_dir
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        
        print(f"Loading model from {model_dir}...")
        
        # Load tokenizer from model directory
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        
        # Load model with memory optimizations
        if torch.cuda.is_available():
            print(f"Loading model in 8-bit on {device}...")
            self.model = AutoModelForCausalLM.from_pretrained(
                str(model_dir),
                load_in_8bit=True,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        else:
            print(f"Loading model on {device}...")
            self.model = AutoModelForCausalLM.from_pretrained(
                str(model_dir),
                low_cpu_mem_usage=True,
                torch_dtype=torch.float32
            )
            self.model.to(device)
        
        # Create text generation pipeline
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        print("Model loaded successfully")
        
        # Initialize response cache
        self.response_cache = LRUCache(capacity=200)
        # Keep track of token counts for optimization
        self.token_counts = {}
    
    def format_context(self, retrieved_docs: List[tuple]) -> str:
        """Format retrieved documents into a context string.
        
        Args:
            retrieved_docs: List of retrieved documents and their scores.
            
        Returns:
            Formatted context string.
        """
        context_parts = []
        for i, (doc, score) in enumerate(retrieved_docs):
            context_parts.append(f"Document {i+1}: {doc.page_content}")
        
        return "\n\n".join(context_parts)
    
    def generate_text(self, prompt: str) -> str:
        """Generate text from a prompt with caching."""
        # Check cache first
        cache_key = simple_hash(prompt)
        if cache_key in RESPONSE_CACHE:
            print("Using cached response")
            return RESPONSE_CACHE[cache_key]
            
        try:
            # Generate with reduced parameters for speed
            result = self.pipe(
                prompt,
                max_new_tokens=min(128, self.max_new_tokens),  # Use smaller value for warmup
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty,
                return_full_text=False
            )
            
            # Get generated text
            response = result[0]['generated_text'].strip()
            
            # Cache the response
            RESPONSE_CACHE[cache_key] = response
            
            return response
        except Exception as e:
            print(f"Error generating text: {e}")
            return f"Error generating response: {str(e)}"
    
    def _create_cache_key(self, query: str, context: List[str]) -> str:
        """Create a cache key from query and context."""
        # Create a deterministic representation of the context
        context_str = "".join(sorted([c[:100] for c in context]))
        
        # Create a hash of the query and context
        key_material = f"{query}::{context_str}"
        return hashlib.md5(key_material.encode()).hexdigest()
    
    def _update_token_stats(self, query: str, token_count: int):
        """Update token usage statistics for optimization."""
        # Store the first few chars as the key
        key = query[:20]
        self.token_counts[key] = token_count
    
    def _get_optimal_max_tokens(self, query: str) -> int:
        """Dynamically determine optimal token count based on query patterns."""
        # Default to config value
        max_tokens = config.MAX_NEW_TOKENS
        
        # Check if we have similar queries to optimize token count
        query_start = query[:20]
        similar_queries = [k for k in self.token_counts.keys() if k.startswith(query_start[:10])]
        
        if similar_queries:
            # Use the average token count of similar queries plus a margin
            avg_tokens = sum(self.token_counts[k] for k in similar_queries) / len(similar_queries)
            # Add 20% margin and round up
            max_tokens = min(config.MAX_NEW_TOKENS, int(avg_tokens * 1.2))
        
        return max_tokens
    
    def generate_rag_response(
        self, 
        query: str, 
        retrieved_docs: List[str],
        task_id: Optional[str] = None
    ) -> str:
        """Generate a response using RAG."""
        try:
            # Create context from retrieved documents
            # Fix: Handle both tuple format (doc, score) and direct document objects
            context = []
            for doc_item in retrieved_docs:
                if isinstance(doc_item, tuple) and len(doc_item) >= 1:
                    # Handle (document, score) tuple format
                    doc = doc_item[0]
                    if hasattr(doc, 'page_content'):
                        context.append(doc.page_content)
                elif hasattr(doc_item, 'page_content'):
                    # Handle direct document object
                    context.append(doc_item.page_content)
                else:
                    # Fallback for other formats
                    logger.warning(f"Unexpected document format: {type(doc_item)}")
                    if isinstance(doc_item, str):
                        context.append(doc_item)
            
            # If no valid context was extracted, return error message
            if not context:
                logger.error("Could not extract content from retrieved documents")
                return "I'm sorry, but I couldn't process the relevant information to answer your question."
            
            # Check cache first
            cache_key = self._create_cache_key(query, context)
            cached_response = self.response_cache.get(cache_key)
            if cached_response:
                logger.info("Using cached response")
                return cached_response
            
            # Format prompt with system template
            prompt = config.SYSTEM_PROMPT_TEMPLATE.format(
                context="\n\n".join(context),
                    question=query
                )
            
            # Get optimal token count based on query patterns
            max_tokens = self._get_optimal_max_tokens(query)
            
            # Generate response with optimized parameters
            start_time = time.time()
            
            # Use early stopping for faster generation
            response = self.pipe(
                    prompt,
                max_new_tokens=max_tokens,
                temperature=config.TEMPERATURE,
                top_p=config.TOP_P,
                repetition_penalty=config.REPETITION_PENALTY,
                    do_sample=True,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id
            )[0]["generated_text"]
            
            elapsed_time = time.time() - start_time
            
            # Extract response (remove the prompt)
            if response.startswith(prompt):
                response = response[len(prompt):].strip()
            
            # Update token stats for future optimization
            token_count = len(self.tokenizer.encode(response))
            self._update_token_stats(query, token_count)
            
            logger.info(f"Response generated in {elapsed_time:.2f} seconds ({token_count} tokens)")
                
                # Cache the response
            self.response_cache.put(cache_key, response)
            
            return response
                
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}", exc_info=True)
            raise
    
    def _format_response(self, response: str, query: str) -> str:
        """Post-process the response to ensure proper formatting."""
        try:
            # Remove document references
            response = self._remove_references(response)
            
            # Add appropriate headers and structure
            response = self._add_structure(response, query)
            
            # Format code blocks
            response = self._format_code_blocks(response)
            
            # Ensure proper spacing
            response = self._fix_spacing(response)
            
            # Final check to ensure we have content
            if not response or not response.strip():
                print("Warning: Empty response after formatting, using original response")
                return "I apologize, but I couldn't process your query about DLPAR properly. DLPAR (Dynamic Logical Partitioning) is a feature in AIX systems that allows for dynamic resource allocation. Please try asking a more specific question about DLPAR functionality."
            
            return response
        except Exception as e:
            print(f"Error in response formatting: {str(e)}")
            import traceback
            traceback.print_exc()
            # Return a simplified response when formatting fails
            return "I apologize, but I encountered an error while formatting my response. DLPAR (Dynamic Logical Partitioning) allows for dynamic resource allocation in AIX systems. Please try asking a more specific question."
    
    def _remove_references(self, text: str) -> str:
        """Remove document references from the response."""
        # Remove phrases like "According to Document X" or "From Document X"
        text = text.replace("According to Document", "")
        text = text.replace("From Document", "")
        text = text.replace("Based on Document", "")
        text = text.replace("As mentioned in Document", "")
        
        # Remove specific document number references
        text = re.sub(r"Document \d+:?\s*", "", text)
        text = re.sub(r"Document \d+\s*states that\s*", "", text)
        text = re.sub(r"In Document \d+,?\s*", "", text)
        text = re.sub(r"\(Document \d+\)", "", text)
        text = re.sub(r"\[Document \d+\]", "", text)
        
        return text
    
    def _add_structure(self, text: str, query: str) -> str:
        """Add headers and improve structure of the response."""
        lines = text.split("\n")
        structured_text = []
        
        # Check what type of query we're answering
        is_how_to = any(kw in query.lower() for kw in ["how to", "how do i", "steps", "guide", "instructions"])
        is_what_is = any(kw in query.lower() for kw in ["what is", "what are", "explain", "describe"])
        is_comparison = any(kw in query.lower() for kw in ["compare", "difference", "versus", "vs", "similarities"])
        
        # Add an appropriate H1 header if none exists
        if not any(line.startswith("# ") for line in lines):
            if is_how_to:
                title = f"# How to {query.replace('how to', '').replace('?', '').strip().capitalize()}"
                structured_text.append(title)
            elif is_what_is:
                topic = query.replace("what is", "").replace("what are", "").replace("?", "").strip().capitalize()
                title = f"# {topic} Overview"
                structured_text.append(title)
            elif is_comparison:
                title = f"# Comparison: {query.replace('compare', '').replace('?', '').strip().capitalize()}"
                structured_text.append(title)
            else:
                title = f"# AIX Systems Information"
                structured_text.append(title)
                
            structured_text.append("")  # Add blank line after title
        
        # Process existing content and enhance structure
        in_list = False
        list_type = None
        has_steps_section = False
        has_summary = False
        
        for i, line in enumerate(lines):
            # Skip the line if it's a reference
            if "Document" in line and any(ref in line for ref in ["According to", "states that", "mentioned in"]):
                continue
                
            # Handle headers (transform or preserve)
            if line.startswith("# "):
                structured_text.append(line)
            elif line.startswith("## "):
                structured_text.append(line)
            elif line.startswith("### "):
                structured_text.append(line)
            # Create subheaders from ALL CAPS text
            elif line.isupper() and len(line) > 5:
                structured_text.append(f"## {line.capitalize()}")
            # Check for potential section headers (short phrases followed by colon)
            elif ":" in line and len(line.split(":")[0].split()) <= 3 and len(line) < 50:
                header = line.split(":")[0].strip()
                content = line.split(":", 1)[1].strip()
                structured_text.append(f"## {header}")
                if content:
                    structured_text.append("")
                    structured_text.append(content)
            # Handle numbered lists
            elif re.match(r"^\d+\.\s", line):
                if not in_list or list_type != "numbered":
                    if not has_steps_section and is_how_to:
                        structured_text.append("## Steps")
                        structured_text.append("")
                        has_steps_section = True
                    in_list = True
                    list_type = "numbered"
                structured_text.append(line)
            # Handle bullet lists
            elif re.match(r"^[-*•]\s", line):
                if not in_list or list_type != "bullet":
                    in_list = True
                    list_type = "bullet"
                structured_text.append(line)
            # Regular text
            else:
                if in_list and line.strip():
                    # This is continuation of a list item with indent
                    structured_text.append("  " + line)
                else:
                    if in_list and not line.strip():
                        # End of list
                        in_list = False
                        list_type = None
                        structured_text.append("")
                    
                    # Add line normally
                    structured_text.append(line)
        
        # Add a summary section at the end for how-to content if none exists
        if is_how_to and not has_summary and len(structured_text) > 5:
            structured_text.append("")
            structured_text.append("## Summary")
            structured_text.append("")
            structured_text.append("Following these steps will help you successfully complete this AIX operation. If you encounter any issues, check system logs for detailed error messages.")
        
        return "\n".join(structured_text)
    
    def _format_code_blocks(self, text: str) -> str:
        """Improve code block formatting."""
        # Replace inline code with better formatting
        text = re.sub(r'`([^`\n]+)`', r'`\1`', text)
        
        # Replace AIX commands with properly formatted code
        command_pattern = r'(?<!\`)((?:sudo\s+)?(?:smitty|lsdev|cfgmgr|mksysb|installp|lslpp|mkszfile|lspv|chdev|rsh|bosboot|reboot|shutdown|halt|lsvg|extendvg|snapshot|dspmsg|odmget|odmchange|lsattr|errpt|diag)\s+[^\n,.;]+)(?![\`])'
        text = re.sub(command_pattern, r'`\1`', text)
        
        return text
    
    def _fix_spacing(self, text: str) -> str:
        """Fix spacing issues for better readability."""
        # Remove redundant blank lines (more than 2 in a row)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Ensure blank line after headers
        text = re.sub(r'(#[^#\n]+)\n([^#\n])', r'\1\n\n\2', text)
        
        # Ensure lists have proper spacing
        text = re.sub(r'(\n\d+\.[^\n]+)\n(\d+\.)', r'\1\n\n\2', text)
        
        # Ensure proper spacing around code blocks
        text = re.sub(r'([^\n])\n`', r'\1\n\n`', text)
        text = re.sub(r'`\n([^\n])', r'`\n\n\1', text)
        
        return text