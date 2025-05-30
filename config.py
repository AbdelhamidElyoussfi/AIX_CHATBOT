"""
Configuration settings for the AIX Systems RAG Chatbot.
"""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DOCS_DIR = BASE_DIR / "Docs"
MODEL_DIR = BASE_DIR / "granite-3.3-2b-instruct"
DATA_DIR = BASE_DIR / "data"
VECTOR_DB_DIR = DATA_DIR / "vector_db"
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"
USER_DATA_DIR = DATA_DIR / "users"

# Create necessary directories
for directory in [DATA_DIR, VECTOR_DB_DIR, STATIC_DIR, TEMPLATES_DIR, USER_DATA_DIR]:
    os.makedirs(directory, exist_ok=True)

# Flask settings
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
FLASK_DEBUG = False
FLASK_SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "your-secret-key-here")

# Model settings - optimized for speed
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Fast, lightweight embedding model
TOP_K_RETRIEVAL = 2  # Reduced from 3 for faster retrieval and processing

# Document processing settings - optimized for speed and quality
CHUNK_SIZE = 800  # Smaller chunks for faster processing
CHUNK_OVERLAP = 150  # Reduced overlap for faster processing while maintaining context

# LLM Generation settings - optimized for speed
MAX_NEW_TOKENS = 400  # Reduced from 512 for faster generation
TEMPERATURE = 0.6  # Reduced from 0.7 for more deterministic (faster) responses
TOP_P = 0.9  # Reduced from 0.95 for faster generation
REPETITION_PENALTY = 1.05  # Reduced from 1.1 for faster generation

# Cache settings
ENABLE_RESPONSE_CACHE = True
CACHE_SIZE = 500
EMBEDDING_CACHE_SIZE = 1000

# System prompt template for RAG responses - optimized for faster responses
SYSTEM_PROMPT_TEMPLATE = """
You are an AIX Systems Assistant, an AI expert on IBM AIX operating systems. Keep responses concise and direct.

CONTEXT:
{context}

FORMATTING:
- Use numbered lists for steps
- Use bullet points for key information
- Be concise and technically accurate

Question: {question}
"""

# UI Configuration
UI_TITLE = "AIX Systems Assistant"
UI_DESCRIPTION = "Your AI expert on IBM AIX systems and PowerHA"
UI_INITIAL_MESSAGE = "Hello! I'm your AIX Systems Assistant. How can I help you today?"
