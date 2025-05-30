"""
Main application entry point for the AIX Systems RAG Chatbot.
"""
import os
import json
from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, url_for, session, flash
from flask_login import LoginManager, login_required, current_user, login_user, logout_user
from pathlib import Path
from datetime import datetime
import logging
from logging.handlers import RotatingFileHandler
import re

from src.model.llm import LLMModel
from src.utils.indexer import index_documents
from src.retrieval.vector_store import VectorStore
from src.models.user import User
import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create logs directory if it doesn't exist
logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)

# Set up file handler
file_handler = RotatingFileHandler(
    logs_dir / "app.log",
    maxBytes=1024 * 1024,  # 1MB
    backupCount=5
)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
))
logger.addHandler(file_handler)

app = Flask(__name__)
app.config['SECRET_KEY'] = config.FLASK_SECRET_KEY

# Initialize login manager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Global variables for model and vector store
llm_model = None
vector_store = None

# Dictionary of greeting patterns and responses
GREETING_RESPONSES = {
    r"^(hi|hello|hey|greetings|howdy)(\s.*)?$": "Hello! How can I help you with AIX systems today?",
    r"^(good morning|good afternoon|good evening)(\s.*)?$": "Hello! How can I assist you with AIX systems?",
    r"^(how are you|how's it going|how are things|what's up)(\s.*)?$": "I'm doing well, thank you for asking! How can I help you with AIX systems today?",
    r"^(help|assist)(\s.*)?$": "I'd be happy to help! What would you like to know about AIX systems?",
    r"^(thank you|thanks)(\s.*)?$": "You're welcome! Is there anything else you'd like to know about AIX systems?",
}

def is_greeting(message):
    """Check if a message is a greeting and return the appropriate response."""
    message = message.lower().strip()
    
    for pattern, response in GREETING_RESPONSES.items():
        if re.match(pattern, message, re.IGNORECASE):
            return response
            
    return None

# User loader callback
@login_manager.user_loader
def load_user(user_id):
    return User.get_user_by_id(user_id)

# Login routes
@app.route('/login')
def login():
    """Render login page."""
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login_post():
    """Process login form submission."""
    username_or_email = request.form.get('username')
    password = request.form.get('password')
    
    if not username_or_email or not password:
        return render_template('login.html', error="Please enter username/email and password")
    
    # Check if login by email
    user = User.get_user_by_email(username_or_email)
    
    # If not found by email, try username
    if not user:
        user = User.get_user_by_username(username_or_email)
    
    # If still not found, or password doesn't match, show error
    if not user or not User.verify_password(user.password_hash, password):
        return render_template('login.html', error="Invalid username/email or password")
    
    # Login successful
    login_user(user)
    return redirect(url_for('home'))

# Registration route
@app.route('/register', methods=['POST'])
def register():
    """Process registration form submission."""
    username = request.form.get('username')
    email = request.form.get('email')
    password = request.form.get('password')
    confirm_password = request.form.get('confirm_password')
    
    # Validate input
    if not username or not email or not password or not confirm_password:
        return render_template('login.html', error="All fields are required", register=True)
    
    if password != confirm_password:
        return render_template('login.html', error="Passwords do not match", register=True)
    
    # Create user
    user, error = User.create_user(username, email, password)
    
    if not user:
        return render_template('login.html', error=error, register=True)
    
    # Login the new user
    login_user(user)
    return redirect(url_for('home'))

# Logout route
@app.route('/logout')
@login_required
def logout():
    """Log out user."""
    logout_user()
    return redirect(url_for('login'))

@app.route('/')
@login_required
def home():
    """Render the main chat interface with no pre-loaded chat."""
    # Clear current session ID to ensure no chat history is loaded
    current_user.current_session_id = None
    current_user._save_user_file()
    
    return render_template('index.html', 
                         title=getattr(config, 'UI_TITLE', 'Systems Assistant'),
                         description=getattr(config, 'UI_DESCRIPTION', 'Your AI expert on IBM AIX systems and PowerHA'),
                         initial_message=getattr(config, 'UI_INITIAL_MESSAGE', 'Hello! How can I help you today?'),
                         user=current_user)

@app.route('/chat/<session_id>')
@login_required
def chat_session(session_id):
    """Render chat interface with specific session."""
    # Check if session exists and belongs to user
    if not current_user.is_valid_session(session_id):
        return redirect(url_for('home'))
    
    # Switch to the requested session
    current_user.switch_chat_session(session_id)
    
    return render_template('index.html', 
                         title=getattr(config, 'UI_TITLE', 'Systems Assistant'),
                         description=getattr(config, 'UI_DESCRIPTION', 'Your AI expert on IBM AIX systems and PowerHA'),
                         initial_message=getattr(config, 'UI_INITIAL_MESSAGE', 'Hello! How can I help you today?'),
                         user=current_user)

@app.route('/api/chat', methods=['POST'])
@login_required
def chat():
    """Handle chat requests."""
    try:
        data = request.json
        query = data.get('message', '').strip()
        task_id = data.get('task_id', None)
        user_timestamp = data.get('user_timestamp', None)
        
        if not query:
            return jsonify({'error': 'Empty message'}), 400
            
        # Check if the necessary components are initialized
        if vector_store is None or llm_model is None:
            logger.error("Chat request received but system is not fully initialized")
            return jsonify({
                'error': 'System initialization error',
                'response': "I'm sorry, but the system is still initializing or encountered an error during startup. Please try again in a few moments."
            }), 503  # Service Unavailable
        
        # Check if the message is a greeting - fast path for greetings
        greeting_response = is_greeting(query)
        if greeting_response:
            logger.info(f"Greeting detected: {query[:50]}...")
            
            # Generate timestamp for internal tracking only (not displayed)
            now = datetime.now()
            timestamp = now.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            iso_timestamp = now.isoformat()
            
            # Save chat to user's current session with separate timestamps
            session_id = current_user.add_message_to_current_session(
                query, 
                greeting_response,
                user_timestamp  # Pass user timestamp separately
            )
            
            return jsonify({
                'response': greeting_response,
                'session_id': session_id
            })
            
        # Retrieve relevant documents
        try:
            logger.info(f"Processing query: {query[:50]}...")
            
            # Optimize retrieval by limiting the number of documents
            retrieved_docs = vector_store.similarity_search(
                query,
                k=min(config.TOP_K_RETRIEVAL, 3)  # Limit to max 3 docs for faster processing
            )
            
            if not retrieved_docs:
                logger.warning("No relevant documents found for query")
                
                return jsonify({
                    'response': "I don't have enough information to answer that question. Could you provide more details or ask a different question?"
                })
            
            # Get response from model using RAG
            logger.info("Generating response...")
            try:
                response = llm_model.generate_rag_response(query, retrieved_docs, task_id=task_id)
                
                if not response or not response.strip():
                    logger.error("Empty response received from model")
                    
                    return jsonify({
                        'response': "I apologize, but I couldn't generate a meaningful response. Please try rephrasing your question."
                    })
                
                # Optimize response cleaning with compiled regex patterns
                response = re.sub(r'(?:^|\n)(?:User Question|Assistant|User)\s*(?:\n|$)', '\n', response, flags=re.IGNORECASE)
                response = re.sub(r'(?:^|\n)\d{1,2}:\d{1,2}(?::\d{1,2})?\s*(?:AM|PM)(?:\n|$)', '\n', response, flags=re.IGNORECASE)
                
                # Clean up the response - remove empty citation patterns
                response = response.replace('()', '').replace('( )', '').replace('(  )', '')
                
                # Fix common bullet point issues
                response = response.replace('•', '• ').replace('⚫', '• ').replace('  •', ' •')
                
                # Clean up excessive whitespace in one pass
                response = re.sub(r'\n{3,}', '\n\n', response).strip()
                
                # Save chat to user's current session with separate timestamps
                session_id = current_user.add_message_to_current_session(
                    query, 
                    response,
                    user_timestamp  # Pass user timestamp separately
                )
                
                logger.info(f"Response generated successfully: {response[:50]}...")
                return jsonify({
                    'response': response,
                    'session_id': session_id
                })
            except Exception as model_error:
                logger.error(f"Error in response generation: {str(model_error)}", exc_info=True)
                return jsonify({
                    'response': "I encountered an error while generating your response. Please try rephrasing your question.",
                    'error': f'Error generating response: {str(model_error)}'
                }), 500
            
        except Exception as e:
            logger.error(f"Error in document retrieval or response generation: {str(e)}", exc_info=True)
            
            return jsonify({
                'response': "I encountered an error while processing your request. Please try again or rephrase your question.",
                'error': f'Error processing request: {str(e)}'
            }), 500
            
    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}", exc_info=True)
        
        return jsonify({
            'response': "I encountered an error while processing your request. Please try again later.",
            'error': 'An error occurred while processing your request'
        }), 500

@app.route('/api/history')
@login_required
def get_history():
    """Get user chat history for current session."""
    try:
        history = current_user.get_chat_history()
        return jsonify({'history': history})
    except Exception as e:
        logger.error(f"Error retrieving chat history: {str(e)}", exc_info=True)
        return jsonify({'error': 'An error occurred while retrieving chat history'}), 500

@app.route('/api/sessions')
@login_required
def get_sessions():
    """Get all user chat sessions."""
    try:
        sessions = current_user.get_chat_sessions()
        current_id = current_user.current_session_id
        return jsonify({
            'sessions': sessions,
            'current_session_id': current_id
        })
    except Exception as e:
        logger.error(f"Error retrieving chat sessions: {str(e)}", exc_info=True)
        return jsonify({'error': 'An error occurred while retrieving chat sessions'}), 500

@app.route('/api/sessions/new', methods=['POST'])
@login_required
def new_session():
    """Create a new chat session."""
    try:
        data = request.json
        title = data.get('title')
        
        session_id = current_user.create_new_chat_session(title)
        
        return jsonify({
            'session_id': session_id,
            'success': True
        })
    except Exception as e:
        logger.error(f"Error creating new chat session: {str(e)}", exc_info=True)
        return jsonify({'error': 'An error occurred while creating a new chat session'}), 500

@app.route('/api/sessions/switch', methods=['POST'])
@login_required
def switch_session():
    """Switch to a different chat session."""
    try:
        data = request.json
        session_id = data.get('session_id')
        
        if not session_id:
            return jsonify({'error': 'No session ID provided'}), 400
            
        success = current_user.switch_chat_session(session_id)
        
        if success:
            return jsonify({
                'success': True,
                'session': current_user.get_current_chat_session()
            })
        else:
            return jsonify({'error': 'Invalid session ID'}), 400
            
    except Exception as e:
        logger.error(f"Error switching chat session: {str(e)}", exc_info=True)
        return jsonify({'error': 'An error occurred while switching chat sessions'}), 500

@app.route('/api/sessions/delete', methods=['POST'])
@login_required
def delete_session():
    """Delete a chat session."""
    try:
        data = request.json
        session_id = data.get('session_id')
        is_last_session = data.get('is_last_session', False)
        
        if not session_id:
            return jsonify({'error': 'No session ID provided'}), 400
        
        # Check if this is the current session
        is_current_session = session_id == current_user.current_session_id
        
        # If this is the last session, we'll handle it specially
        if is_last_session:
            # First delete the session
            success = current_user.delete_chat_session(session_id)
            if not success:
                return jsonify({'error': 'Invalid session ID'}), 400
                
            # At this point, the user should have no sessions
            # We'll tell the frontend to redirect to root
            return jsonify({
                'success': True,
                'current_session_id': None,
                'create_new': True,
                'redirect_to_root': True,
                'message': 'Last session deleted successfully'
            })
        else:
            # Normal case - not the last session
            success = current_user.delete_chat_session(session_id)
            
            if success:
                # Check if we need to create a new session
                create_new = False
                
                # Check if we somehow ended up with no sessions
                sessions = current_user.get_chat_sessions()
                if len(sessions) == 0 or not current_user.current_session_id:
                    # Create a fresh session with no messages
                    current_user.create_new_chat_session()
                    create_new = True
                
                # If the deleted session was the current one, redirect to root
                redirect_to_root = is_current_session
                    
                return jsonify({
                    'success': True,
                    'current_session_id': current_user.current_session_id,
                    'create_new': create_new,
                    'redirect_to_root': redirect_to_root
                })
            else:
                return jsonify({'error': 'Invalid session ID'}), 400
            
    except Exception as e:
        logger.error(f"Error deleting chat session: {str(e)}", exc_info=True)
        return jsonify({'error': 'An error occurred while deleting the chat session'}), 500

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files."""
    return send_from_directory(config.STATIC_DIR, filename)

@app.errorhandler(404)
def not_found_error(error):
    """Handle 404 errors."""
    return jsonify({'error': 'Resource not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error(f"Server error: {str(error)}", exc_info=True)
    return jsonify({'error': 'Internal server error'}), 500

# Add a generic error handler for all other HTTP errors
@app.errorhandler(Exception)
def handle_exception(e):
    """Handle all unhandled exceptions."""
    logger.error(f"Unhandled exception: {str(e)}", exc_info=True)
    return jsonify({
        'error': 'An unexpected error occurred',
        'message': str(e)
    }), 500

def initialize_app():
    """Initialize the application components."""
    global llm_model, vector_store
    
    try:
        logger.info("=" * 50)
        logger.info(f"AIX Systems RAG Chatbot")
        logger.info("=" * 50)
        
        # Index documents
        # [1/3] Indexing documents...
        logger.info("\n[1/3] Indexing documents...")
        try:
            vector_store = index_documents(force_reindex=False)
            if not vector_store:
                logger.error("Failed to initialize vector store")
                raise RuntimeError("Vector store initialization failed")
        except Exception as e:
            logger.error(f"Error during document indexing: {str(e)}", exc_info=True)
            raise RuntimeError(f"Document indexing failed: {str(e)}")
        
        # Load model
        logger.info("\n[2/3] Loading language model...")
        try:
            llm_model = LLMModel()
            if not llm_model:
                logger.error("Failed to initialize language model")
                raise RuntimeError("Language model initialization failed")
        except Exception as e:
            logger.error(f"Error loading language model: {str(e)}", exc_info=True)
            raise RuntimeError(f"Language model loading failed: {str(e)}")
        
        logger.info("\n[3/3] Initializing web interface...")
        logger.info(f"\nServer ready! Access the interface at http://{config.FLASK_HOST}:{config.FLASK_PORT}")
        
    except Exception as e:
        logger.error(f"Error initializing application: {str(e)}", exc_info=True)
        # Set global variables to indicate initialization failed
        vector_store = None
        llm_model = None
        raise

def main():
    """Main entry point."""
    try:
        initialize_app()
        app.run(
            host=config.FLASK_HOST,
            port=config.FLASK_PORT,
            debug=config.FLASK_DEBUG
        )
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
