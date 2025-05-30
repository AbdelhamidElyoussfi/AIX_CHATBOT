"""
User model for database authentication and chat session management.
"""
import os
import json
import uuid
import hashlib
import secrets
from datetime import datetime, timedelta
from flask_login import UserMixin
from pathlib import Path
import config
import shutil

class User(UserMixin):
    """User model for Flask-Login."""
    
    def __init__(self, id, username, email, password_hash=None):
        self.id = id
        self.username = username
        self.email = email
        self.password_hash = password_hash
        self.session_ids = []  # List of session IDs
        self.current_session_id = None
        self._sessions_cache = {}  # In-memory cache for loaded sessions
        self._sessions_dir = Path(config.USER_DATA_DIR) / self.id / "sessions"
        self._user_file = Path(config.USER_DATA_DIR) / f"{self.id}.json"
        self._load_user_file()

    def get_id(self):
        """Required for Flask-Login."""
        return self.id
    
    @staticmethod
    def _hash_password(password, salt=None):
        """Hash password with a salt using PBKDF2."""
        if salt is None:
            salt = secrets.token_hex(16)
        hash_obj = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt.encode('utf-8'), 100000)
        password_hash = salt + ":" + hash_obj.hex()
        return password_hash
    
    @staticmethod
    def verify_password(stored_hash, provided_password):
        """Verify a password against a stored hash."""
        if not stored_hash or not provided_password:
            return False
        
        try:
            salt, hash_val = stored_hash.split(':')
            hash_obj = hashlib.pbkdf2_hmac('sha256', provided_password.encode('utf-8'), salt.encode('utf-8'), 100000)
            calculated_hash = hash_obj.hex()
            return calculated_hash == hash_val
        except Exception:
            return False
        
    @staticmethod
    def get_user_by_id(user_id):
        """Retrieve a user by ID from storage."""
        user_file = Path(config.USER_DATA_DIR) / f"{user_id}.json"
        if not user_file.exists():
            return None
            
        with open(user_file, 'r') as f:
            user_data = json.load(f)
            
        user = User(
            id=user_data['id'],
            username=user_data['username'],
            email=user_data['email'],
            password_hash=user_data.get('password_hash')
        )
        
        user.session_ids = user_data.get('session_ids', [])
        user.current_session_id = user_data.get('current_session_id')
        
        return user
    
    @staticmethod
    def get_user_by_email(email):
        """Retrieve a user by email from storage."""
        if not email:
            return None
            
        # Scan user files to find matching email
        user_dir = Path(config.USER_DATA_DIR)
        user_dir.mkdir(exist_ok=True, parents=True)
        
        for user_file in user_dir.glob("*.json"):
            try:
                with open(user_file, 'r') as f:
                    user_data = json.load(f)
                    
                if user_data.get('email') == email:
                    return User.get_user_by_id(user_data['id'])
            except Exception:
                continue
                
        return None
    
    @staticmethod
    def get_user_by_username(username):
        """Retrieve a user by username from storage."""
        if not username:
            return None
            
        # Scan user files to find matching username
        user_dir = Path(config.USER_DATA_DIR)
        user_dir.mkdir(exist_ok=True, parents=True)
        
        for user_file in user_dir.glob("*.json"):
            try:
                with open(user_file, 'r') as f:
                    user_data = json.load(f)
                    
                if user_data.get('username') == username:
                    return User.get_user_by_id(user_data['id'])
            except Exception:
                continue
                
        return None
    
    @staticmethod
    def create_user(username, email, password):
        """Create a new user."""
        # Check if user with this email already exists
        if User.get_user_by_email(email):
            return None, "Email already registered"
            
        # Check if user with this username already exists
        if User.get_user_by_username(username):
            return None, "Username already taken"
            
        # Generate user ID
        user_id = str(uuid.uuid4())
        
        # Hash password
        password_hash = User._hash_password(password)
        
        # Create user
        user = User(
            id=user_id,
            username=username,
            email=email,
            password_hash=password_hash
        )
        
        # Create initial chat session
        user.create_new_chat_session()
        
        # Save user
        user.save()
        
        return user, None
        
    def _load_user_file(self):
        if self._user_file.exists():
            with open(self._user_file, 'r') as f:
                user_data = json.load(f)
            self.session_ids = user_data.get('session_ids', [])
            self.current_session_id = user_data.get('current_session_id')
        else:
            self.session_ids = []
            self.current_session_id = None
        # Ensure sessions dir exists
        self._sessions_dir.mkdir(parents=True, exist_ok=True)

    def _save_user_file(self):
        user_data = {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'password_hash': self.password_hash,
            'session_ids': self.session_ids,
            'current_session_id': self.current_session_id
        }
        tmp_file = self._user_file.with_suffix('.tmp')
        with open(tmp_file, 'w') as f:
            json.dump(user_data, f)
        shutil.move(str(tmp_file), str(self._user_file))

    def _session_file(self, session_id):
        return self._sessions_dir / f"{session_id}.json"

    def _load_session(self, session_id):
        session_file = self._session_file(session_id)
        if not session_file.exists():
            return None
        with open(session_file, 'r') as f:
            return json.load(f)

    def _save_session(self, session):
        session_file = self._session_file(session['id'])
        tmp_file = session_file.with_suffix('.tmp')
        with open(tmp_file, 'w') as f:
            json.dump(session, f)
        shutil.move(str(tmp_file), str(session_file))

    def _delete_session_file(self, session_id):
        session_file = self._session_file(session_id)
        if session_file.exists():
            session_file.unlink()
    
    def save(self):
        """Save user data to file."""
        self._save_user_file()

    def create_new_chat_session(self, title=None):
        """Create a new chat session."""
        session_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # If no title is provided, use a default one with date
        if not title:
            title = f"Chat {datetime.now().strftime('%b %d, %Y')}"
        
        session = {
            'id': session_id,
            'title': title,
            'created_at': timestamp,
            'updated_at': timestamp,
            'messages': []
        }
        
        self._save_session(session)
        self.session_ids.insert(0, session_id)  # Newest first
        self.current_session_id = session_id
        self._save_user_file()
        return session_id
    
    def get_current_chat_session(self):
        """Get the current chat session."""
        # If no current session, create one
        if not self.current_session_id or self.current_session_id not in self.session_ids:
            self.create_new_chat_session()
        
        return self._load_session(self.current_session_id)
    
    def get_chat_sessions(self):
        """Get all chat sessions."""
        sessions = []
        for sid in self.session_ids:
            session = self._load_session(sid)
            if session:
                sessions.append(session)
        # Sort by updated_at (newest first)
        sessions.sort(key=lambda x: x['updated_at'], reverse=True)
        return sessions
    
    def add_message_to_current_session(self, query, response, user_timestamp=None):
        """Add a new message to the current chat session."""
        # Get current session or create if none exists
        if not self.current_session_id or self.current_session_id not in self.session_ids:
            self.create_new_chat_session()
        
        session = self._load_session(self.current_session_id)
        if not session:
            session = {
                'id': self.current_session_id,
                'title': f"Chat {datetime.now().strftime('%b %d, %Y')}",
                'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'updated_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'messages': []
            }
        
        # Generate timestamp for internal use only (not displayed in UI)
        now = datetime.now()
        response_timestamp = now.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        response_iso_timestamp = now.isoformat()
        response_display_timestamp = ""  # Empty string for display (timestamps hidden)
        
        # If user_timestamp is provided, use it for internal data only
        if user_timestamp:
            # Parse user timestamp if it's a string
            if isinstance(user_timestamp, str):
                try:
                    user_time = datetime.fromisoformat(user_timestamp.replace('Z', '+00:00'))
                    user_display_timestamp = ""  # Empty string for display (timestamps hidden)
                    user_iso_timestamp = user_timestamp
                    user_full_timestamp = user_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                except ValueError:
                    # If parsing fails, use current time
                    user_display_timestamp = ""  # Empty string for display (timestamps hidden)
                    user_iso_timestamp = now.isoformat()
                    user_full_timestamp = now.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            else:
                # If it's a dict with iso and display fields
                user_iso_timestamp = user_timestamp.get('iso', now.isoformat())
                user_display_timestamp = ""  # Empty string for display (timestamps hidden)
                user_full_timestamp = user_timestamp.get('full', now.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3])
        else:
            # If no user timestamp provided, use current time but slightly earlier
            user_time = now - timedelta(seconds=1)
            user_display_timestamp = ""  # Empty string for display (timestamps hidden)
            user_iso_timestamp = user_time.isoformat()
            user_full_timestamp = user_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        
        # Add user message with its own timestamp
        session['messages'].append({
            'query': query,
            'timestamp': user_display_timestamp,
            'iso_timestamp': user_iso_timestamp,
            'full_timestamp': user_full_timestamp,
            'type': 'user'
        })
        
        # Add bot response with its own timestamp
        session['messages'].append({
            'response': response,
            'timestamp': response_display_timestamp,
            'iso_timestamp': response_iso_timestamp,
            'full_timestamp': response_timestamp,
            'type': 'bot'
        })
        
        # Update session timestamp
        session['updated_at'] = response_timestamp
        
        # Update session title if it's the first message and has a generic title
        if len(session['messages']) <= 2 and 'Chat ' in session['title']:
            # Create a more descriptive title based on the user's question
            clean_query = query.strip()
            
            # Remove common question prefixes
            prefixes_to_remove = [
                "tell me", "how to", "how do i", "can you", "what is", "what are", 
                "explain", "could you", "i need to", "i want to", "please"
            ]
            
            for prefix in prefixes_to_remove:
                if clean_query.lower().startswith(prefix):
                    clean_query = clean_query[len(prefix):].strip()
                    break
            
            # Capitalize first letter
            if clean_query:
                clean_query = clean_query[0].upper() + clean_query[1:]
                
            # Remove trailing punctuation
            while clean_query and clean_query[-1] in ".?!":
                clean_query = clean_query[:-1]
                
            # Add a topic prefix based on query content
            if clean_query.lower().startswith("how"):
                title = f"How to {clean_query[4:].strip()}"
            elif "restart" in clean_query.lower() or "start" in clean_query.lower():
                title = f"Restarting {clean_query}"
            elif "install" in clean_query.lower():
                title = f"Installing {clean_query}"
            elif "configure" in clean_query.lower() or "setup" in clean_query.lower():
                title = f"Configuring {clean_query}"
            elif "troubleshoot" in clean_query.lower() or "fix" in clean_query.lower() or "solve" in clean_query.lower():
                title = f"Troubleshooting {clean_query}"
            else:
                title = clean_query
            
            # Limit title length
            if len(title) > 40:
                title = title[:37] + '...'
                
            session['title'] = title
        
        self._save_session(session)
        self._save_user_file()
        
        # Return the session ID in case it was just created
        return self.current_session_id
    
    def switch_chat_session(self, session_id):
        """Switch to a different chat session."""
        if session_id in self.session_ids:
            self.current_session_id = session_id
            self._save_user_file()
            return True
        return False
    
    def delete_chat_session(self, session_id):
        """Delete a chat session."""
        if session_id in self.session_ids:
            # Store whether we're deleting the current session
            is_current_session = session_id == self.current_session_id
            
            # Delete the session
            self.session_ids.remove(session_id)
            
            # If deleting current session, handle differently
            if is_current_session:
                # If we have other sessions, we'll let the frontend handle redirection
                # Don't automatically switch to another session
                self.current_session_id = None
            
            # If no sessions left, prepare for a new empty one
            if len(self.session_ids) == 0:
                # Don't create it here, let the controller handle it
                self.current_session_id = None
            elif not self.current_session_id and not is_current_session:
                # If we have other sessions but no current one,
                # and we didn't delete the current session,
                # set the most recent as current
                sessions = self.get_chat_sessions()
                if sessions:
                    self.current_session_id = sessions[0]['id']
            
            self._delete_session_file(session_id)
            self._save_user_file()
            return True
        return False
    
    def is_valid_session(self, session_id):
        """Check if a session ID exists and belongs to this user."""
        if not session_id:
            return False
        return session_id in self.session_ids
    
    def get_chat_history(self):
        """Retrieve the user's current chat history."""
        if not self.current_session_id or self.current_session_id not in self.session_ids:
            return []
        
        session = self._load_session(self.current_session_id)
        if not session:
            return []
            
        # Format messages for the frontend
        formatted_messages = []
        
        # Process messages in pairs (user message followed by bot response)
        messages = session.get('messages', [])
        
        # Check if we have the new format (with 'type' field) or old format
        if messages and 'type' in messages[0]:
            # New format - messages are stored separately with type field
            i = 0
            while i < len(messages):
                user_msg = None
                bot_msg = None
                
                # Find user message
                if i < len(messages) and messages[i].get('type') == 'user':
                    user_msg = messages[i]
                    i += 1
                
                # Find corresponding bot message
                if i < len(messages) and messages[i].get('type') == 'bot':
                    bot_msg = messages[i]
                    i += 1
                
                # If we found a valid pair, add it to formatted messages
                if user_msg and bot_msg:
                    formatted_messages.append({
                        'query': user_msg.get('query', ''),
                        'response': bot_msg.get('response', ''),
                        'timestamp': bot_msg.get('timestamp', ''),
                        'iso_timestamp': bot_msg.get('iso_timestamp', ''),
                        'user_timestamp': user_msg.get('timestamp', ''),
                        'user_iso_timestamp': user_msg.get('iso_timestamp', '')
                    })
                # Handle orphaned user message (no response yet)
                elif user_msg:
                    formatted_messages.append({
                        'query': user_msg.get('query', ''),
                        'response': '',
                        'timestamp': user_msg.get('timestamp', ''),
                        'iso_timestamp': user_msg.get('iso_timestamp', ''),
                        'user_timestamp': user_msg.get('timestamp', ''),
                        'user_iso_timestamp': user_msg.get('iso_timestamp', '')
                    })
                # Handle orphaned bot message (shouldn't happen, but just in case)
                elif bot_msg:
                    formatted_messages.append({
                        'query': '',
                        'response': bot_msg.get('response', ''),
                        'timestamp': bot_msg.get('timestamp', ''),
                        'iso_timestamp': bot_msg.get('iso_timestamp', '')
                    })
        else:
            # Old format - each message has both query and response
            for msg in messages:
                # Ensure all messages have the required timestamp fields
                if 'iso_timestamp' not in msg and 'timestamp' in msg:
                    # Try to parse the timestamp or use current time
                    try:
                        dt = datetime.strptime(msg['timestamp'], "%Y-%m-%d %H:%M:%S")
                        msg['iso_timestamp'] = dt.isoformat()
                    except:
                        msg['iso_timestamp'] = datetime.now().isoformat()
                
                # Ensure display timestamp is in correct format
                if 'timestamp' in msg:
                    try:
                        # Try to parse full timestamp and convert to display format
                        dt = datetime.strptime(msg['timestamp'], "%Y-%m-%d %H:%M:%S.%f")
                        msg['timestamp'] = dt.strftime("%I:%M %p")
                    except:
                        # If it fails, leave as is
                        pass
                
                formatted_messages.append(msg)
        
        return formatted_messages
    
    # Legacy method for backward compatibility
    def add_chat(self, query, response, timestamp):
        """Add a new chat entry to the user's history."""
        self.add_message_to_current_session(query, response) 