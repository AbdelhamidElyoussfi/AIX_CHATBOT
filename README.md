# AIX Systems Assistant

A specialized RAG (Retrieval-Augmented Generation) chatbot designed to provide expert assistance on IBM AIX operating systems and PowerHA.


## Features

- **Intelligent AIX Systems Support**: Get accurate answers to your AIX system administration questions
- **User Authentication**: Secure login and registration system
- **Chat Session Management**: Create, switch between, and delete conversation sessions
- **Retrieval-Augmented Generation**: Uses document retrieval to provide accurate, context-aware responses
- **Responsive UI**: Clean, modern interface that works on desktop and mobile devices

## Screenshots

### Login Screen
![AIX Systems Assistant Interface](screenshots/Screenshot%202025-05-28%20174939.png)

### Main Chat Interface
![Screenshot 2025-05-28 180508](https://github.com/user-attachments/assets/6461db26-ed63-4c1c-942c-95a8d288cd80)

### How the chatbot answer 
![Screenshot 2025-05-28 194149](https://github.com/user-attachments/assets/8b239194-84b9-4903-bd8b-8d874d10f197)
![Screenshot 2025-05-28 195457](https://github.com/user-attachments/assets/1e5e3c9f-7399-4cf5-a75f-e0457dbe4b3a)


## Technology Stack

- **Backend**: Flask (Python)
- **AI/ML**: 
  - LangChain for RAG pipeline
  - Sentence Transformers for embeddings
  - Local LLM support (Granite 3.3 2B Instruct)
  - ChromaDB for vector storage
- **Frontend**: HTML, CSS, JavaScript
- **Authentication**: Flask-Login

## Installation

1. Clone the repository:
```bash
git clone https://github.com/AbdelhamidElyoussfi/AIX_CHATBOT.git
cd AIX_CHATBOT
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
export FLASK_SECRET_KEY="your-secret-key-here"
```

4. Place your AIX documentation in the `Docs` directory.

5. Run the application:
```bash
python app.py
```

## Usage

1. Access the application at `http://localhost:5000`
2. Register a new account or log in with existing credentials
3. Start a new chat session and ask questions about AIX systems
4. Create multiple chat sessions for different topics or issues
5. Review your chat history at any time

## Configuration

The application can be configured by editing the `config.py` file:

- Model settings (embedding model, retrieval parameters)
- Document processing settings (chunk size, overlap)
- LLM generation parameters (temperature, max tokens)
- UI customization

## Project Structure

```
.
├── app.py                 # Main Flask application
├── config.py              # Configuration settings
├── requirements.txt       # Python dependencies
├── data/                  # Vector database and user data
├── Docs/                  # AIX documentation for RAG
├── logs/                  # Application logs
├── screenshots/           # Application screenshots
├── src/                   # Source code modules
│   ├── model/             # LLM implementation
│   ├── retrieval/         # Vector store and retrieval logic
│   ├── utils/             # Utility functions
│   └── models/            # Data models
├── static/                # Static assets (CSS, JS)
└── templates/             # HTML templates
```

## License

[MIT License](LICENSE)

## Acknowledgments

- IBM for AIX documentation
- Open-source AI/ML community
