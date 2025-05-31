from flask import Flask, request, jsonify
import os
from main import RAGChatbot
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()
api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

if not api_token:
    logger.error("HuggingFace API token not found")
    raise ValueError("Missing HUGGINGFACEHUB_API_TOKEN")

# Initialize chatbot
logger.info("Initializing RAG chatbot...")
chatbot = RAGChatbot(huggingface_token=api_token)
logger.info("Chatbot ready")

@app.route('/upload', methods=['POST'])
def upload():
    if 'pdfs' not in request.files:
        return jsonify({'error': 'No PDF files uploaded'}), 400

    saved_paths = []
    for pdf in request.files.getlist('pdfs'):
        try:
            filename = pdf.filename
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            pdf.save(path)
            saved_paths.append(path)
            logger.info(f"Saved PDF: {filename}")
        except Exception as e:
            logger.error(f"Error saving {pdf.filename}: {str(e)}")
            continue

    if not saved_paths:
        return jsonify({'error': 'Could not save any PDFs'}), 400

    chatbot.ingest_pdfs(saved_paths)
    return jsonify({'message': f'{len(saved_paths)} PDF(s) processed successfully'})

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    if not data or 'question' not in data:
        return jsonify({'error': 'No question provided'}), 400

    answer = chatbot.ask(data['question'])
    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)