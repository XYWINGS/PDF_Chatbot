# server.py

from dotenv import load_dotenv
import os
load_dotenv()  # Load environment variables from .env
api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

from flask import Flask, request, jsonify
from main import RAGChatbot
import os

# Create upload directory if it doesn't exist
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


if not api_token:
    raise ValueError(api_token)

os.environ["HUGGINGFACEHUB_API_TOKEN"] = api_token


# Initialize the chatbot
chatbot = RAGChatbot(huggingface_token=api_token)

@app.route('/upload', methods=['POST'])
def upload():
    if 'pdfs' not in request.files:
        return jsonify({'error': 'No PDF files uploaded'}), 400

    pdfs = request.files.getlist('pdfs')
    saved_paths = []

    for pdf in pdfs:
        filename = pdf.filename
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        pdf.save(path)
        saved_paths.append(path)

    chatbot.ingest_pdfs(saved_paths)
    return jsonify({'message': 'PDFs processed and ingested successfully'})

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    if not data or 'question' not in data:
        return jsonify({'error': 'No question provided'}), 400

    answer = chatbot.ask(data['question'])
    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(debug=True)
