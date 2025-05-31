# PDF Chatbot - RAG Implementation

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![LangChain](https://img.shields.io/badge/LangChain-0.1.0-orange)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Inference-yellow)

A conversational AI system that answers questions based on content extracted from uploaded PDF documents using Retrieval-Augmented Generation (RAG) with HuggingFace models.

## Features

- 📄 Upload and process multiple PDF documents
- 💬 Conversational interface with memory
- 🔍 Semantic search over document content
- 🤖 Powered by HuggingFace's state-of-the-art LLMs
- 🧠 Context-aware question answering

## Prerequisites

- Python 3.8+
- HuggingFace API token
- PDF documents to query

## Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/PDF_Chatbot.git
   cd PDF_Chatbot
   ```

2. **Set up virtual environment**
   ```bash
   python -m venv venv
   # macOS/Linux
   source venv/bin/activate
   # Windows
   venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**
   Create a `.env` file in the root directory with your HuggingFace token:
   ```
   HUGGINGFACEHUB_API_TOKEN="your_huggingface_api_token_here"
   ```

## Usage

1. **Start the server**
   ```bash
   python server.py
   ```

2. **Access the API endpoints**

   - **Upload PDFs** (POST request to `/upload`)
     ```bash
     curl -X POST -F "pdfs=@document1.pdf" -F "pdfs=@document2.pdf" http://localhost:5000/upload
     ```

   - **Ask questions** (POST request to `/ask`)
     ```bash
     curl -X POST -H "Content-Type: application/json" -d '{"question":"What is the main topic?"}' http://localhost:5000/ask
     ```

## Configuration

Customize the chatbot behavior by modifying these parameters in `main.py`:

```python
# LLM Configuration
self.llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",  # Change model here
    temperature=0.1,  # Adjust creativity (0-1)
    max_new_tokens=512,  # Response length limit
    do_sample=True
)

# Text Processing
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # Adjust document chunk size
    chunk_overlap=200  # Adjust overlap between chunks
)
```

## Recommended Models

Try these alternative HuggingFace models by changing the `repo_id`:

- `google/flan-t5-base` - Good for basic QA
- `mistralai/Mistral-7B-Instruct-v0.1` - More advanced
- `meta-llama/Llama-2-7b-chat-hf` - Requires approval

## Troubleshooting

**Error: "An error occurred: Unsupported chat history format"**
- Ensure `return_messages=True` in ConversationBufferMemory
- Clear conversation history if errors persist

**Slow responses**
- Try smaller PDFs or reduce chunk size
- Use a less resource-intensive model

## Acknowledgments

- LangChain for the RAG framework
- HuggingFace for the models and inference API
- FAISS for efficient vector search