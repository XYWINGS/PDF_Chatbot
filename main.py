import os
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGChatbot:
    def __init__(self, huggingface_token: str):
        logger.info("Initializing RAGChatbot...")
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = huggingface_token
        
        # Initialize with updated imports
        logger.info("Loading embeddings model...")
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        logger.info("Initializing LLM...")
        self.llm = HuggingFaceEndpoint(
            repo_id="HuggingFaceH4/zephyr-7b-beta",
            temperature=0.1,
            max_new_tokens=512,
            do_sample=True,
            timeout=30  # Added timeout for better error handling
        )
        
        logger.info("Setting up conversation memory...")
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=False,
            output_key='answer'  # Added for better chain compatibility
        )
        self.qa_chain = None
        logger.info("RAGChatbot initialized successfully")

    def ingest_pdfs(self, pdf_paths):
        if not pdf_paths:
            logger.warning("No PDF paths provided")
            return
            
        logger.info(f"Processing {len(pdf_paths)} PDF(s)...")
        documents = []
        for path in pdf_paths:
            try:
                logger.info(f"Loading PDF: {path}")
                loader = PyPDFLoader(path)
                documents.extend(loader.load())
            except Exception as e:
                logger.error(f"Error loading PDF {path}: {str(e)}")
                continue

        if not documents:
            logger.error("No documents loaded from PDFs")
            return

        logger.info("Splitting documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        texts = text_splitter.split_documents(documents)
        
        logger.info("Creating vector store...")
        vectorstore = FAISS.from_documents(texts, self.embeddings)
        
        logger.info("Setting up QA chain...")
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=vectorstore.as_retriever(),
            memory=self.memory,
            chain_type="stuff",
            return_source_documents=True,  # Added for debugging
            verbose=True  # Enable verbose logging
        )
        logger.info("PDF ingestion completed successfully")

    def ask(self, question: str):
        if not self.qa_chain:
            logger.warning("QA chain not initialized - no PDFs processed")
            return "Please upload PDFs first."
            
        logger.info(f"Processing question: {question}")
        try:
            result = self.qa_chain.invoke({"question": question})
            logger.info("Question processed successfully")
            return result["answer"]
        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            return f"An error occurred: {str(e)}"