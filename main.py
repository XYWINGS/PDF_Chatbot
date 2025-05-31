import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceEndpoint
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

class RAGChatbot:
    def __init__(self, huggingface_token: str):
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = huggingface_token

        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        # Updated to use HuggingFaceEndpoint
        self.llm = HuggingFaceEndpoint(
            repo_id="google/flan-t5-base",
            task="text2text-generation",
            temperature=0.1,
            max_new_tokens=512
        )
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=False)
        self.qa_chain = None

    def ingest_pdfs(self, pdf_paths):
        documents = []
        for path in pdf_paths:
            loader = PyPDFLoader(path)
            documents.extend(loader.load())

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)
        vectorstore = FAISS.from_documents(texts, self.embeddings)

        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=vectorstore.as_retriever(),
            memory=self.memory,
            chain_type="stuff"
        )

    def ask(self, question: str):
        if not self.qa_chain:
            return "Please upload PDFs first."
        try:
            result = self.qa_chain.invoke({"question": question})
            return result["answer"]
        except Exception as e:
            return f"An error occurred: {str(e)}"