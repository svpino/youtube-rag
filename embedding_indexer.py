from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

class EmbeddingIndexer:
    def __init__(self, API_KEY):
        self.embedding_model = GoogleGenerativeAIEmbeddings(google_api_key=API_KEY, model="models/embedding-001")

    def create_vectorstore(self, texts):
        db = Chroma.from_documents(texts, self.embedding_model, persist_directory="./chroma_db_")

        db.persist()

        db_vectorstore= Chroma(persist_directory="./chroma_db_", embedding_function=self.embedding_model)
        
        return db_vectorstore

if __name__ == "__main__":
    from document_processor import DocumentProcessor
    import os
    from dotenv import load_dotenv

    load_dotenv()

    GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

    processor = DocumentProcessor("transcription.txt")
    texts = processor.load_and_split()

    indexer = EmbeddingIndexer(GOOGLE_API_KEY)
    vectorstore = indexer.create_vectorstore(texts)
    print("Vector store created successfully")