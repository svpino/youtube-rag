from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough

class RAGChain:
    def __init__(self, api_key, vectorstore):
        self.vectorstore = vectorstore
        self.llm = self.get_llm(api_key)
        self.chat_template = self.get_template()

    def get_llm(self, api_key):

        if os.getenv("GEMINI_API_KEY"):
             model = ChatGoogleGenerativeAI(google_api_key=api_key, 
                                   model="gemini-1.5-flash")
             return model
        else:
            raise ValueError("No valid API key found! Please set one in .env file.")

    def get_template(self):
        chat_template = ChatPromptTemplate.from_messages([
        # System Message Prompt Template
        SystemMessage(content="""You are a Helpful AI Bot.
                    Given a context and question from user,
                    you should answer based on the given context."""),
        # Human Message Prompt Template
        HumanMessagePromptTemplate.from_template("""Answer the question based on the given context.
        Context: {context}
        Question: {question}
        Answer: """)
        ])
        return chat_template
    
    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    def create_chain(self):
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5}) 
        output_parser = StrOutputParser()
        
        rag_chain = (
            {"context": retriever | self.format_docs, "question": RunnablePassthrough()}
            | self.chat_template
            | self.llm
            | output_parser
        )
        return rag_chain
        

if __name__ == "__main__":
    from document_processor import DocumentProcessor
    from embedding_indexer import EmbeddingIndexer
    
    load_dotenv()

    GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
    processor = DocumentProcessor("transcription.txt")
    texts = processor.load_and_split()
    print(f"Processed {len(texts)} text chunks")

    indexer = EmbeddingIndexer(GOOGLE_API_KEY)
    vectorstore = indexer.create_vectorstore(texts)
    print("Vector store created successfully")

    rag_chain = RAGChain(GOOGLE_API_KEY, vectorstore)
    qa_chain = rag_chain.create_chain()

    query = "What is the capital of France?"
    response = qa_chain.invoke(query)
    print(f"Answer: {response}")