class Chatbot:
    def __init__(self, qa_chain):
        self.qa_chain = qa_chain

    def get_response(self, user_input):
        try:
            response = self.qa_chain.invoke(user_input)
            return response
        except Exception as e:
            return f"An error occurred: {str(e)}"
        

if __name__ == "__main__":
    from rag_chain import RAGChain
    from document_processor import DocumentProcessor
    from embedding_indexer import EmbeddingIndexer
    import os
    from dotenv import load_dotenv
    
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
    
    chatbot = Chatbot(qa_chain)


    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("Chatbot: Goodbye!")
            break
        response = chatbot.get_response(user_input)
        print(f"Chatbot: {response}")