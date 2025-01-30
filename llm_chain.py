from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

class SimpleLLM:
    def __init__(self, api_key):
        self.llm = self.get_llm(api_key)
        self.chat_template = self.get_template()

    def get_llm(self, api_key):
        if os.getenv("GEMINI_API_KEY"):
            model = ChatGoogleGenerativeAI(google_api_key=api_key, model="gemini-1.5-flash")
            return model
        else:
            raise ValueError("No valid API key found! Please set one in .env file.")

    def get_template(self):
        chat_template = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a knowledgeable AI assistant.
                        Your goal is to provide accurate and concise answers to user queries.
                        Ensure clarity and structured responses where applicable."""),
            
            HumanMessagePromptTemplate.from_template("""{question}""")
        ])
        return chat_template

    def create_chain(self):
        output_parser = StrOutputParser()
        llm_chain = self.chat_template | self.llm | output_parser
        return llm_chain

if __name__ == "__main__":
    load_dotenv()
    
    GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
    
    simple_llm = SimpleLLM(GOOGLE_API_KEY)
    qa_chain = simple_llm.create_chain()

    query = "What is the capital of France?"
    response = qa_chain.invoke({"question": query})
    print(f"Answer: {response}")
