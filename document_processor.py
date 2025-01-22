import nltk
#nltk.download('punkt_tab')

from langchain.schema import Document
from langchain_text_splitters import NLTKTextSplitter

class DocumentProcessor:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_and_split(self):
        with open(self.file_path, 'r') as file:
            text = file.read()

        documents = [Document(page_content=text)]

        text_splitter = NLTKTextSplitter(chunk_size=500, chunk_overlap=100)

        chunks = text_splitter.split_documents(documents)
        return chunks


if __name__ == "__main__":
    processor = DocumentProcessor("transcription.txt")
    texts = processor.load_and_split()
    print(f"Processed {len(texts)} text chunks")