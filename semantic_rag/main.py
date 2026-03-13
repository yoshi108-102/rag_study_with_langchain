from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv

load_dotenv()

file_path = "./nke-10k-2023.pdf"
loader = PyPDFLoader(file_path)

docs = loader.load()
print(type(docs))
print(len(docs))