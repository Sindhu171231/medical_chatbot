# Load raw PDFs
# Create chunks
# Create vector embeddings 
# Store embeddings in FAISS

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS


# Function to load PDF files
def load_pdf_files(data):
    loader = DirectoryLoader(data, glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

# Load PDFs from the 'data/' directory
DATA_PATH = "data/"
documents = load_pdf_files(data=DATA_PATH)
print("length of documnets",len(documents))
#step2:create chunks
def create_chunks(extracted_data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
    text_chunks=text_splitter.split_documents(extracted_data)
    return text_chunks

chunks=create_chunks(documents);
print("length of chunks ",len(chunks));
#step3:create vector embeddings 
def get_embedding_model():
    embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model
#step4:store in FAISS 
embedding_model= get_embedding_model()
DB_FAISS_PATH="vectorstore/db_faiss"
db=FAISS.from_documents(chunks,embedding_model)
db.save_local(DB_FAISS_PATH)
print("done")