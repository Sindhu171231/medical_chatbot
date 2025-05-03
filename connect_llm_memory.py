import os
from huggingface_hub import login
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint

# Step 1: Set Hugging Face token from environment variable
HF_TOKEN = os.getenv("HF_TOKEN")  # Make sure this is set in your environment or set it manually here
# Example (not recommended for production):
# HF_TOKEN = "hf_YourTokenHere"

# Step 2: Load the LLM from Hugging Face
def load_llm(huggingface_repo_id):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        task="text-generation",
        temperature=0.5,
        max_new_tokens=512,
        huggingfacehub_api_token=HF_TOKEN
    )
    return llm

# Step 3: Define the custom prompt template
custom_prompt_template = """
You are a helpful and empathetic medical assistant. Use the context provided below to answer the user's medical question as accurately and clearly as possible.

If the context is not sufficient to answer the question, say "I'm sorry, I don't have enough information to answer that accurately. Please consult a healthcare professional."

Context:
{context}

Question:
{question}

Answer in a simple, human-like, and informative manner:
"""

def set_custom_prompt(template):
    return PromptTemplate(template=template, input_variables=["context", "question"])

# Step 4: Load FAISS vectorstore
DB_FAISS_PATH = "vectorstore/db_faiss"  # Make sure this path exists and has your vector store
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

db = FAISS.load_local(
    DB_FAISS_PATH,
    embeddings=embedding_model,
    allow_dangerous_deserialization=True
)

# Step 5: Create the question-answering chain
huggingface_repo_id = "mistralai/Mistral-7B-Instruct-v0.3"

qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(huggingface_repo_id),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": set_custom_prompt(custom_prompt_template)}
)

# Step 6: Get user query and invoke the chain
user_query = input("Write your medical query here: ")
response = qa_chain.invoke({"query": user_query})

# Step 7: Print the result and sources
print("\nAnswer:", response["result"])
print("\nSource Documents:")

