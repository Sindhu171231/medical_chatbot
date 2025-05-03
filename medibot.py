import streamlit as st
import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS

DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def set_custom_prompt(template):
    return PromptTemplate(template=template, input_variables=["context", "question"])

def load_llm(huggingface_repo_id, HF_TOKEN):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        task="text-generation",
        temperature=0.5,
        max_new_tokens=512,
        huggingfacehub_api_token=HF_TOKEN
    )
    return llm

def main():
    st.title("ASK Chatbot")
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg['role']):
            st.markdown(msg['content'])

    prompt = st.chat_input("Ask your query here")
    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        huggingface_repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
        HF_TOKEN = os.getenv("HF_TOKEN")
        llm = load_llm(huggingface_repo_id, HF_TOKEN)
        
        try:
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store")
                return

            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                return_source_documents=True,
                chain_type_kwargs={"prompt": set_custom_prompt(custom_prompt_template)}
            )

            response = qa_chain.invoke({'query': prompt})
            result = response['result']
            st.chat_message('assistant').markdown(result)
            st.session_state.messages.append({'role': 'assistant', 'content': result})
        except Exception as e:
            st.error(f"An error occurred: {e}")

custom_prompt_template = """
You are a helpful and empathetic medical assistant. Use the context provided below to answer the user's medical question as accurately and clearly as possible.

If the context is not sufficient to answer the question, say "I'm sorry, I don't have enough information to answer that accurately. Please consult a healthcare professional."

Context:
{context}

Question:
{question}

Answer in a simple, human-like, and informative manner:
"""

if __name__ == "__main__":
    main()
