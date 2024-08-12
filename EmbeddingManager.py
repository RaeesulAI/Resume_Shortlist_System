import os
import shutil
import re 
import streamlit as st
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings # for embedding word2vector 
from langchain_community.vectorstores import FAISS # for vector embedding provide by FB


# fuction to create text into chucks
def split_document(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=750,
        separators=["\n\n", "\n# ", "\n- ", "\n\t"],
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    return chunks

# Function to clear previous data
def clear_previous_data():
    if os.path.exists("job_store"):
        shutil.rmtree("job_store")
    if os.path.exists("resume_stores"):
        shutil.rmtree("resume_stores")
    if 'job_id' in st.session_state:
        del st.session_state['job_id']
    if 'resume_ids' in st.session_state:
        del st.session_state['resume_ids']

# function remove special characters
def sanitize_filename(filename):
    # Remove or replace special characters
    filename = re.sub(r'[^\w\-_\. ]', '_', filename)
    # Replace spaces with underscores
    return filename.replace(' ', '_')

# function to convert chunks into vectors
def create_vector_store(chunks, store_name):
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(chunks, embeddings)
    vector_store.save_local(store_name)
    
    return vector_store