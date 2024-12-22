import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.spacy_embeddings import SpacyEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
import os
from langchain.schema import BaseMessage

# Load environment variables
load_dotenv()

# Initialize embeddings
embeddings = SpacyEmbeddings(model_name="en_core_web_sm")

# Extract text from PDF files
def extract_text_from_pdf(pdf_files):
    text_data = []
    for file in pdf_files:
        pdf_reader = PdfReader(file)
        for page_num, page in enumerate(pdf_reader.pages, start=1):
            text_data.append({"text": page.extract_text(), "page": page_num, "source": file.name})
    return text_data

# Split text into manageable chunks
def chunk_text(data, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = []
    for item in data:
        split_chunks = text_splitter.split_text(item["text"])
        chunks.extend([{"text": chunk, "page": item["page"], "source": item["source"]} for chunk in split_chunks])
    return chunks

# Create and save a vector store
def create_vector_store(chunks):
    texts = [chunk["text"] for chunk in chunks]
    metadata = [{"page": chunk["page"], "source": chunk["source"]} for chunk in chunks]
    vector_db = FAISS.from_texts(texts, embeddings, metadatas=metadata)
    vector_db.save_local("faiss_db")
    return vector_db

# Retrieve relevant chunks based on the query
def retrieve_relevant_chunks(query, vector_db):
    retriever = vector_db.as_retriever()
    relevant_documents = retriever.get_relevant_documents(query)
    return relevant_documents  # Returns a list of Document objects

# Generate a response using the LLM
def generate_response(query, relevant_chunks):
    # Initialize the LLM
    llm = ChatAnthropic(model="claude-3-sonnet-20240229", temperature=0, api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    # Combine the retrieved chunks into a single context
    context = "\n".join([chunk.page_content for chunk in relevant_chunks])  # Use .page_content
    
    # Define a prompt template
    prompt = f"""
    You are a helpful assistant. Answer queries using only the provided context.
    Question: {query}
    Context: {context}
    """
    
    # Create the messages correctly
    system_message = BaseMessage(role="system", content="You are a helpful assistant.")
    user_message = BaseMessage(role="user", content=prompt)
    
    # Generate a response from the LLM
    response = llm.generate([system_message, user_message])
    
    return response.generations[0].message["content"]

# Handle user input and generate a response
def handle_user_query(query, vector_db):
    relevant_chunks = retrieve_relevant_chunks(query, vector_db)
    if not relevant_chunks:
        return "No relevant information found in the provided context."
    return generate_response(query, relevant_chunks)

# Main Streamlit app
def main():
    st.set_page_config(page_title="Chat with PDF using RAG")
    st.title("RAG-based Chat with PDFs")
    
    # Sidebar for uploading and processing PDFs
    with st.sidebar:
        st.header("Upload PDFs")
        pdf_files = st.file_uploader("Upload your PDF files", accept_multiple_files=True, type=["pdf"])
        
        if st.button("Process PDFs"):
            if pdf_files:
                with st.spinner("Processing PDFs..."):
                    text_data = extract_text_from_pdf(pdf_files)
                    chunks = chunk_text(text_data)
                    create_vector_store(chunks)
                    st.success("PDFs processed successfully!")
            else:
                st.error("Please upload at least one PDF file.")

    # Main input for querying
    query = st.text_input("Ask a question based on the uploaded PDFs")
    
    if query:
        try:
            vector_db = FAISS.load_local("faiss_db", embeddings, allow_dangerous_deserialization=True)
            with st.spinner("Fetching answer..."):
                response = handle_user_query(query, vector_db)
                st.write(response)
        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
