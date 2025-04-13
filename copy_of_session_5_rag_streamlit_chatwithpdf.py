# -*- coding: utf-8 -*-
"""
Building a Retrieval Augmented Generation (RAG) Chatbot
Using Gemini, LangChain, and ChromaDB

This script provides the backend components for a RAG chatbot system.
"""

# Import required libraries
import os
import pdfplumber
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from session_4_rag_backend import (
    setup_api_key,
    upload_pdf,
    parse_pdf,
    create_document_chunks,
    init_embedding_model,
    embed_documents,
    store_embeddings,
    get_context_from_chunks,
    query_with_full_context
)

# Configure Streamlit app
st.set_page_config(
    page_title="RAG Chatbot with Gemini",
    page_icon="📚",
    layout="wide"
)
# Initialize Streamlit session state
if "conversation" not in st.session_state:
    st.session_state.conversation = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = None
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

# Main function for the app
def main():
    # Sidebar for configuration
    with st.sidebar:
        st.title("RAG Chatbot")
        st.subheader("Configuration")
        
        # API Key input
        api_key = st.text_input("Enter Gemini API Key:", type="password")
        if api_key and st.button("Set API Key"):
            setup_api_key(api_key)
            st.success("API Key set successfully!")
        
        st.divider()
        
        # File uploader
        st.subheader("Upload Documents")
        uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
        if uploaded_files and st.button("Process Documents"):
            process_documents(uploaded_files)
        
        # Display processed files
        if st.session_state.processed_files:
            st.subheader("Processed Documents")
            for file in st.session_state.processed_files:
                st.write(f"- {file}")
        
        st.divider()
        
        # Advanced options
        with st.expander("Advanced Options"):
            st.slider("Number of chunks to retrieve (k)", min_value=1, max_value=10, value=3, key="k_value")
            st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.1, key="temperature")

    # Main app area
    st.title("Retrieval Augmented Generation Chatbot")
    
    if not st.session_state.vectorstore:
        st.info("Please upload and process documents to start chatting.")
        
        # Usage instructions
        with st.expander("How to use this app"):
            st.markdown("""
            1. Enter your Gemini API Key in the sidebar.
            2. Upload one or more PDF documents.
            3. Click "Process Documents" to analyze them.
            4. Ask questions about the documents in the chat.
            
            The system uses Retrieval Augmented Generation to provide answers based on your documents.
            """)
    else:
        # Chat interface
        display_chat()
        
        # User query handling
        user_query = st.chat_input("Ask a question about your documents...")
        if user_query:
            handle_user_query(user_query)
# Process uploaded PDF documents
def process_documents(uploaded_files):
    """Processes uploaded PDF documents and creates a vector store."""
    try:
        # Initialize progress indicators
        progress_bar = st.sidebar.progress(0)
        status_text = st.sidebar.empty()

        # Initialize embedding model if needed
        if st.session_state.embedding_model is None:
            status_text.text("Initializing embedding model...")
            st.session_state.embedding_model = init_embedding_model()
            if st.session_state.embedding_model is None:
                st.sidebar.error("Failed to initialize embedding model. Check your API key.")
                return

        # Process uploaded files
        all_chunks = []
        processed_file_names = []
        for i, uploaded_file in enumerate(uploaded_files):
            progress_bar.progress(int((i / len(uploaded_files)) * 100))
            status_text.text(f"Processing {uploaded_file.name}...")

            # Save file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                pdf_path = tmp_file.name

            # Extract and process document content
            pdf_file = upload_pdf(pdf_path)
            if not pdf_file:
                st.sidebar.warning(f"Failed to process {uploaded_file.name}")
                continue
            text = parse_pdf(pdf_file)
            if not text:
                st.sidebar.warning(f"Failed to extract text from {uploaded_file.name}")
                continue

            # Chunk document and add metadata
            chunks = create_document_chunks(text)
            if chunks:
                all_chunks.extend([{"content": chunk, "source": uploaded_file.name} for chunk in chunks])
                processed_file_names.append(uploaded_file.name)
            else:
                st.sidebar.warning(f"Failed to create chunks from {uploaded_file.name}")
            
            # Clean up temp files
            os.unlink(pdf_path)

        # Update progress and store embeddings
        progress_bar.progress(100)
        if all_chunks:
            vectorstore = store_embeddings(
                st.session_state.embedding_model,
                [chunk["content"] for chunk in all_chunks],
                persist_directory="./streamlit_chroma_db"
            )
            if vectorstore:
                st.session_state.vectorstore = vectorstore
                st.session_state.processed_files = processed_file_names
                st.sidebar.success(f"Processed {len(processed_file_names)} documents successfully!")
            else:
                st.sidebar.error("Failed to create vector database.")
        else:
            st.sidebar.error("No valid data extracted from documents.")

        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()

    except Exception as e:
        st.sidebar.error(f"Error processing documents: {str(e)}")


# Handle user queries
def handle_user_query(query):
    """Processes user queries and displays responses."""
    if not st.session_state.vectorstore:
        st.error("Please process documents before asking questions.")
        return

    # Display user query and "thinking" message
    st.session_state.conversation.append({"role": "user", "content": query})
    thinking_placeholder = st.info("🤔 Thinking...")

    try:
        # Query the RAG system
        response, context, _ = query_with_full_context(
            query,
            st.session_state.vectorstore,
            k=st.session_state.k_value,
            temperature=st.session_state.temperature
        )

        # Append assistant response
        st.session_state.conversation.append({"role": "assistant", "content": response, "context": context})
        thinking_placeholder.empty()
        display_chat()

    except Exception as e:
        thinking_placeholder.empty()
        st.session_state.conversation.append({"role": "assistant", "content": f"Error: {str(e)}"})
        display_chat()


# Display the chat interface
def display_chat():
    """Displays the conversation between the user and the assistant."""
    for message in st.session_state.conversation:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message.get("context"):
                with st.expander("View source context"):
                    st.text(message["context"])


# Reset conversation
def reset_conversation():
    """Clears the conversation history."""
    st.session_state.conversation = []


# Main entry point
if __name__ == "__main__":
    main()
