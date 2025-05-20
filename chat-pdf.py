# Streamlit is used for creating the web interface
import streamlit as st

# PyPDF2 is used to read and extract text from PDFs
from PyPDF2 import PdfReader

# dotenv is used to load environment variables from a .env file
from dotenv import load_dotenv

# LangChain utility to split long text into manageable chunks
from langchain.text_splitter import CharacterTextSplitter

# Gemini-compatible embedding and chat model interfaces
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

# FAISS is a vector store to enable similarity search on embeddings
from langchain_community.vectorstores import FAISS

# Memory component to store conversation history
from langchain.memory import ConversationBufferMemory

# Tool for building a Q&A-style chatbot with retrieval
from langchain.chains import ConversationalRetrievalChain

# Message types for formatting chat history
from langchain.schema import HumanMessage, AIMessage 

# Use Google API key by setting it in place of the OpenAI key (for compatibility)
import os
os.environ["OPENAI_API_KEY"] = os.getenv("GOOGLE_API_KEY")


def get_pdf_content(documents):
    """Extract text from all uploaded PDF documents."""
    raw_text = ""
    for document in documents:
        pdf_reader = PdfReader(document)
        for page in pdf_reader.pages:
            text = page.extract_text()
            if text:
                raw_text += text + "\n"
    return raw_text


def get_chunks(text):
    """Split long text into smaller chunks for easier embedding and retrieval."""
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,       # maximum number of characters per chunk
        chunk_overlap=200,     # overlap between chunks for better context preservation
        length_function=len
    )
    return text_splitter.split_text(text)


def get_embeddings(chunks):
    """Convert text chunks to vector embeddings and store them in FAISS."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_storage = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vector_storage


def start_conversation(vector_embeddings):
    """Set up the conversational chatbot with memory and retriever."""
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")  # Initialize Gemini LLM
    memory = ConversationBufferMemory(
        memory_key='chat_history',     # Used to store the history in LangChain format
        return_messages=True
    )
    conversation = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_embeddings.as_retriever(),
        memory=memory
    )
    return conversation


def process_query(query_text):
    """Handle user input, send it to the chatbot, and display the chat history."""
    response = st.session_state.conversation({'question': query_text})
    st.session_state.chat_history = response["chat_history"]

    # Render the full conversation in the chat UI
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.write(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant", avatar="🤖"):
                st.write(message.content)


def main():
    load_dotenv()  # Load environment variables

    st.title("📄 Chat with PDFs")  # App title
    st.sidebar.header("Upload PDF Files")  # Sidebar instruction

    # Initialize conversation and chat history in Streamlit session state
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    # File uploader UI element in sidebar
    documents = st.sidebar.file_uploader(
        "Upload your PDFs", type=["pdf"], accept_multiple_files=True
    )

    # Button to trigger PDF processing and chatbot initialization
    if st.sidebar.button("Process PDFs"):
        with st.spinner("Processing..."):
            extracted_text = get_pdf_content(documents)

            if not extracted_text.strip():
                st.sidebar.error("No text found in PDFs. Try another file.")
                return

            # Convert extracted text to chunks, get embeddings, and start conversation
            text_chunks = get_chunks(extracted_text)
            vector_embeddings = get_embeddings(text_chunks)
            st.session_state.conversation = start_conversation(vector_embeddings)

        st.sidebar.success("PDFs processed! Start chatting.")

    # Chat input box for user to ask questions
    query = st.chat_input("Ask something about the PDFs:")

    # Respond to user queries if chatbot is initialized
    if query and st.session_state.conversation:
        process_query(query)
    elif query:
        st.warning("Please upload and process PDFs first.")


# Entry point of the script
if __name__ == "__main__":
    main()
