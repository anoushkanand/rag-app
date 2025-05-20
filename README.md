# 🤖 Chat with PDFs using Gemini

This Streamlit app allows you to upload one or more PDF documents and interact with them via a conversational interface powered by **Gemini (Google Generative AI)**. It uses **LangChain**, **FAISS**, and **Google Generative AI Embeddings** to perform semantic search and question-answering over your documents.

---

## 🧠 Features

- 📄 Upload and parse PDF documents
- 🧩 Split documents into manageable text chunks
- 📌 Embed text using Gemini's embedding model (`embedding-001`)
- 🧠 Store embeddings in a FAISS vector store
- 💬 Ask questions about the content and get AI-generated answers
- 🔁 Maintains conversation history using memory buffer

## 📦 Tech Stack

- Streamlit: Web Interface
- Langchain: LLM Integration
- Google Generative AI: Gemini Models
- FAISS: Vector Database
- PyPDF2: PDF Parsing

---
### To make it work:
- Create a .env file in the root directory and add your Gemini API key:
- GOOGLE_API_KEY=your_google_genai_api_key
