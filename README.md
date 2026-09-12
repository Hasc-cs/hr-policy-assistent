# 👩‍💼 HR Policy Assistant using RAG

An AI-powered HR Policy Assistant that allows users to upload an HR Policy PDF and ask questions about the policy.

The application uses Retrieval-Augmented Generation (RAG) to retrieve relevant information from the uploaded PDF before generating an answer.

## 🚀 Features

- Upload an HR Policy PDF
- Extract text using PyMuPDF
- Split the policy into chunks
- Create embeddings using Sentence Transformers
- Store embeddings in FAISS
- Retrieve relevant policy sections
- Generate answers using Groq
- Uses `openai/gpt-oss-20b`
- Simple Streamlit user interface
- Shows retrieved policy sections

## 🧠 RAG Architecture

```text
                HR Policy PDF
                     ↓
                 PyMuPDF
                     ↓
              Text Extraction
                     ↓
               Text Chunking
                     ↓
          Sentence Transformers
                     ↓
                 Embeddings
                     ↓
                FAISS Index
                     ↓
User Question → Question Embedding
                     ↓
             Similarity Search
                     ↓
         Relevant Policy Sections
                     ↓
                  Groq LLM
           openai/gpt-oss-20b
                     ↓
                Final Answer
