import streamlit as st
import fitz  # PyMuPDF
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq


# -------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------

st.set_page_config(
    page_title="HR Policy Assistant",
    page_icon="👩‍💼",
    layout="wide"
)

st.title("👩‍💼 HR Policy Assistant")
st.write(
    "Upload an HR Policy PDF and ask questions about it using RAG."
)


# -------------------------------------------------
# LOAD EMBEDDING MODEL
# -------------------------------------------------

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )


embedding_model = load_embedding_model()


# -------------------------------------------------
# EXTRACT TEXT FROM PDF
# -------------------------------------------------

def extract_text_from_pdf(uploaded_file):
    pdf_bytes = uploaded_file.read()

    document = fitz.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    pages = []

    for page in document:
        text = page.get_text()

        if text.strip():
            pages.append(text)

    document.close()

    return "\n".join(pages)


# -------------------------------------------------
# CREATE TEXT CHUNKS
# -------------------------------------------------

def create_chunks(text, chunk_size=500, overlap=100):
    words = text.split()
    chunks = []

    start = 0

    while start < len(words):
        end = start + chunk_size

        chunk = " ".join(words[start:end])

        if chunk.strip():
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# -------------------------------------------------
# CREATE FAISS VECTOR INDEX
# -------------------------------------------------

def create_faiss_index(chunks):
    embeddings = embedding_model.encode(
        chunks,
        convert_to_numpy=True
    )

    embeddings = np.array(
        embeddings,
        dtype="float32"
    )

    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    return index


# -------------------------------------------------
# RETRIEVE RELEVANT CHUNKS
# -------------------------------------------------

def retrieve_relevant_chunks(
    question,
    index,
    chunks,
    top_k=4
):
    question_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True
    ).astype("float32")

    faiss.normalize_L2(question_embedding)

    top_k = min(top_k, len(chunks))

    scores, indices = index.search(
        question_embedding,
        top_k
    )

    retrieved_chunks = []

    for score, idx in zip(
        scores[0],
        indices[0]
    ):
        if idx != -1:
            retrieved_chunks.append(
                {
                    "text": chunks[idx],
                    "score": float(score)
                }
            )

    return retrieved_chunks


# -------------------------------------------------
# ASK GROQ MODEL
# -------------------------------------------------

def generate_answer(
    question,
    retrieved_chunks,
    groq_api_key
):
    client = Groq(api_key=groq_api_key)

    context = "\n\n".join(
        [
            f"Policy Section {i + 1}:\n{item['text']}"
            for i, item in enumerate(retrieved_chunks)
        ]
    )

    system_prompt = """
You are an HR Policy Assistant.

Answer the user's question ONLY using the HR policy context provided.

Rules:
- Do not invent information.
- Do not use outside knowledge.
- If the answer is not in the provided policy context, say:
  "I could not find this information in the uploaded HR policy."
- Keep answers clear, helpful, and professional.
"""

    user_prompt = f"""
HR POLICY CONTEXT:

{context}

QUESTION:

{question}
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        temperature=0.2,
        max_tokens=600
    )

    return response.choices[0].message.content


# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------

with st.sidebar:
    st.header("⚙️ Settings")

    groq_api_key = st.text_input(
        "Groq API Key",
        type="password"
    )

    st.info(
        "Your API key is used only to send your question "
        "and retrieved HR policy context to Groq."
    )


# -------------------------------------------------
# FILE UPLOAD
# -------------------------------------------------

uploaded_file = st.file_uploader(
    "📄 Upload your HR Policy PDF",
    type=["pdf"]
)


# -------------------------------------------------
# PROCESS PDF
# -------------------------------------------------

if uploaded_file is not None:

    file_identifier = (
        uploaded_file.name,
        uploaded_file.size
    )

    if (
        "file_identifier" not in st.session_state
        or st.session_state.file_identifier != file_identifier
    ):

        with st.spinner("Processing your HR Policy PDF..."):

            text = extract_text_from_pdf(uploaded_file)

            if not text.strip():
                st.error(
                    "No readable text was found in this PDF."
                )
                st.stop()

            chunks = create_chunks(text)

            if len(chunks) == 0:
                st.error(
                    "Could not create text chunks from this PDF."
                )
                st.stop()

            index = create_faiss_index(chunks)

            st.session_state.index = index
            st.session_state.chunks = chunks
            st.session_state.file_identifier = file_identifier

        st.success(
            f"PDF processed successfully! "
            f"{len(chunks)} searchable chunks created."
        )

    else:
        st.success("HR Policy is ready for questions.")


# -------------------------------------------------
# QUESTION SECTION
# -------------------------------------------------

st.subheader("💬 Ask a Question")

question = st.text_input(
    "Example: What is the annual leave policy?",
    placeholder="Type your HR policy question here..."
)


if st.button("Ask HR Assistant", type="primary"):

    if not groq_api_key:
        st.warning(
            "Please enter your Groq API key in the sidebar."
        )
        st.stop()

    if uploaded_file is None:
        st.warning(
            "Please upload an HR Policy PDF first."
        )
        st.stop()

    if not question.strip():
        st.warning(
            "Please enter a question."
        )
        st.stop()

    if "index" not in st.session_state:
        st.warning(
            "Please wait until the PDF finishes processing."
        )
        st.stop()

    # Step 1: Retrieve relevant chunks
    with st.spinner("Searching the HR policy..."):
        retrieved_chunks = retrieve_relevant_chunks(
            question,
            st.session_state.index,
            st.session_state.chunks,
            top_k=4
        )

    # Step 2: Generate answer
    try:
        with st.spinner("Generating answer..."):
            answer = generate_answer(
                question,
                retrieved_chunks,
                groq_api_key
            )

        st.subheader("🤖 Answer")
        st.success(answer)

    except Exception as e:
        st.error(f"Error: {str(e)}")

    # Show retrieved sections
    with st.expander("📚 View relevant policy sections"):

        for i, item in enumerate(retrieved_chunks):
            st.markdown(
                f"### Section {i + 1} "
                f"(Similarity: {item['score']:.2f})"
            )
            st.write(item["text"])


# -------------------------------------------------
# FOOTER
# -------------------------------------------------

st.markdown("---")

st.caption(
    "Built with Streamlit, FAISS, Sentence Transformers, "
    "PyMuPDF, and Groq."
)
