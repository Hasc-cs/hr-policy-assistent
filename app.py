import streamlit as st
import pymupdf
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="HR Policy Assistant",
    page_icon="👩‍💼",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("👩‍💼 HR Policy Assistant")

st.write(
    "Upload an HR Policy PDF and ask questions about it "
    "using Retrieval-Augmented Generation (RAG)."
)


# ============================================================
# LOAD SENTENCE TRANSFORMER MODEL
# ============================================================

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )


embedding_model = load_embedding_model()


# ============================================================
# EXTRACT TEXT FROM PDF
# ============================================================

def extract_text_from_pdf(uploaded_file):

    pdf_bytes = uploaded_file.read()

    document = pymupdf.open(
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


# ============================================================
# CREATE TEXT CHUNKS
# ============================================================

def create_chunks(
    text,
    chunk_size=500,
    overlap=100
):

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk = " ".join(
            words[start:end]
        )

        if chunk.strip():
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# ============================================================
# CREATE FAISS INDEX
# ============================================================

def create_faiss_index(chunks):

    embeddings = embedding_model.encode(
        chunks,
        convert_to_numpy=True
    )

    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    # Normalize vectors.
    # This allows Inner Product to work
    # approximately as cosine similarity.
    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(embeddings)

    return index


# ============================================================
# RETRIEVE RELEVANT POLICY CHUNKS
# ============================================================

def retrieve_relevant_chunks(
    question,
    index,
    chunks,
    top_k=4
):

    question_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True
    )

    question_embedding = np.asarray(
        question_embedding,
        dtype="float32"
    )

    faiss.normalize_L2(
        question_embedding
    )

    number_of_results = min(
        top_k,
        len(chunks)
    )

    scores, indices = index.search(
        question_embedding,
        number_of_results
    )

    results = []

    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        if idx != -1:

            results.append(
                {
                    "text": chunks[idx],
                    "score": float(score)
                }
            )

    return results


# ============================================================
# GENERATE ANSWER USING GROQ
# ============================================================

def generate_answer(
    question,
    retrieved_chunks,
    groq_api_key
):

    client = Groq(
        api_key=groq_api_key
    )

    context_parts = []

    for i, item in enumerate(
        retrieved_chunks
    ):

        context_parts.append(
            f"""
Policy Section {i + 1}:

{item["text"]}
"""
        )

    context = "\n".join(
        context_parts
    )

    system_prompt = """
You are an HR Policy Assistant.

Your job is to answer questions using ONLY
the HR policy context provided by the application.

Follow these rules:

1. Do not invent information.
2. Do not use outside knowledge.
3. Do not make assumptions.
4. If the answer cannot be found in the provided
   policy context, say:

   "I could not find this information in the uploaded HR policy."

5. Give clear and professional answers.
6. Keep the answer reasonably concise.
"""

    user_prompt = f"""
HR POLICY CONTEXT:

{context}

EMPLOYEE QUESTION:

{question}

Answer the employee's question using only
the HR policy context above.
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


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Settings")

    groq_api_key = st.text_input(
        "Groq API Key",
        type="password",
        help="Enter your Groq API key."
    )

    st.markdown("---")

    st.subheader("🧠 RAG Pipeline")

    st.write(
        """
        PDF
        ↓
        PyMuPDF
        ↓
        Text Chunks
        ↓
        Sentence Transformers
        ↓
        FAISS
        ↓
        Relevant Sections
        ↓
        Groq
        ↓
        Answer
        """
    )


# ============================================================
# PDF UPLOAD
# ============================================================

st.subheader("📄 Upload HR Policy")

uploaded_file = st.file_uploader(
    "Choose an HR Policy PDF",
    type=["pdf"]
)


# ============================================================
# PROCESS PDF
# ============================================================

if uploaded_file is not None:

    file_identifier = (
        uploaded_file.name,
        uploaded_file.size
    )

    # Process only when a new PDF is uploaded
    if (
        "file_identifier" not in st.session_state
        or st.session_state.file_identifier
        != file_identifier
    ):

        with st.spinner(
            "Processing HR Policy PDF..."
        ):

            try:

                extracted_text = (
                    extract_text_from_pdf(
                        uploaded_file
                    )
                )

            except Exception as e:

                st.error(
                    f"Could not read the PDF: {e}"
                )

                st.stop()

            if not extracted_text.strip():

                st.error(
                    "No readable text was found in the PDF."
                )

                st.stop()

            chunks = create_chunks(
                extracted_text
            )

            if not chunks:

                st.error(
                    "Could not create text chunks."
                )

                st.stop()

            with st.spinner(
                "Creating FAISS vector index..."
            ):

                index = create_faiss_index(
                    chunks
                )

            # Save everything in session state
            st.session_state.index = index

            st.session_state.chunks = chunks

            st.session_state.file_identifier = (
                file_identifier
            )

        st.success(
            f"✅ PDF processed successfully! "
            f"{len(chunks)} searchable chunks created."
        )

    else:

        st.success(
            "✅ HR Policy is ready for questions."
        )


# ============================================================
# QUESTION AREA
# ============================================================

st.subheader("💬 Ask a Question")

question = st.text_input(
    "Enter your HR policy question",
    placeholder=(
        "Example: What is the annual leave policy?"
    )
)


# ============================================================
# ASK BUTTON
# ============================================================

if st.button(
    "🔎 Ask HR Assistant",
    type="primary",
    use_container_width=True
):

    # Check API key
    if not groq_api_key:

        st.warning(
            "⚠️ Please enter your Groq API key "
            "in the sidebar."
        )

        st.stop()

    # Check PDF
    if uploaded_file is None:

        st.warning(
            "⚠️ Please upload an HR Policy PDF first."
        )

        st.stop()

    # Check question
    if not question.strip():

        st.warning(
            "⚠️ Please enter a question."
        )

        st.stop()

    # Check FAISS index
    if "index" not in st.session_state:

        st.warning(
            "⚠️ Please wait for the PDF to finish processing."
        )

        st.stop()

    # --------------------------------------------------------
    # RETRIEVAL
    # --------------------------------------------------------

    with st.spinner(
        "🔍 Searching the HR policy..."
    ):

        retrieved_chunks = (
            retrieve_relevant_chunks(
                question,
                st.session_state.index,
                st.session_state.chunks,
                top_k=4
            )
        )

    # --------------------------------------------------------
    # GENERATION
    # --------------------------------------------------------

    try:

        with st.spinner(
            "🤖 Generating answer..."
        ):

            answer = generate_answer(
                question,
                retrieved_chunks,
                groq_api_key
            )

        # ----------------------------------------------------
        # DISPLAY ANSWER
        # ----------------------------------------------------

        st.subheader("🤖 Answer")

        st.success(answer)

    except Exception as e:

        st.error(
            f"❌ Error while generating answer: {e}"
        )

        st.stop()

    # --------------------------------------------------------
    # DISPLAY SOURCES
    # --------------------------------------------------------

    st.subheader("📚 Retrieved Policy Sections")

    for i, item in enumerate(
        retrieved_chunks
    ):

        with st.expander(
            f"Policy Section {i + 1} "
            f"• Similarity: {item['score']:.3f}"
        ):

            st.write(
                item["text"]
            )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "HR Policy Assistant | "
    "Streamlit + RAG + FAISS + "
    "Sentence Transformers + PyMuPDF + Groq"
)
