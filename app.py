"""Streamlit UI.  Run: streamlit run app.py --server.fileWatcherType none"""
import os, time, streamlit as st
from src.rag import RAGStore, answer, LLMError

st.set_page_config(page_title="Ask My PDF", page_icon="📄", layout="wide")
st.title("📄 Ask My PDF")

def get_api_key():
    """Key comes from .streamlit/secrets.toml or the GEMINI_API_KEY environment variable."""
    try:
        k = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        k = ""
    return (k or os.getenv("GEMINI_API_KEY", "")).strip()

API_KEY = get_api_key()

@st.cache_resource
def get_store():
    return RAGStore()
store = get_store()

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Documents")
    files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    for f in files or []:
        if f.name not in store.docs:
            with st.spinner(f"Indexing {f.name}..."):
                n = store.add_pdf(f.name, f.read())
            st.success(f"{f.name}: {n} chunks")
    st.caption(f"Indexed: {len(store.docs)} doc(s), {len(store.chunks)} chunks")
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()
    with st.expander("Advanced"):
        min_score = st.slider("Relevance cutoff", 0.0, 1.0, 0.40, 0.05,
                              help="Below this, the bot refuses instead of guessing")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])

if q := st.chat_input("Ask a question about your documents"):
    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.write(q)
    with st.chat_message("assistant"):
        if not store.chunks:
            st.warning("Upload a PDF first.")
        else:
            t0 = time.time()
            hist = [(m["role"], m["content"]) for m in st.session_state.messages[:-1]]
            try:
                ans, hits = answer(store, q, hist, "gemini", API_KEY, min_score)
            except LLMError as e:
                ans, hits = f"⚠️ {e}", []
            except Exception as e:
                ans, hits = f"⚠️ Something went wrong: {e}", []
            st.write(ans)
            st.caption(f"⏱ {time.time() - t0:.1f}s")
            if hits:
                with st.expander("Sources"):
                    for h in hits:
                        st.markdown(f"**{h['doc']} — page {h['page']}**\n\n{h['text'][:300]}...")
            st.session_state.messages.append({"role": "assistant", "content": ans})
