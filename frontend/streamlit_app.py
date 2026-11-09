"""DocuMind web interface. Run from the project root with:
    streamlit run frontend/streamlit_app.py
"""
import sys
from pathlib import Path

# Let this file import the `app` package that lives one folder up
sys.path.append(str(Path(__file__).resolve().parent.parent))

import streamlit as st

from app.rag.generator import answer_question
from app.rag.ingestion import ingest_pdf, list_documents

st.set_page_config(page_title="DocuMind", page_icon="📄")
st.title("DocuMind")
st.caption("Upload research papers, then ask questions. Answers cite their sources.")

with st.sidebar:
    st.header("Documents")
    files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    if files and st.button("Index documents"):
        for f in files:
            with st.spinner(f"Indexing {f.name}..."):
                n = ingest_pdf(f.getvalue(), f.name)
            st.success(f"{f.name}: {n} chunks") if n else st.warning(f"{f.name}: no text found")
    st.write("**Indexed:**", ", ".join(list_documents()) or "nothing yet")

question = st.text_input("Ask a question about your documents")
if question:
    with st.spinner("Searching your documents..."):
        result = answer_question(question)
    st.markdown(result["answer"])
    with st.expander(f"Retrieved passages ({len(result['sources'])})"):
        for s in result["sources"]:
            st.markdown(f"**{s['source']}, p.{s['page']}** (similarity {s['score']})")
            st.text(s["text"][:500])
