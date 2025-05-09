import streamlit as st
from agents import handle_query

st.set_page_config(
    page_title="RAG Q&A Assistant",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="auto",
)

st.title("RAG-Powered Multi-Agent Q&A Assistant")

query = st.text_input("Ask me anything (e.g. 'define AI', 'calculate 3 * 8', or a general question):")

if query:
    with st.spinner("🔎 Processing your query..."):
        result = handle_query(query)

    st.subheader("🛠️ Selected Tool:")
    st.success(result['tool'])

    if result["context"]:
        st.subheader("Retrieved Context:")
        for i, snippet in enumerate(result["context"], 1):
            st.markdown(f"**Chunk {i}:**")
            st.info(snippet)

    st.subheader("Answer:")
    st.markdown(f"<div style= padding: 12px; border-radius: 8px; font-size: 16px;'>{result['answer']}</div>", unsafe_allow_html=True)

st.markdown("---")
