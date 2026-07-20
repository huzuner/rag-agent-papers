"""
app.py — Step 3: a simple Streamlit chat UI on top of agent.py.

Run with:  streamlit run app.py
"""

import streamlit as st
from agent import ask, CONFIDENCE_THRESHOLD

st.set_page_config(page_title="Research RAG Assistant", page_icon="🔎")

st.title("🔎 Uncertainty-Aware Research Assistant")
st.caption(
    "Ask questions about the indexed publications. Answers include a "
    "confidence note when the retrieved passages are a weak match."
)

if "history" not in st.session_state:
    st.session_state.history = []

for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])

question = st.chat_input("Ask a question about the indexed documents...")

if question:
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and reasoning..."):
            result = ask(question)

        st.markdown(result["answer"])

        confidence = result["confidence"]
        attempts = result["attempt"] + 1
        if result["confident"]:
            st.success(f"Confidence: {confidence} (answered on attempt {attempts})")
        else:
            st.warning(
                f"Confidence: {confidence} — below threshold "
                f"({CONFIDENCE_THRESHOLD}) after {attempts} attempt(s)."
            )

        with st.expander("Retrieved passages used for this answer"):
            for i, doc in enumerate(result["docs"], 1):
                st.markdown(f"**Passage {i}:**\n\n{doc}")

    st.session_state.history.append({"role": "assistant", "content": result["answer"]})

with st.sidebar:
    st.subheader("About")
    st.markdown(
        "This assistant retrieves passages from your indexed publications "
        "and answers using a local LLM (via Ollama). If retrieval "
        "confidence is low, it retries with a broadened query before "
        "falling back to a caveated answer instead of guessing."
    )
    st.markdown("Rebuild the index after adding new documents:")
    st.code("python ingest.py", language="bash")
