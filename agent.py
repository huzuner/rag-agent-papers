"""
agent.py — Step 2 of the project: a small LangGraph agent on top of the
Chroma store built by ingest.py.

The flow (this is the "agentic" part — not just a single retrieve+answer
call, but a small decision loop):

    retrieve --> assess_confidence --+-- (confident enough) --> answer
                     ^                |
                     |                +-- (not confident, attempts left)
                     |                        |
                     +------------------------+   (broaden the query, retry)
                     |
                     +-- (not confident, out of attempts) --> answer_with_caveat

Why this matters for a portfolio project: a plain RAG script always answers,
even when the retrieved passages are a poor match — which is exactly how
LLMs end up confidently hallucinating. Wrapping retrieval in a small graph
that can (a) retry with a broadened query and (b) explicitly flag low
confidence in the final answer is a simple, honest way to reduce that.

Run with:  python agent.py "your question here"
"""

import sys
from typing import TypedDict, List

from langchain_community.vectorstores import Chroma

# --- Toggle for sandbox testing vs. your own machine ----------------------
# Same idea as in ingest.py: this sandbox has no internet access to Ollama,
# so USE_FAKE_LLM lets us test the graph's control flow (the retry/branch
# logic) without a real model. On your machine, set both flags to False.
USE_FAKE_EMBEDDINGS = False
USE_FAKE_LLM = False

if USE_FAKE_EMBEDDINGS:
    from langchain_community.embeddings import FakeEmbeddings
    embeddings = FakeEmbeddings(size=384)
else:
    from langchain_ollama import OllamaEmbeddings
    import os
    ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=ollama_host)

if not USE_FAKE_LLM:
    from langchain_ollama import ChatOllama
    import os
    ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    llm = ChatOllama(model="llama3.2:1b", temperature=0.2, base_url=ollama_host)

PERSIST_DIR = "chroma_db"

# Below this similarity score, we don't trust the retrieved passage enough
# to answer confidently on the first try. This is a simple heuristic, not
# a calibrated probability — a natural next improvement (and a good
# interview talking point given a Bayesian background) would be to replace
# it with a properly calibrated confidence estimate.
CONFIDENCE_THRESHOLD = 0.35
MAX_ATTEMPTS = 2


class AgentState(TypedDict):
    question: str
    query: str            # the query actually sent to the retriever (may be reformulated)
    attempt: int
    docs: List[str]
    confidence: float
    answer: str
    confident: bool


def _load_vectorstore():
    return Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)


def retrieve_node(state: AgentState) -> AgentState:
    vectordb = _load_vectorstore()
    results = vectordb.similarity_search_with_relevance_scores(state["query"], k=3)

    if not results:
        state["docs"] = []
        state["confidence"] = 0.0
        return state

    docs = [doc.page_content for doc, _score in results]
    # relevance scores are already normalized to roughly [0, 1] by Chroma
    # (higher = more relevant); we take the top hit as our confidence proxy.
    top_score = max(score for _doc, score in results)

    state["docs"] = docs
    state["confidence"] = round(top_score, 3)
    print(f"[agent] attempt {state['attempt']}: query={state['query']!r} "
          f"-> top confidence={state['confidence']}")
    return state


def assess_confidence_node(state: AgentState) -> AgentState:
    state["confident"] = state["confidence"] >= CONFIDENCE_THRESHOLD
    return state


def broaden_query_node(state: AgentState) -> AgentState:
    # Simple, transparent broadening strategy: strip the query down to its
    # key nouns by just re-using the original question without extra
    # qualifiers. In a more advanced version, an LLM call here could
    # rewrite the query — kept simple and deterministic for this project.
    state["attempt"] += 1
    state["query"] = state["question"]
    return state


def _route_after_confidence(state: AgentState) -> str:
    if state["confident"]:
        return "answer"
    if state["attempt"] < MAX_ATTEMPTS:
        return "broaden_query"
    return "answer_with_caveat"


def answer_node(state: AgentState) -> AgentState:
    context = "\n\n".join(state["docs"])
    state["answer"] = _generate_answer(state["question"], context)
    return state


def answer_with_caveat_node(state: AgentState) -> AgentState:
    context = "\n\n".join(state["docs"]) if state["docs"] else ""
    base_answer = _generate_answer(state["question"], context) if context else (
        "I couldn't find a passage in the indexed documents that confidently "
        "answers this question."
    )
    state["answer"] = (
        base_answer
        + "\n\n[Note: retrieval confidence was low for this question — "
          "treat this answer with caution and consider checking the source "
          "documents directly.]"
    )
    return state


def _generate_answer(question: str, context: str) -> str:
    prompt = (
        "Answer the question using only the context below. "
        "If the context doesn't contain the answer, say so plainly.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    if USE_FAKE_LLM:
        # Deterministic stub so the graph is testable without a real model.
        return f"[FAKE LLM OUTPUT] Would answer '{question}' using the retrieved context above."
    else:
        response = llm.invoke(prompt)
        return response.content


def build_graph():
    from langgraph.graph import StateGraph, END

    graph = StateGraph(AgentState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("assess_confidence", assess_confidence_node)
    graph.add_node("broaden_query", broaden_query_node)
    graph.add_node("answer", answer_node)
    graph.add_node("answer_with_caveat", answer_with_caveat_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "assess_confidence")
    graph.add_conditional_edges(
        "assess_confidence",
        _route_after_confidence,
        {
            "answer": "answer",
            "broaden_query": "broaden_query",
            "answer_with_caveat": "answer_with_caveat",
        },
    )
    graph.add_edge("broaden_query", "retrieve")
    graph.add_edge("answer", END)
    graph.add_edge("answer_with_caveat", END)

    return graph.compile()


def ask(question: str) -> AgentState:
    app = build_graph()
    initial_state: AgentState = {
        "question": question,
        "query": question,
        "attempt": 0,
        "docs": [],
        "confidence": 0.0,
        "answer": "",
        "confident": False,
    }
    final_state = app.invoke(initial_state)
    return final_state


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "What is Orthanq and what problem does it solve?"
    result = ask(question)
    print("\n--- Final answer ---")
    print(result["answer"])
    print(f"\n(confidence: {result['confidence']}, attempts used: {result['attempt'] + 1})")
