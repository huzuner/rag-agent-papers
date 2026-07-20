"""
ingest.py — Step 1 of the RAG pipeline.

What this does, in order:
1. Load raw text documents from data/sample_docs/
2. Split them into overlapping chunks (so retrieval can find precise passages,
   not whole documents)
3. Embed each chunk into a vector (a list of numbers capturing its meaning)
4. Store the vectors in a local Chroma database on disk, so we only pay the
   embedding cost once, not every time we ask a question

Run with:  python ingest.py
"""

import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# --- Embedding backend ---------------------------------------------------
# In production (on your own machine, with Ollama running) you'd use:
#
#   from langchain_community.embeddings import OllamaEmbeddings
#   embeddings = OllamaEmbeddings(model="nomic-embed-text")
#
# This sandbox has no internet access to download Ollama or model weights,
# so USE_FAKE_EMBEDDINGS lets us test the full pipeline end-to-end here.
# Switch it to False and use the OllamaEmbeddings line above when you run
# this on your own laptop.
USE_FAKE_EMBEDDINGS = False

if USE_FAKE_EMBEDDINGS:
    from langchain_community.embeddings import FakeEmbeddings
    embeddings = FakeEmbeddings(size=384)
    print("[ingest] Using FakeEmbeddings for local testing.")
    print("[ingest] -> Switch USE_FAKE_EMBEDDINGS to False + use OllamaEmbeddings on your machine.")
else:
    from langchain_ollama import OllamaEmbeddings
    import os
    ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=ollama_host)

DATA_DIR = "data/sample_docs"
PERSIST_DIR = "chroma_db"


def load_and_chunk():
    txt_loader = DirectoryLoader(DATA_DIR, glob="*.txt", loader_cls=TextLoader)
    txt_docs = txt_loader.load()
    print(f"[ingest] Loaded {len(txt_docs)} .txt documents from {DATA_DIR}")

    # PyPDFLoader treats each PDF page as a separate Document. That's fine —
    # our splitter below will further chunk each page as needed, and page
    # numbers are kept in the metadata (useful for citing "page 3" later).
    pdf_loader = DirectoryLoader(DATA_DIR, glob="*.pdf", loader_cls=PyPDFLoader)
    pdf_docs = pdf_loader.load()
    print(f"[ingest] Loaded {len(pdf_docs)} PDF pages from {DATA_DIR}")

    raw_docs = txt_docs + pdf_docs
    print(f"[ingest] Total raw documents: {len(raw_docs)}")

    # chunk_size=500 chars, chunk_overlap=80: overlap prevents a fact from
    # being awkwardly cut in half at a chunk boundary and never fully
    # findable by retrieval.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(raw_docs)
    print(f"[ingest] Split into {len(chunks)} chunks")
    return chunks


def build_vectorstore(chunks):
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
    )
    print(f"[ingest] Stored {len(chunks)} chunk vectors in ./{PERSIST_DIR}")
    return vectordb


if __name__ == "__main__":
    chunks = load_and_chunk()
    build_vectorstore(chunks)
    print("[ingest] Done. You can now run agent.py or app.py to query the store.")
