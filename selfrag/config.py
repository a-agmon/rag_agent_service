"""Central configuration helpers for SelfRAG.

Everything that needs initialising exactly once (chat model, embeddings DB, 
retriever) lives here to ensure the same objects are reused across nodes.
"""
from __future__ import annotations

import os
import logging
from typing import List

import polars as pl
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_ollama import ChatOllama

# ---------------------------------------------------------------------------
# Custom DFEmbedder retriever ------------------------------------------------
# ---------------------------------------------------------------------------

try:
    from dfembed import DfEmbedder  # type: ignore
except ModuleNotFoundError as e:
    raise ImportError(
        "DfEmbedder package not found. Make sure it is installed in your environment."
    ) from e

_DB_NAME = os.getenv("TMDB_DB", "tmdb_db")
_CSV_PATH = os.getenv("TMDB_CSV", "tmdb.csv")

_embedder: DfEmbedder | None = None
_retriever: "DfEmbedderRetriever" | None = None

class DfEmbedderRetriever(BaseRetriever):
    """LangChain-style wrapper around the dfembed similarity search."""

    df_embedder: DfEmbedder
    table_name: str = "films"
    k: int = 5

    def _get_relevant_documents(self, query: str) -> List[Document]:  # type: ignore[override]
        logging.info("[Retriever] Finding similar docs for query: %s", query)
        similar = self.df_embedder.find_similar(
            query=query, table_name=self.table_name, k=self.k
        )
        docs = [Document(page_content=txt) for txt in similar]
        logging.info("[Retriever] Returned %d docs", len(docs))
        return docs


def _lazy_embedder() -> DfEmbedder:
    """Create (or load) the vector database lazily on first use."""
    global _embedder
    if _embedder is not None:
        return _embedder

    if not os.path.exists(_DB_NAME):
        logging.info("Vector DB not found – building from %s", _CSV_PATH)
        df = pl.read_csv(_CSV_PATH)
        _embedder = DfEmbedder(database_name=_DB_NAME)
        _embedder.index_table(df.to_arrow(), table_name="films")
    else:
        _embedder = DfEmbedder(database_name=_DB_NAME)
    return _embedder


def get_retriever(k: int = 5) -> DfEmbedderRetriever:
    """Return a singleton retriever instance."""
    global _retriever
    if _retriever is None:
        _retriever = DfEmbedderRetriever(df_embedder=_lazy_embedder(), k=k)
    return _retriever

# ---------------------------------------------------------------------------
# Chat model ----------------------------------------------------------------
# ---------------------------------------------------------------------------

_llm: ChatOllama | None = None


def get_llm(temperature: float = 0.7) -> ChatOllama:
    """Singleton for the Ollama chat model."""
    global _llm
    if _llm is None:
        _llm = ChatOllama(model="qwen3:8b", temperature=temperature)
    return _llm 