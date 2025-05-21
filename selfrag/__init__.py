"""SelfRAG – a minimal Retrieve-Augment-Generate pipeline powered by LangGraph."""
from __future__ import annotations

from selfrag.graph import build_graph
import logging

__all__ = ["SelfRAG", "build_graph"]


class SelfRAG:
    """Thin convenience wrapper around the LangGraph graph.

    Example
    -------
    >>> from selfrag import SelfRAG
    >>> rag = SelfRAG()
    >>> print(rag("Who directed Inception?"))
    """

    def __init__(self, max_loops: int = 3):
        logging.info(f"[SelfRAG] Initializing with max_loops={max_loops}")
        self._graph = build_graph(max_loops=max_loops)

    def __call__(self, query: str):
        logging.info(f"[SelfRAG] Invoked with query: {query}")
        result = self._graph.invoke({"query": query})
        logging.info("[SelfRAG] Returning answer.")
        return result["answer"] 