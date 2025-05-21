"""Node: retrieve
Fetches top-k documents from the vector database based on the (enhanced) query.
"""
from selfrag.config import get_retriever
from selfrag.state import SessionState
import logging

node_name = "retrieve"

_retriever = get_retriever()


def retrieve(state: SessionState) -> SessionState:
    query = state.enhanced_query or state.query
    logging.info(f"[Retrieve] Retrieving documents for query: {query}")
    docs = _retriever.invoke(query)  # BaseRetriever implements .invoke
    state.documents = [d.page_content for d in docs]
    logging.info(f"[Retrieve] Retrieved {len(state.documents)} documents.")
    return state 