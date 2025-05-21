"""Node: enhance_query
Rewrites the raw user query to maximise recall during retrieval.
"""
from langchain.prompts import ChatPromptTemplate
from selfrag.config import get_llm
from selfrag.state import SessionState
from selfrag.utils import remove_think_blocks
import logging

# LLM instance (singleton)
_llm = get_llm()

_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a search assistant. Improve the user query for retrieval."
            " Rewrite it and add keywords so that a similarity search will find more relevant documents."
            " Keep it short (one sentence).",
        ),
        ("human", "{query}"),
    ]
)

def enhance(state: SessionState) -> SessionState:
    """Rewrite the query and store it in state.enhanced_query."""
    logging.info(f"[EnhanceQuery] Enhancing query: {state.query}")
    response = _llm.invoke(_PROMPT.format(query=state.query))  # type: ignore[arg-type]
    response = remove_think_blocks(response.content.strip())
    state.enhanced_query = response
    logging.info(f"[EnhanceQuery] Enhanced query: {state.enhanced_query}")
    return state 