"""Node: generate
Drafts an answer from the retrieved context.
"""
from langchain.prompts import ChatPromptTemplate
from selfrag.config import get_llm
from selfrag.state import SessionState
from selfrag.utils import remove_think_blocks
import logging

_llm = get_llm()

_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert assistant. Answer the question using ONLY the provided context."
            " If the context is insufficient, say you don't know. Cite sources by mentioning any film title you use.",
        ),
        (
            "human",
            "Question: {question}\n\nContext:\n{context}",
        ),
    ]
)

def generate(state: SessionState) -> SessionState:
    context = "\n---\n".join(state.documents)
    logging.info(f"[Generate] Generating answer for question: {state.query}")
    resp = _llm.invoke(
        _PROMPT.format(question=state.query, context=context)  # type: ignore[arg-type]
    )
    answer = remove_think_blocks(resp.content.strip())
    state.answer = resp.content.strip()
    logging.info(f"[Generate] Generated answer: {state.answer}")
    return state 