"""Node: ground
Grades whether the answer is fully supported by the retrieved context.
"""
from langchain.prompts import ChatPromptTemplate
from selfrag.config import get_llm
from selfrag.state import SessionState
from selfrag.utils import remove_think_blocks
import logging

_judge = get_llm(temperature=0)  # deterministic grading

_RUBRIC = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a grader. Reply YES if the answer is completely supported by the context. Otherwise reply NO."
            " Answer with only YES or NO.",
        ),
        (
            "human",
            "Question: {question}\nAnswer: {answer}\nContext:\n{context}",
        ),
    ]
)

def ground(state: SessionState) -> SessionState:
    ctx = "\n---\n".join(state.documents)
    logging.info(f"[Ground] Grading answer for question: {state.query}")
    verdict = _judge.invoke(
        _RUBRIC.format(question=state.query, answer=state.answer, context=ctx)  # type: ignore[arg-type]
    ).content.strip().upper()
    verdict = remove_think_blocks(verdict)
    state.grounded = verdict.startswith("YES")
    state.step += 1
    logging.info(f"[Ground] Verdict: {verdict}, Grounded: {state.grounded}, Step: {state.step}")
    return state 