from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class SessionState:
    """Data object that flows between LangGraph nodes."""

    # Raw user query
    query: str

    # Rewritten query optimised for retrieval
    enhanced_query: Optional[str] = None

    # Retrieved context chunks (plain text)
    documents: List[str] = field(default_factory=list)

    # Draft answer produced by the language model
    answer: Optional[str] = None

    # Whether the answer is fully grounded in the context
    grounded: Optional[bool] = None

    # Loop counter – prevents infinite self-RAG loops
    step: int = 0 