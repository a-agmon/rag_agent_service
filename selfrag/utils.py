import re

def remove_think_blocks(text: str) -> str:
    """
    Removes all <THINK>...</THINK> blocks from the input text.
    This is useful for cleaning LLM outputs that include reasoning in these tags.
    """
    return re.sub(r"<THINK>.*?</THINK>", "", text, flags=re.DOTALL | re.IGNORECASE).strip() 