# Agentic RAG Microservice Example with LangGraph

This repository demonstrates how to build an **agentic microservice** for Retrieval-Augmented Generation (RAG) using [LangGraph](https://github.com/langchain-ai/langgraph). The service combines autonomous agent reasoning with retrieval and generation, providing a robust pattern for building intelligent, self-improving APIs.

---

## What is an Agentic RAG Microservice?

An **agentic microservice** is a self-contained service that embeds an intelligent agent capable of:

- Understanding and enhancing user queries
- Retrieving relevant context from a knowledge base or vector database
- Generating answers using a language model
- Assessing the quality or groundedness of its own answers
- Iteratively refining its process until a satisfactory, grounded answer is produced

This pattern goes beyond traditional microservices by enabling adaptive, multi-step reasoning and self-correction within a single API endpoint.

---

## How This Example Works

This repo implements an agentic RAG microservice as a FastAPI application with a `/trigger` endpoint. The workflow is orchestrated by LangGraph and consists of the following loop:

```
┌─► enhance_query ─► retrieve ─► generate_answer ─► ground_check ─┐
|                                                                |
└────────── repeat while ground_check == "NOT GROUNDED" ─────────┘
```

**Key components:**
- **Query Enhancement:** Improves the user's question for better retrieval.
- **Retriever:** Finds relevant context from a vector database (custom `DfEmbedder`).
- **Generator:** Uses a local language model (Ollama `qwen3:8b` by default) to answer.
- **Grounding Check:** Verifies if the answer is well-supported by retrieved evidence.
- **Self-Improvement Loop:** If not grounded, the agent refines its process and tries again.

**Directory structure:**
- `selfrag/config.py` – Central configuration for models and retrievers
- `selfrag/graph.py` – LangGraph workflow definition
- `nodes/` – Modular steps for each node in the agent loop

---

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   # (Or use poetry with pyproject.toml)
   ```

2. **Set up Ollama (local LLM):**
   ```bash
   export OLLAMA_HOST=http://localhost:11434
   # Adjust if your Ollama instance runs elsewhere
   ```

3. **Run the CLI demo:**
   ```bash
   python examples/demo_cli.py
   ```

---

## Running as an API Microservice

Start the FastAPI server:
```bash
uvicorn api:app --reload
```

Trigger the agent via HTTP:
```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"query": "Who directed Inception?"}' \
  http://localhost:8000/trigger
```

**Sample response:**
```json
{
  "answer": "Christopher Nolan"
}
```

---

## Customization

- **Change the language model:** Edit the model name in `selfrag/config.py`.
- **Swap out the retriever:** Replace or extend the `DfEmbedder` logic.
- **Modify the workflow:** Adjust the LangGraph definition in `selfrag/graph.py` or add new nodes in `nodes/`.

---

## Why Use This Pattern?

- **Autonomy:** The agent can reason, self-correct, and adapt its answers.
- **Modularity:** Each step is a node, making it easy to extend or swap components.
- **Production-ready:** Exposes a simple API endpoint for integration into larger systems.

---

## References

- [LangGraph](https://github.com/langchain-ai/langgraph)
- [LangChain](https://github.com/langchain-ai/langchain)
- [Ollama](https://ollama.com/)
- [DfEmbedder](https://github.com/dfembed/dfembed) (custom vector DB)

---

## License

MIT 