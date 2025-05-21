# SelfRAG – LangGraph Retrieval-Augmented Generation

Minimal reference implementation of the *Self-RAG* loop:

```
┌─► enhance_query ─► retrieve ─► generate_answer ─► ground_check ─┐
|                                                                |
└────────── repeat while ground_check == "NOT GROUNDED" ─────────┘
```

---

## What is an Agentic Microservice?

An **agentic microservice** is a self-contained service that encapsulates an autonomous agent capable of reasoning, decision-making, and acting within a defined workflow. Unlike traditional microservices, which typically expose static endpoints for CRUD operations or business logic, agentic microservices embed an intelligent agent that can:

- Perceive and process complex inputs (e.g., natural language queries)
- Plan and execute multi-step reasoning or actions
- Adapt its behavior based on feedback or intermediate results
- Interact with external tools, databases, or APIs as part of its workflow

### How This Project Implements an Agentic Microservice

In this repository, the agentic microservice is realized as a FastAPI application that exposes a `/trigger` endpoint. When a request is received, the service:

1. **Enhances the input query** for better retrieval.
2. **Retrieves relevant context** from a vector database.
3. **Generates an answer** using a local language model.
4. **Checks if the answer is grounded** in retrieved evidence.
5. **Repeats the loop** if the answer is not sufficiently grounded, refining the query and context.

This loop is orchestrated using [LangGraph](https://github.com/langchain-ai/langgraph), and each step is modularized as a node (see the `nodes/` directory).

**Diagram:**
```
User Query ─► [Agentic Microservice]
                 │
                 ▼
      ┌─► enhance_query ─► retrieve ─► generate_answer ─► ground_check ─┐
      |                                                                |
      └────────── repeat while ground_check == "NOT GROUNDED" ─────────┘
                 │
                 ▼
              Answer
```

* **Language model:** local Ollama `qwen3:8b` (change in `selfrag/config.py`).
* **Vector DB:** custom `DfEmbedder` backed by the `tmdb_db` folder (built from `tmdb.csv` on first run).
* **Frameworks:** [LangChain](https://github.com/langchain-ai/langchain), [LangGraph](https://github.com/langchain-ai/langgraph).

## Quick start

```bash
pip install -r requirements.txt           # install deps (see pyproject.toml for poetry)
export OLLAMA_HOST=http://localhost:11434 # adjust if needed
python examples/demo_cli.py
```

---

## API Usage

A FastAPI service is available to trigger the SelfRAG agent via HTTP.

### Run the API server

```bash
uvicorn api:app --reload
```

### Call the /trigger endpoint with curl

```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"query": "Who directed Inception?"}' \
  http://localhost:8000/trigger
```

**Response:**
```json
{
  "answer": "Christopher Nolan"
}
```

See `selfrag/graph.py` for the workflow definition and the `nodes/` package for each individual step. 