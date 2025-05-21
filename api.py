import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from selfrag import SelfRAG

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

app = FastAPI(title="SelfRAG Agent API")
rag_agent = SelfRAG()

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str

@app.post("/trigger", response_model=QueryResponse)
def trigger_agent(request: QueryRequest):
    logging.info(f"[API] Received query: {request.query}")
    try:
        answer = rag_agent(request.query)
        logging.info(f"[API] Returning answer: {answer}")
        return QueryResponse(answer=answer)
    except Exception as e:
        logging.error(f"[API] Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 