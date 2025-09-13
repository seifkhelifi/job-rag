from fastapi import FastAPI, Query
from pydantic import BaseModel
from rag import JobRAGPipeline
from utils.load_config import load_config
import uvicorn

# Load config
config = load_config("../config.yml")

# Initialize pipeline (global)
pipeline = JobRAGPipeline(
    csv_path=config["data"]["csv_path"],
    index_name=config["vector_store"]["index_name"],
    persistence_path=config["vector_store"]["persistence_path"],
    chunk_size=config["chunking"]["chunk_size"],
    chunk_overlap=config["chunking"]["chunk_overlap"],
    chunk_strategy=config["chunking"]["strategy"],
    delete_if_exists=False,
    # Ranking parameters
    top_n=config["ranking"]["top_n"],
    ranking_model=config["ranking"]["model"],
    # Retrieval parameters
    similarity_top_k=config["retrieval"]["similarity_top_k"],
    vector_store_query_mode=config["retrieval"]["vector_store_query_mode"],
    alpha=config["retrieval"]["alpha"],
    # Model parameters
    embedding_model=config["models"]["embedding"],
    llm_model=config["models"]["llm"],
    # Additional parameters
    tech_specialties=config["tech_specialties"],
    locations=config["locations"],
    # Server parameters
    server_hostname=config["server"]["hostname"],
    server_port=config["server"]["port"],
)

# Setup pipeline once
pipeline.setup()

# FastAPI app
app = FastAPI(
    title="Job RAG API", description="Job search using RAG pipeline", version="1.0"
)


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    response: str


@app.post("/query", response_model=QueryResponse)
def query_job(request: QueryRequest):
    """Submit a job query."""
    result = pipeline.query(request.query)
    return {"response": result["answer"]}


@app.on_event("shutdown")
def shutdown_event():
    """Clean up pipeline resources when API shuts down."""
    pipeline.close()


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=config["api"]["hostname"],
        port=config["api"]["port"],
        reload=True,
    )
