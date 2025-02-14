from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import julia
import julia.Main
import os

# Initialize Julia runtime
jl = julia.Julia()
julia_script_path = os.getenv("JULIA_SCRIPT_PATH", "../julia_core/memory_model.jl")
jl.include(julia_script_path)

# FastAPI app
app = FastAPI()

# Request models
class MemoryQuery(BaseModel):
    query_vector: List[float]
    top_k: int = 5

class MemoryStoreRequest(BaseModel):
    key: List[float]
    value: List[float]

@app.post("/retrieve_memory")
def retrieve_memory(query: MemoryQuery):
    try:
        result = jl.memory_retrieval(query.query_vector, query.top_k)
        return {"retrieved_memory": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/store_memory")
def store_memory(data: MemoryStoreRequest):
    try:
        jl.store_memory(data.key, data.value)
        return {"status": "Memory stored successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "API is running"}

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host=host, port=port)
