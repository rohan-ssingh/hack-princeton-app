import asyncio
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from dedalus_labs import AsyncDedalus, DedalusRunner
from dotenv import load_dotenv
import uvicorn
import json
from datetime import date, datetime

from langchain_core.documents import Document
from load import Storage

load_dotenv()

app = FastAPI()

# --- Pydantic Models ---
class UserQueryRequest(BaseModel):
    user_query: str

class UserQueryResponse(BaseModel):
    text_response: str
    documents: List[Document]

# --- Storage Initialization ---
FAISS_PATH = "faiss_index"
storage = Storage(path=FAISS_PATH, from_path=True)

with open('backend/prompts.json') as f:
    prompts = json.load(f)

# --- Tools ---
def get_current_datetime() -> str:
    """
    Returns the current date and time in ISO format.
    """
    return datetime.now().isoformat()

# --- API Endpoint ---
@app.post("/user-query", response_model=UserQueryResponse)
async def user_query_endpoint(user_request: UserQueryRequest):
    """
    Handles user queries.
    """
    
    client = AsyncDedalus()
    runner = DedalusRunner(client)

    return await runner.run(
        input=prompts['user_query'].format(user_request.user_query),
        model="google/gemini-2.5-pro",
        tools=[storage.rag, get_current_datetime],
        verbose=True
    )

if __name__ == "__main__":
    # To run this, you need to install uvicorn: pip install uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)