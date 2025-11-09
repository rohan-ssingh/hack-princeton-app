import asyncio
from fastapi import FastAPI
from dotenv import load_dotenv
import uvicorn
import json
from langchain_core.messages import SystemMessage, HumanMessage

from schemas import UserQueryRequest, UserQueryResponse
from agent import create_agent_graph
from load import Storage

load_dotenv()

app = FastAPI()

# --- App Initialization ---
FAISS_PATH = "faiss_index"
storage = Storage(path=FAISS_PATH, from_path=True)
app_graph = create_agent_graph(storage)

with open('backend/prompts.json') as f:
    prompts = json.load(f)
system_prompt = prompts['user_query']

# --- API Endpoint ---
@app.post("/user-query", response_model=UserQueryResponse)
async def user_query_endpoint(user_request: UserQueryRequest):
    """
    Handles user queries.
    """
    
    inputs = {"messages": [SystemMessage(content=system_prompt), HumanMessage(content=user_request.user_query)]}
    response = app_graph.invoke(inputs)
    
    return response['final_response']

if __name__ == "__main__":
    # To run this, you need to install uvicorn: pip install uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)