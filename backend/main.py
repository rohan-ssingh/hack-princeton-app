from typing import List

import json
import logging
import os
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from chat_query import create_agent_graph
from load import Storage
from schemas import ChatMessagePayload, UserQueryRequest, UserQueryResponse

load_dotenv()

LOG_LEVEL = os.getenv("AGENT_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("agent")

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
FAISS_PATH = str((BASE_DIR.parent / "faiss_index").resolve())
storage = Storage(path=FAISS_PATH, from_path=True)
app_graph = create_agent_graph(storage)
logger.info("Agent graph initialised; FAISS path=%s", FAISS_PATH)

with open(BASE_DIR / "prompts.json") as f:
    prompts = json.load(f)
system_prompt = prompts["user_query"]
logger.debug("Loaded system prompt (%d chars)", len(system_prompt))


def _convert_conversation(conversation: List[ChatMessagePayload]):
    history = []
    for message in conversation:
        content = (message.content or "").strip()
        if not content:
            continue
        logger.debug("Conversation turn -> role=%s, content_snippet=%s", message.role, content[:80])

        if message.role == "assistant":
            history.append(AIMessage(content=content))
        elif message.role == "system":
            history.append(SystemMessage(content=content))
        else:
            history.append(HumanMessage(content=content))
    return history

@app.post("/user-query", response_model=UserQueryResponse)
async def user_query_endpoint(user_request: UserQueryRequest):
    history = [SystemMessage(content=system_prompt)]
    existing_messages = _convert_conversation(user_request.conversation)
    logger.debug("Received %d prior turns", len(existing_messages))
    history.extend(existing_messages)
    history.append(HumanMessage(content=user_request.user_query.strip()))

    logger.info("Invoking agent graph with %d total messages", len(history))
    response = app_graph.invoke({"messages": history, "documents": []})
    final_response = response.get("final_response")

    if isinstance(final_response, UserQueryResponse):
        logger.info(
            "Agent produced response with %d supporting documents",
            len(final_response.documents),
        )
        return final_response

    validated = UserQueryResponse.model_validate(final_response)
    logger.info(
        "Agent response validated with %d supporting documents",
        len(validated.documents),
    )
    return validated


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
