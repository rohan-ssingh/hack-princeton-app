import datetime
import logging
import operator
from typing import Annotated, Any, List, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from load import Storage, make_rag_tool
from llm import llm
from schemas import DocumentPayload, UserQueryResponse

logger = logging.getLogger(__name__)


@tool
def get_current_datetime() -> str:
    """
    Returns the current date and time in ISO format.
    """
    return datetime.datetime.now().isoformat()


class AgentState(TypedDict):
    messages: Annotated[List[Any], operator.add]
    final_response: UserQueryResponse | None
    documents: Annotated[List[Document], operator.add]


def _deserialize_tool_output(raw_content: Any) -> Any:
    """Best-effort conversion of tool output into Python objects."""
    if isinstance(raw_content, str):
        logger.debug(
            "Attempting to deserialize tool output string (len=%d)",
            len(raw_content),
        )
        # Skip obvious error strings before attempting eval
        if raw_content.startswith("Error invoking tool"):
            logger.error("Tool returned error text: %s", raw_content)
            return {}

        try:
            return eval(  # noqa: S307 – trusted internal string
                raw_content,
                {
                    "Document": Document,
                    "datetime": datetime,
                    "date": datetime.date,
                    "Retrieval": dict,
                },
            )
        except Exception:  # pragma: no cover - never fail the whole request
            logger.warning("Failed to deserialize tool output", exc_info=True)
            return {}
    return raw_content


def _collect_documents(candidate: Any) -> List[Document]:
    documents: List[Document] = []

    if not candidate:
        return documents

    if isinstance(candidate, dict):
        candidate = candidate.get("documents", [])

    if not isinstance(candidate, list):
        return documents

    for item in candidate:
        if isinstance(item, Document):
            logger.debug(
                "Collected Document id=%s metadata_keys=%s",
                getattr(item, "id", None),
                list(getattr(item, "metadata", {}).keys()),
            )
            documents.append(item)
        elif isinstance(item, dict):
            try:
                metadata = item.get("metadata", {}) or {}
                doc = Document(
                    page_content=item.get("page_content", ""),
                    metadata=metadata,
                    id=item.get("id"),
                )
                documents.append(doc)
                logger.debug(
                    "Constructed Document from dict id=%s metadata_keys=%s",
                    item.get("id"),
                    list(metadata.keys()),
                )
            except Exception:
                logger.exception("Failed to convert retrieved item into Document")
                continue
    return documents


def create_agent_graph(storage: Storage):
    """
    Creates and compiles the langgraph agent.
    """

    rag_tool = make_rag_tool(storage)
    tools = [rag_tool, get_current_datetime]
    tool_node = ToolNode(tools)
    model = llm.bind_tools(tools)

    def should_continue(state: AgentState):
        messages = state["messages"]
        last_message = messages[-1]
        logger.debug(
            "Evaluating continuation: last message type=%s has_tool_calls=%s",
            type(last_message).__name__,
            bool(getattr(last_message, "tool_calls", None)),
        )
        if isinstance(last_message, BaseMessage) and not getattr(
            last_message, "tool_calls", None
        ):
            return "end"
        if not getattr(last_message, "tool_calls", None):
            return "end"
        return "continue"

    def call_model(state: AgentState):
        messages = state["messages"]
        logger.debug("Calling model with %d messages", len(messages))
        response = model.invoke(messages)
        logger.debug(
            "Model responded with type=%s has_tool_calls=%s",
            type(response).__name__,
            bool(getattr(response, "tool_calls", None)),
        )
        return {"messages": [response]}

    def call_tools(state: AgentState):
        tool_messages = tool_node.invoke(state["messages"])

        documents: List[Document] = []
        for message in tool_messages:
            if isinstance(message, ToolMessage) and message.name == "rag":
                logger.info("Processing output from tool=%s", message.name)
                retrieval_output = _deserialize_tool_output(message.content)
                documents.extend(_collect_documents(retrieval_output))
        logger.info("Tool invocation produced %d documents", len(documents))
        return {"messages": tool_messages, "documents": documents}

    def build_final_response(state: AgentState) -> dict:
        documents = state.get("documents", []) or []
        raw_content = state["messages"][-1].content
        logger.debug(
            "Building final response from message content type=%s",
            type(raw_content).__name__,
        )

        if (
            isinstance(raw_content, list)
            and raw_content
            and isinstance(raw_content[0], dict)
            and "text" in raw_content[0]
        ):
            final_response_text = raw_content[0]["text"]
        elif isinstance(raw_content, str):
            final_response_text = raw_content
        else:
            final_response_text = str(raw_content)

        payloads: List[DocumentPayload] = []
        seen_keys: set[str] = set()

        for document in documents:
            try:
                if isinstance(document, Document):
                    payload = DocumentPayload.from_document(document)
                elif isinstance(document, DocumentPayload):
                    payload = document
                elif isinstance(document, dict):
                    raw_metadata = document.get("metadata", {})
                    if isinstance(raw_metadata, dict):
                        safe_metadata = {
                            str(key): DocumentPayload._to_json_safe(value)
                            for key, value in raw_metadata.items()
                        }
                    else:
                        safe_metadata = {}

                    payload = DocumentPayload(
                        id=str(document.get("id")) if document.get("id") else None,
                        page_content=str(document.get("page_content", "")),
                        metadata=safe_metadata,
                    )
                else:
                    continue
            except Exception:
                continue

            dedupe_key = payload.metadata.get("url") or payload.id
            if dedupe_key:
                dedupe_key = str(dedupe_key)
                if dedupe_key in seen_keys:
                    continue
                seen_keys.add(dedupe_key)

            payloads.append(payload)
            logger.debug("Prepared payload with url=%s", payload.metadata.get("url"))

        final_response = UserQueryResponse(
            text_response=final_response_text,
            documents=payloads,
        )
        logger.info(
            "Final response ready: text_len=%d documents=%d",
            len(final_response_text),
            len(payloads),
        )
        return {"final_response": final_response}

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    workflow.add_node("action", call_tools)
    workflow.add_node("builder", build_final_response)

    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "action",
            "end": "builder",
        },
    )
    workflow.add_edge("action", "agent")
    workflow.add_edge("builder", END)
    return workflow.compile()
