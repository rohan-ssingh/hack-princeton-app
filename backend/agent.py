from typing import TypedDict, Annotated, Any, List
import operator
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from datetime import datetime

from load import Storage
from llm import llm
from schemas import UserQueryResponse
from langchain_core.documents import Document

@tool
def get_current_datetime() -> str:
    """
    Returns the current date and time in ISO format.
    """
    return datetime.now().isoformat()

class AgentState(TypedDict):
    messages: Annotated[list[Any], operator.add]
    final_response: UserQueryResponse | None

def create_agent_graph(storage: Storage):
    """
    Creates and compiles the langgraph agent.
    """
    tools = [storage.rag, get_current_datetime]
    tool_node = ToolNode(tools)
    model = llm.bind_tools(tools)

    def should_continue(state):
        messages = state['messages']
        last_message = messages[-1]
        if not last_message.tool_calls:
            return "end"
        else:
            return "continue"

    def call_model(state):
        messages = state['messages']
        response = model.invoke(messages)
        return {"messages": [response]}

    def build_final_response(state) -> dict:
        documents = []
        for message in state['messages']:
            if isinstance(message, ToolMessage):
                content = message.content
                if isinstance(content, dict) and 'documents' in content:
                    # content['documents'] is a list of Document objects
                    documents.extend(content['documents'])
        
        final_response_text = state['messages'][-1].content

        final_response = UserQueryResponse(
            text_response=final_response_text,
            documents=documents
        )
        return {"final_response": final_response}

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    workflow.add_node("action", tool_node)
    workflow.add_node("builder", build_final_response)

    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "action",
            "end": "builder"
        }
    )
    workflow.add_edge('action', 'agent')
    workflow.add_edge('builder', END)
    return workflow.compile()