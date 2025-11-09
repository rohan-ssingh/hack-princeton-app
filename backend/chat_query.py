from typing import TypedDict, Annotated, Any, List
import operator
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
import datetime

from load import Storage
from llm import llm
from schemas import UserQueryResponse
from langchain_core.documents import Document

@tool
def get_current_datetime() -> str:
    """
    Returns the current date and time in ISO format.
    """
    return datetime.datetime.now().isoformat()

class AgentState(TypedDict):
    messages: Annotated[list[Any], operator.add]
    final_response: UserQueryResponse | None
    documents: Annotated[list[Document], operator.add]

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

    def call_tools(state):
        tool_messages = tool_node.invoke(state['messages'])

        documents = []
        for message in tool_messages:
            print(message)
            if isinstance(message, ToolMessage) and message.name == "rag":
                print('found message, type' , type(message.content))
                # The content of the ToolMessage from the rag tool is the Retrieval object
                retrieval_output = message.content
                if isinstance(retrieval_output, str):
                    try:
                        # The string may contain Document and datetime objects, so we use eval.
                        # This is generally unsafe, but here we trust the source.
                        retrieval_output = eval(retrieval_output, {"Document": Document, "datetime": datetime})
                    except Exception:
                        # If parsing fails, we'll just have an empty dict.
                        retrieval_output = {}
                
                if isinstance(retrieval_output, dict) and 'documents' in retrieval_output:
                    documents.extend(retrieval_output['documents'])
        
        return {"messages": tool_messages, "documents": documents}

    def build_final_response(state) -> dict:
        documents = state.get('documents', [])
        
        raw_content = state['messages'][-1].content
        if isinstance(raw_content, list) and raw_content and isinstance(raw_content[0], dict) and 'text' in raw_content[0]:
            final_response_text = raw_content[0]['text']
        elif isinstance(raw_content, str):
            final_response_text = raw_content
        else:
            final_response_text = str(raw_content)

        final_response = UserQueryResponse(
            text_response=final_response_text,
            documents=documents
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
            "end": "builder"
        }
    )
    workflow.add_edge('action', 'agent')
    workflow.add_edge('builder', END)
    return workflow.compile()