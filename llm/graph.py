from typing import TypedDict, Annotated, List
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage
import logging

import instructor
from instructor.core.client import Instructor
from instructor.exceptions import InstructorRetryException
from pydantic import BaseModel, Field, model_validator
from openai import OpenAI
from langchain_core.messages import HumanMessage
from langchain_core.runnables.config import RunnableConfig
from typing import TypedDict, Dict, List, Union, Annotated, Sequence
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    BaseMessage,
    ToolMessage,
    SystemMessage,
)
from langchain_core.tools import tool
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import AIMessage
from llm.routing import create_plan
from llm.routing import RouteType, TaskType
from llm.prompts import MATH_SYSTEM_PROMPT
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    current_query: str
    route: RouteType
    task_type: TaskType
    clarify_message: str | None

@tool
def add(a: float, b: float) -> float:
    """Addition of two numbers: a + b."""
    return a + b

@tool
def subtract(a: float, b: float) -> float:
    """Subtraction of two numbers: a - b."""
    return a - b

@tool
def multiply(a: float, b: float) -> float:
    """Multiplication of two numbers: a * b."""
    return a * b

@tool
def divide(a: float, b: float) -> float:
    """Division of two numbers: a / b. Raises a ValueError if b is zero."""
    if b == 0:
        raise ValueError("Cannot divide by zero!")
    return a / b

math_tools = [add, subtract, multiply, divide]

def router_node(state: AgentState, config: RunnableConfig) -> dict:
    last_message = state["messages"][-1].content

    instructor_client = config.get("configurable", {}).get("instructor_client")
    if not instructor_client:
        raise ValueError("Could not find instructor_client")

    decision = create_plan(instructor_client, last_message, "llama3.2")

    return {
        "user_query": last_message,
        "route": decision.route_type,
        "clarify_message": decision.clarify_message,
    }


def rag_search_node(state: AgentState) -> dict:
    return {
        "messages": [AIMessage(content="rag search")]
    }


def direct_node(state: AgentState) -> dict:
    return {
        "messages": [
            AIMessage(content="direct answer")
        ]
    }


def clarify_node(state: AgentState) -> dict:
    return {"messages": [AIMessage(content=state["clarify_message"])]}


def math_node(state: AgentState, config: RunnableConfig) -> dict:
    last_message = state["messages"][-1].content
    
    langchain_client = config.get("configurable", {}).get("langchain_client")
    if not langchain_client:
        raise ValueError("Krytyczny błąd: Nie wstrzyknięto chat_model do obsługi narzędzi!")

    math_langchain_client = langchain_client.bind_tools(math_tools)
    

    system_prompt = SystemMessage(
        content=MATH_SYSTEM_PROMPT
    )
    
    response = math_langchain_client.invoke([system_prompt, {"role": "user", "content": last_message}])
    
    if response.tool_calls:
        tool_call = response.tool_calls[0]
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        
        
        tools_map = {t.name: t for t in math_tools}
        selected_tool = tools_map[tool_name]
        
        try:
            result = selected_tool.invoke(tool_args)
            final_content = f"Wykonano operację: {tool_name}. Wynik: {result}."
        except Exception as e:
            final_content = f"Błąd podczas obliczeń matematycznych: {str(e)}"
    else:
        final_content = "Nie udało mi się wykonać działania matematycznego."

    return {"messages": [AIMessage(content=final_content)]}


def route_condition(state: AgentState) -> str:
    return state["route"]





# GRAPH
graph = StateGraph(AgentState)

graph.add_node("router", router_node)
graph.add_node("rag_search", rag_search_node)
graph.add_node("direct", direct_node)
graph.add_node("clarify", clarify_node)
graph.add_node("math", math_node)

graph.set_entry_point("router")

graph.add_conditional_edges(
    "router",
    route_condition,
    {
        RouteType.RAG_SEARCH: "rag_search",
        RouteType.DIRECT: "direct",
        RouteType.CLARIFY: "clarify",
        RouteType.MATH: "math",
    },
)

graph.add_edge("rag_search", END)
graph.add_edge("direct", END)
graph.add_edge("clarify", END)
graph.add_edge("math", END)


agent = graph.compile()
