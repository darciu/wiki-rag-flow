from typing import TypedDict, Annotated, List
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage
import logging
import random
import math
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
from llm.routing import create_plan, direct_query
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

@tool
def power(base: float, exponent: float) -> float:
    """Raises the base to the power of the exponent: base ** exponent."""
    return math.pow(base, exponent)

@tool
def square_root(a: float) -> float:
    """Calculates the square root of a single number. The number must be non-negative."""
    if a < 0:
        raise ValueError("Cannot calculate the square root of a negative number.")
    return math.sqrt(a)

@tool
def absolute_value(a: float) -> float:
    """Calculates the absolute value of a single number."""
    return abs(a)

math_tools = [add, subtract, multiply, divide, power, square_root, absolute_value]

def router_node(state: AgentState, config: RunnableConfig) -> dict:
    last_message = state["messages"][-1].content

    instructor_client = config.get("configurable", {}).get("instructor_client")
    if not instructor_client:
        raise ValueError("Could not find instructor_client")

    decision = create_plan(instructor_client, last_message, "llama3.2")

    return {
        "current_query": last_message,
        "route": decision.route_type,
        "clarify_message": decision.clarify_message,
    }


def rag_search_node(state: AgentState) -> dict:
    return {}


def direct_node(state: AgentState, config: RunnableConfig) -> dict:

    instructor_client = config.get("configurable", {}).get("instructor_client")
    if not instructor_client:
        raise ValueError("Could not find instructor_client")
    
    current_query = state["current_query"]
    
    decision = direct_query(instructor_client, current_query, "llama3.2")
    if decision.knows_answer == True:

        return {"messages": [AIMessage(content=decision.answer)]}
    else:
        answers = ["Przykro mi, lecz nie dysponuję informacjami pozwalającymi odpowiedzieć na to pytanie.",
                   "Chciałbym pomóc, ale to zagadnienie wykracza poza zakres mojej wiedzy.",
                   "Niestety, nie znam odpowiedzi na ten temat.",
                   "Nie posiadam danych na ten temat, ale mogę spróbować pomóc w innej kwestii.",
                   "Tym razem nie będę w stanie udzielić wyjaśnień.",
                   "Niestety, nie jestem w stanie udzielić odpowiedzi na to pytanie.",]

        return {"messages": [AIMessage(content=random.choice(answers))]}


def clarify_node(state: AgentState) -> dict:
    return {"messages": [AIMessage(content=state["clarify_message"])]}


def math_node(state: AgentState, config: RunnableConfig) -> dict:
    current_query = state["current_query"]
    
    langchain_client = config.get("configurable", {}).get("langchain_client")
    if not langchain_client:
        raise ValueError("Could not find langchain client")

    math_langchain_client = langchain_client.bind_tools(math_tools)
    

    system_prompt = SystemMessage(
        content=MATH_SYSTEM_PROMPT
    )
    
    response = math_langchain_client.invoke([system_prompt, {"role": "user", "content": current_query}])
    
    if response.tool_calls:
        tool_call = response.tool_calls[0]
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        
        
        tools_map = {t.name: t for t in math_tools}
        selected_tool = tools_map[tool_name]
        
        answers = [
            "Wynik to: ",
            "Obliczona wartość wynosi: ",
            "Rezultat obliczeń: ",
            "Po przeliczeniu otrzymano wynik: ",
            "Końcowa wartość to: ",
            "Uzyskany rezultat wynosi: ",
            "System wygenerował wynik: ",
            "Wynik: "
        ]

        try:
            result = selected_tool.invoke(tool_args)
            final_content = random.choice(answers) + str(result)
        except Exception as e:
            final_content = f"Błąd podczas obliczeń matematycznych: {str(e)}"
    else:
        final_content = "Nie udało mi się wykonać działania matematycznego."

    return {"messages": [AIMessage(content=final_content)]}


def route_condition(state: AgentState) -> str:
    return state["route"]

def task_type_condition(state: AgentState) -> str:
    return state["task_type"]


def lookup_node(state: AgentState, config: RunnableConfig) -> dict:
    # standardowo szerokie zapytanie do bazy RAG z n-1, n+1
    return {
        "messages": [AIMessage(content="rag search")]
    }

def compare_node(state: AgentState, config: RunnableConfig) -> dict:
    # wyszukiwanie encji oraz ich wspólnego mianownika do porównania
    # następnie tyle wyszukań w bazie RAG ile udało znaleźć się encji (ale dosyć wąskich wyszukiwań bez n-1, n+1)
    return {
        "messages": [AIMessage(content="rag search")]
    }

def summarize_node(state: AgentState, config: RunnableConfig) -> dict:
    # wyszukuje szeroki kontekst dla zadanego tematu i zwraca streszczenie (tutaj przyda się n-1, n+1 chunk)
    return {
        "messages": [AIMessage(content="rag search")]
    }



# GRAPH
graph = StateGraph(AgentState)

graph.add_node("router", router_node)
graph.add_node("rag_search", rag_search_node)
graph.add_node("direct", direct_node)
graph.add_node("clarify", clarify_node)
graph.add_node("math", math_node)
graph.add_node("lookup", lookup_node)
graph.add_node("compare", compare_node)
graph.add_node("summarize", summarize_node)


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

graph.add_conditional_edges(
    "rag_search",
    task_type_condition,
    {
        TaskType.LOOKUP: "lookup",
        TaskType.COMPARE: "compare",
        TaskType.SUMMARIZE: "summarize",
    }
)

graph.add_edge("direct", END)
graph.add_edge("clarify", END)
graph.add_edge("math", END)
graph.add_edge("lookup", END)
graph.add_edge("compare", END)
graph.add_edge("summarize", END)


agent = graph.compile()
