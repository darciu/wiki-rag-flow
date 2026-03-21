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
from langgraph.checkpoint.memory import MemorySaver
from collections import defaultdict
from langchain_core.tools import tool
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import AIMessage
from llm.routing import (
    create_plan,
    direct_query,
    process_query,
    lookup_query,
    summarize_query,
    precompare_query,
    compare_query,
)
from llm.routing import RouteType, TaskType
from llm.prompts import MATH_SYSTEM_PROMPT

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    current_query: str
    answer: str
    further_questions: str | None
    route: RouteType
    task_type: TaskType
    clarify_message: str | None


### UTILS ###


def unique_chunks(results: list[dict]) -> list[dict]:
    """Filter unique chunks of given wiki article with the highest rank_score"""
    unique_map: dict = {}

    for item in results:
        key = (item["source_id"], item["chunk_id"])

        current_score = item.get("rank_score", -float("inf"))
        if key not in unique_map or current_score > unique_map[key].get(
            "rank_score", -float("inf")
        ):
            unique_map[key] = item
    return sorted(unique_map.values(), key=lambda x: x["rank_score"], reverse=True)


def prepare_context_for_llm(sorted_chunks: list[dict], question) -> str:
    """Query for LLM containing found wiki chunks and also primary question"""
    if not sorted_chunks:
        return "Brak dostępnego kontekstu."

    blocks = []
    current_block = {
        "source_id": sorted_chunks[0]["source_id"],
        "source_title": sorted_chunks[0]["source_title"],
        "last_chunk_id": sorted_chunks[0]["chunk_id"],
        "text": sorted_chunks[0]["chunk_text"],
    }

    for i in range(1, len(sorted_chunks)):
        next_chunk = sorted_chunks[i]

        is_same_doc = next_chunk["source_id"] == current_block["source_id"]
        is_sequential = False
        if (
            next_chunk.get("chunk_id") is not None
            and current_block.get("last_chunk_id") is not None
        ):
            is_sequential = next_chunk["chunk_id"] == current_block["last_chunk_id"] + 1

        if is_same_doc and is_sequential:
            current_block["text"] += " " + next_chunk["chunk_text"]
            current_block["last_chunk_id"] = next_chunk["chunk_id"]
        else:
            blocks.append(current_block)
            current_block = {
                "source_id": next_chunk["source_id"],
                "source_title": next_chunk["source_title"],
                "last_chunk_id": next_chunk["chunk_id"],
                "text": next_chunk["chunk_text"],
            }

    blocks.append(current_block)

    xml_output = "<context>\n"
    for block in blocks:
        xml_output += (
            f'  <document id="{block["source_id"]}" title="{block["source_title"]}">\n'
            f"    {block['text'].strip()}\n"
            f"  </document>\n"
        )
    xml_output += "</context>"

    xml_output += f"\n\n<question>{question}</question>"

    return xml_output


def prepare_comparison_context_for_llm(
    sorted_chunks: list[dict],
    question: str,
    entities: list[str],
    comparison_aspects: list[str],
) -> str:

    blocks = []
    current_block = {
        "source_id": sorted_chunks[0]["source_id"],
        "source_title": sorted_chunks[0]["source_title"],
        "last_chunk_id": sorted_chunks[0]["chunk_id"],
        "text": sorted_chunks[0]["chunk_text"],
    }

    for i in range(1, len(sorted_chunks)):
        next_chunk = sorted_chunks[i]

        is_same_doc = next_chunk["source_id"] == current_block["source_id"]
        is_sequential = False
        if (
            next_chunk.get("chunk_id") is not None
            and current_block.get("last_chunk_id") is not None
        ):
            is_sequential = next_chunk["chunk_id"] == current_block["last_chunk_id"] + 1

        if is_same_doc and is_sequential:
            current_block["text"] += " " + next_chunk["chunk_text"]
            current_block["last_chunk_id"] = next_chunk["chunk_id"]
        else:
            blocks.append(current_block)
            current_block = {
                "source_id": next_chunk["source_id"],
                "source_title": next_chunk["source_title"],
                "last_chunk_id": next_chunk["chunk_id"],
                "text": next_chunk["chunk_text"],
            }

    blocks.append(current_block)

    xml_output = "<context>\n"
    for block in blocks:
        xml_output += (
            f'  <document id="{block["source_id"]}" title="{block["source_title"]}">\n'
            f"    {block['text'].strip()}\n"
            f"  </document>\n"
        )
    xml_output += "</context>\n\n"

    xml_output += "<comparison_meta>\n"

    if entities:
        xml_output += "  <entities>\n"
        for entity in entities:
            xml_output += f"    <entity>{entity}</entity>\n"
        xml_output += "  </entities>\n"

    if comparison_aspects:
        xml_output += "  <aspects>\n"
        for aspect in comparison_aspects:
            xml_output += f"    <aspect>{aspect}</aspect>\n"
        xml_output += "  </aspects>\n"

    xml_output += "</comparison_meta>\n\n"

    xml_output += f"<question>{question}</question>"

    return xml_output


def get_neighbour_context_keys(results: list[dict]) -> list[tuple[str, int]]:
    """For given list of wiki article chunks, get also N-1 and N+1 chunks that do not overlap with existing ones"""
    existing_keys = {(res["source_id"], res["chunk_id"]) for res in results}
    missing_keys = set()

    for res in results:
        s_id = res["source_id"]
        c_id = res["chunk_id"]

        if c_id - 1 >= 0:
            prev_key = (s_id, c_id - 1)
            if prev_key not in existing_keys:
                missing_keys.add(prev_key)

        next_key = (s_id, c_id + 1)
        if next_key not in existing_keys:
            missing_keys.add(next_key)

    return list(missing_keys)


### TOOLS ###


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

### NODES ###


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
        "task_type": decision.task_type,
    }


def direct_node(state: AgentState, config: RunnableConfig) -> dict:
    instructor_client = config.get("configurable", {}).get("instructor_client")
    if not instructor_client:
        raise ValueError("Could not find instructor_client")

    current_query = state["current_query"]

    decision = direct_query(instructor_client, current_query, "llama3.2")
    if decision.knows_answer == True:
        return {"messages": [AIMessage(content=decision.answer)]}
    else:
        answers = [
            "Przykro mi, lecz nie dysponuję informacjami pozwalającymi odpowiedzieć na to pytanie.",
            "Chciałbym pomóc, ale to zagadnienie wykracza poza zakres mojej wiedzy.",
            "Niestety, nie znam odpowiedzi na ten temat.",
            "Nie posiadam danych na ten temat, ale mogę spróbować pomóc w innej kwestii.",
            "Tym razem nie będę w stanie udzielić wyjaśnień.",
            "Niestety, nie jestem w stanie udzielić odpowiedzi na to pytanie.",
        ]

        return {"messages": [AIMessage(content=random.choice(answers))]}


def clarify_node(state: AgentState) -> dict:
    return {"messages": [AIMessage(content=state["clarify_message"])]}


def math_node(state: AgentState, config: RunnableConfig) -> dict:
    current_query = state["current_query"]

    langchain_client = config.get("configurable", {}).get("langchain_client")
    if not langchain_client:
        raise ValueError("Could not find langchain client")

    math_langchain_client = langchain_client.bind_tools(math_tools)

    system_prompt = SystemMessage(content=MATH_SYSTEM_PROMPT)

    response = math_langchain_client.invoke(
        [system_prompt, HumanMessage(content=current_query)]
    )

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
            "Wynik: ",
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
    route = state.get("route")

    if route == RouteType.RAG_SEARCH:
        task = state.get("task_type")
        if task == TaskType.LOOKUP:
            return "lookup"
        elif task == TaskType.COMPARE:
            return "compare"
        elif task == TaskType.SUMMARIZE:
            return "summarize"
        else:
            return "lookup"

    elif route == RouteType.DIRECT:
        return "direct"
    elif route == RouteType.CLARIFY:
        return "clarify"
    elif route == RouteType.MATH:
        return "math"

    return "direct"


def lookup_node(state: AgentState, config: RunnableConfig) -> dict:
    weaviate_client = config.get("configurable", {}).get("weaviate_client")
    if not weaviate_client:
        raise ValueError("Could not find weaviate client")

    instructor_client = config.get("configurable", {}).get("instructor_client")
    if not instructor_client:
        raise ValueError("Could not find instructor_client")

    nlp_toolkit = config.get("configurable", {}).get("nlp_toolkit")
    if not nlp_toolkit:
        raise ValueError("Could not find nlp_toolkit")

    current_query = state["current_query"]
    decision = process_query(instructor_client, current_query, "llama3.2")

    all_queries = [current_query] + decision.queries

    basic_chunks = []
    for query_text in all_queries:
        query_results = weaviate_client.single_wikichunk_hybrid_fetch(
            query_text, 8, 0.5
        )
        scores = nlp_toolkit.rank(
            query_text, [elem["chunk_text"] for elem in query_results]
        )
        for score, elem in zip(scores, query_results, strict=True):
            elem["rank_score"] = score
        basic_chunks.extend(query_results)
    basic_chunks = unique_chunks(basic_chunks)

    basic_chunks = basic_chunks[:10]

    sorted_chunks = sorted(basic_chunks, key=lambda x: (x["source_id"], x["chunk_id"]))

    context_for_llm = prepare_context_for_llm(sorted_chunks, current_query)

    lookup_decision = lookup_query(instructor_client, context_for_llm, "llama3.2")

    return {
        "messages": [AIMessage(content=lookup_decision.answer)],
        "answer": lookup_decision.answer,
        "further_questions": lookup_decision.further_questions,
    }


def compare_node(state: AgentState, config: RunnableConfig) -> dict:

    weaviate_client = config.get("configurable", {}).get("weaviate_client")
    if not weaviate_client:
        raise ValueError("Could not find weaviate client")

    instructor_client = config.get("configurable", {}).get("instructor_client")
    if not instructor_client:
        raise ValueError("Could not find instructor_client")

    nlp_toolkit = config.get("configurable", {}).get("nlp_toolkit")
    if not nlp_toolkit:
        raise ValueError("Could not find nlp_toolkit")

    current_query = state["current_query"]
    decision = precompare_query(instructor_client, current_query, "llama3.2")

    if not decision:
        return {
            "answer": "Nie udało mi się znaleźć odpowiedzi na zadaną kwestię.",
            "further_questions": [],
        }

    logger.info(decision)

    all_chunks = []
    for query_text in decision.search_queries:
        query_results = weaviate_client.single_wikichunk_hybrid_fetch(
            query_text, 8, 0.5
        )
        scores = nlp_toolkit.rank(
            query_text, [elem["chunk_text"] for elem in query_results]
        )
        for score, elem in zip(scores, query_results, strict=True):
            elem["rank_score"] = score

        all_chunks.extend(query_results[:3])

    sorted_chunks = sorted(all_chunks, key=lambda x: (x["source_id"], x["chunk_id"]))

    context_for_llm = prepare_comparison_context_for_llm(
        sorted_chunks, current_query, decision.entities, decision.comparison_aspects
    )

    compare_decision = compare_query(instructor_client, context_for_llm, "llama3.2")

    return {
        "messages": [AIMessage(content=compare_decision.comparison)],
        "answer": compare_decision.comparison,
        "further_questions": compare_decision.further_questions,
    }


def summarize_node(state: AgentState, config: RunnableConfig) -> dict:

    weaviate_client = config.get("configurable", {}).get("weaviate_client")
    if not weaviate_client:
        raise ValueError("Could not find weaviate client")

    instructor_client = config.get("configurable", {}).get("instructor_client")
    if not instructor_client:
        raise ValueError("Could not find instructor_client")

    nlp_toolkit = config.get("configurable", {}).get("nlp_toolkit")
    if not nlp_toolkit:
        raise ValueError("Could not find nlp_toolkit")

    current_query = state["current_query"]
    decision = process_query(instructor_client, current_query, "llama3.2")

    all_queries = [current_query] + decision.queries

    basic_chunks = []
    for query_text in all_queries:
        query_results = weaviate_client.single_wikichunk_hybrid_fetch(
            query_text, 8, 0.5
        )
        scores = nlp_toolkit.rank(
            query_text, [elem["chunk_text"] for elem in query_results]
        )
        for score, elem in zip(scores, query_results, strict=True):
            elem["rank_score"] = score
        basic_chunks.extend(query_results)
    basic_chunks = unique_chunks(basic_chunks)

    basic_chunks = basic_chunks[:4]

    missing_keys = get_neighbour_context_keys(basic_chunks)
    grouped_source_chunk_id = defaultdict(list)
    for s_id, c_id in missing_keys:
        grouped_source_chunk_id[s_id].append(c_id)

    extended_chunks = weaviate_client.batch_wikichunk_fetch(grouped_source_chunk_id)

    sorted_chunks = sorted(
        extended_chunks, key=lambda x: (x["source_id"], x["chunk_id"])
    )

    context_for_llm = prepare_context_for_llm(sorted_chunks, current_query)

    summarize_decision = summarize_query(instructor_client, context_for_llm, "llama3.2")

    return {
        "messages": [AIMessage(content=summarize_decision.summary)],
        "answer": summarize_decision.summary,
        "further_questions": summarize_decision.further_questions,
    }


# GRAPH
graph = StateGraph(AgentState)

graph.add_node("router", router_node)
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
        "lookup": "lookup",
        "compare": "compare",
        "summarize": "summarize",
        "direct": "direct",
        "clarify": "clarify",
        "math": "math",
    },
)


graph.add_edge("direct", END)
graph.add_edge("clarify", END)
graph.add_edge("math", END)
graph.add_edge("lookup", END)
graph.add_edge("compare", END)
graph.add_edge("summarize", END)

memory = MemorySaver()

agent = graph.compile(checkpointer=memory)
