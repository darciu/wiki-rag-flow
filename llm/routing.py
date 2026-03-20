from typing import TypedDict, Annotated, List
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage
import logging
from enum import StrEnum
from pydantic import BaseModel, Field, model_validator
from instructor.core.client import Instructor
from instructor.exceptions import InstructorRetryException
from llm.prompts import PLANNER_SYSTEM_PROMPT, DIRECT_ANSWER_SYSTEM_PROMPT, PROCESS_SYSTEM_PROMPT, LOOKUP_SYSTEM_PROMPT, SUMMARIZE_SYSTEM_PROMPT, PRECOMPARE_SYSTEM_PROMPT, COMPARE_SYSTEM_PROMPT
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
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_ollama import ChatOllama
from instructor.core.client import Instructor
from instructor.exceptions import InstructorRetryException
from backend.app.schemas import RouteType, TaskType

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class QueryPlanner(BaseModel):
    route_type: RouteType = Field(
        ..., description="Choice of the main route: clarify, direct, math or rag_search."
    )
    task_type: TaskType | None = Field(
        default=None,
        description="Task type: lookup, compare, summarize. Only if route=rag_search."
    )
    clarify_message: str | None = Field(
        default=None,
        description="Clarification question for the user, only if route=clarify."
    )


    @model_validator(mode="after")
    def validate_consistency(self):
        if self.route_type == RouteType.CLARIFY and not self.clarify_message:
            raise ValueError("clarify_message is required when route=clarify")
        
        if self.route_type == RouteType.RAG_SEARCH and not self.task_type:
            raise ValueError("task should not be empty when route=rag_search")

        if self.route_type != RouteType.CLARIFY:
            self.clarify_message = None


        return self

class DirectQuestion(BaseModel):
    answer: str = Field(
        ...,
        description="A substantive and concise answer to the user's question.",
    )
    knows_answer: bool = Field(
        ...,
        description="Does the LLM model have enough knowledge to answer this question? True if yes, False if it must admit it doesn't know.",
    )

    @model_validator(mode="after")
    def validate_content(self) -> "DirectQuestion":
        if not self.answer or len(self.answer.strip()) < 5:
            raise ValueError(
                "The answer field must contain meaningful content, even if you admit you cannot provide the answer."
            )
        return self

class QueryProcessing(BaseModel):
    queries: list[str] = Field(
        ..., description="List of 1-3 different paraphrases of the underlying query."
    )


class LookupQuery(BaseModel):
    answer: str = Field(
        description="Answer to the question from <context>."
    )
    further_questions: List[str] = Field(
        description="List of one or two questions generated from the given <context>, other than <question>."
    )


class SummarizeQuery(BaseModel):
    summary: str = Field(
        description="Summary of given text."
    )
    further_questions: List[str] = Field(
        description="List of one or two questions generated from the given <context>, other than <question>."
    )

def build_search_queries(entities, comparison_aspects):
    search_queries = []
    if comparison_aspects:
        aspects = " ".join(comparison_aspects)
    else:
        aspects = ""
    
    for entity in entities:
        search_queries.append(entity+ " " + aspects)
    search_queries.append(", ".join(entities) + " " + aspects)
    return search_queries

class PreQueryCompare(BaseModel):

    entities: list[str] = Field(
        min_length=2,
        description="List of entities appearing in user text. Return entities in their base form."
    )

    comparison_aspects: list[str] = Field(default_factory=list, description="A criterion for comparing entities listed by the user.")

    search_queries: list[str] = Field(default_factory=list)

    
    @model_validator(mode="after")
    def generate_search_queries(self):
        if not self.search_queries:
            self.search_queries = build_search_queries(
                entities=self.entities,
                comparison_aspects=self.comparison_aspects,
            )
        return self

class CompareQuery(BaseModel):
    comparison: str = Field(
        description="Compare <entities> using <context>. Focus on <aspects> if provided; otherwise, extract and compare main features."
    )
    further_questions: List[str] = Field(
        description="List of one or two questions generated from the given <context>, other than <question>."
    )

def create_plan(llm_client: Instructor, question: str, model_name: str) -> QueryPlanner:

    try:
        return llm_client.chat.completions.create(
            model=model_name,
            response_model=QueryPlanner,
            max_retries=5,
            temperature=0,
            messages=[
                {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
                {"role": "system", "content": "The knowledge database contains Polish Wikipedia articles divided into chunks."},
                {"role": "user", "content": question},
            ],
        )
    except InstructorRetryException as e:
        print(
            f"Warning! Model could not generate reply after {e.n_attempts} retires."
        )
    

def direct_query(
    client: Instructor, user_query: str, model_name: str
) -> DirectQuestion:

    try:
        return client.chat.completions.create(
            model=model_name,
            response_model=DirectQuestion,
            max_retries=3,
            messages=[
                {"role": "system", "content": DIRECT_ANSWER_SYSTEM_PROMPT},
                {"role": "user", "content": user_query},
            ],
        )
    except InstructorRetryException:
        return DirectQuestion(
            answer="Przepraszam, ale nie jestem w stanie odpowiedzieć na to pytanie.",
            knows_answer=False,
            confidence_score=0.0,
        )
    


def process_query(
    client: Instructor, user_query: str, model_name: str
) -> QueryProcessing:

    try:
        return client.chat.completions.create(
            model=model_name,
            response_model=QueryProcessing,
            temperature=0.0,
            max_retries=3,
            messages=[
                {"role": "system", "content": PROCESS_SYSTEM_PROMPT},
                {"role": "user", "content": user_query},
            ],
        )

    except InstructorRetryException as e:
        logger.info(
            f"Warning! Model could not generate reply after {e.n_attempts} retires."
        )
        return QueryProcessing(queries=[])
    

def lookup_query(client: Instructor, context: str, model_name: str) -> LookupQuery:
    try:
        system_prompt = LOOKUP_SYSTEM_PROMPT

        decision = client.chat.completions.create(
            model=model_name,
            response_model=LookupQuery,
            max_retries=3,
            temperature=0.0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": context},
            ],
        )
        return decision
    except InstructorRetryException as e:
        logger.info(
            f"Warning! Model could not generate reply after {e.n_attempts} retires."
        )

        return LookupQuery(
            answer="Nie udało mi się znaleźć odpowiedzi na zadaną kwestię.",
            further_questions = [],
        )
    

def summarize_query(client: Instructor, context: str, model_name: str) -> SummarizeQuery:
    try:
        system_prompt = SUMMARIZE_SYSTEM_PROMPT

        decision = client.chat.completions.create(
            model=model_name,
            response_model=SummarizeQuery,
            max_retries=3,
            temperature=0.0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": context},
            ],
        )
        return decision
    except InstructorRetryException as e:
        logger.info(
            f"Warning! Model could not generate reply after {e.n_attempts} retires."
        )

        return SummarizeQuery(
            summary="Nie udało mi się podsumować danej kwestii.",
            further_questions = [],
        )
    


def precompare_query(llm_client: Instructor, question: str, model_name: str) -> PreQueryCompare:

    try:
        return llm_client.chat.completions.create(
            model=model_name,
            response_model=PreQueryCompare,
            max_retries=5,
            temperature=0.1,
            messages=[
                {"role": "system", "content": PRECOMPARE_SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
        )
    except InstructorRetryException as e:
        print(
            f"Warning! Model could not generate reply after {e.n_attempts} retires."
        )


def compare_query(client: Instructor, context: str, model_name: str) -> CompareQuery:
    try:
        system_prompt = COMPARE_SYSTEM_PROMPT

        decision = client.chat.completions.create(
            model=model_name,
            response_model=CompareQuery,
            max_retries=3,
            temperature=0.0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": context},
            ],
        )
        return decision
    except InstructorRetryException as e:
        logger.info(
            f"Warning! Model could not generate reply after {e.n_attempts} retires."
        )

        return CompareQuery(
            comparison="Nie udało mi się znaleźć odpowiedzi na zadaną kwestię.",
            further_questions = [],
        )