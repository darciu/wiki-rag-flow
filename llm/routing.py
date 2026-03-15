from typing import TypedDict, Annotated, List
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage
import logging
from enum import StrEnum
from pydantic import BaseModel, Field, model_validator
from instructor.core.client import Instructor
from instructor.exceptions import InstructorRetryException
from llm.prompts import PLANNER_SYSTEM_PROMPT, DIRECT_ANSWER_SYSTEM_PROMPT
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

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


from enum import StrEnum


class RouteType(StrEnum):
    RAG_SEARCH = "rag_search"
    DIRECT = "direct"
    CLARIFY = "clarify"
    MATH = "math"

class TaskType(StrEnum):
    LOOKUP = "lookup"
    COMPARE = "compare"
    SUMMARIZE = "summarize"

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