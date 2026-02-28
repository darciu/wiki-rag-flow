from pydantic import BaseModel, Field
from typing import Literal
from enum import StrEnum
from uuid import UUID, uuid4
from datetime import datetime

class RouteType(StrEnum):
    RAG_SEARCH = "RAG_SEARCH"
    DIRECT = "DIRECT"
    CLARIFY = "CLARIFY"

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    model_name: str = "llama3.2"
    id: UUID = Field(default_factory=uuid4, description="Chat request ID")


class ChatResponse(BaseModel):
    answer: str = Field(..., description="Chat response")
    suggested_prompts: list[str] = Field(default_factory=list, description="Suggest")
    route: RouteType = Field(..., description="LLMs routing")
    id: UUID = Field(default_factory=uuid4, description="Chat response ID")

class FeedbackRequest(BaseModel):
    rating: Literal["up", "down"]

class FeedbackResponse(BaseModel):
    rating: Literal["up", "down"]
    route: RouteType = Field(..., description="LLMs routing")
    id: UUID = Field(default_factory=uuid4, description="Chat response ID")
    created_at: datetime



