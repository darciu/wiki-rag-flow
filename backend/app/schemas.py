from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class RouteType(StrEnum):
    RAG_SEARCH = "RAG_SEARCH"
    DIRECT = "DIRECT"
    CLARIFY = "CLARIFY"

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    model_name: str = "llama3.2"
    id: UUID = Field(default_factory=uuid4, description="Chat request ID")
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ChatResponse(BaseModel):
    answer: str = Field(..., description="Chat response")
    suggested_prompts: list[str] = Field(default_factory=list, description="Suggest")
    route: RouteType = Field(..., description="LLMs routing")
    id: UUID = Field(default_factory=uuid4, description="Chat response ID")
    user_agent: str | None = Field(None, description="User data from browser")
    session_id: str  = Field(..., description="Cookie session id")
    app_run_id: UUID = Field(..., description="FastAPI run ID")
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

class FeedbackRequest(BaseModel):
    rating: Literal["up", "down"]

class FeedbackResponse(BaseModel):
    rating: Literal["up", "down"]
    chat_route: RouteType = Field(..., description="LLMs routing")
    chat_response_id: UUID = Field(..., description="Chat response ID")
    user_agent: str | None = Field(None, description="User data from browser")
    session_id: str  = Field(..., description="Cookie session id")
    app_run_id: UUID = Field(..., description="FastAPI run ID")
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))



