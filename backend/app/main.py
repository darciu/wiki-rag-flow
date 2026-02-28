from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Depends, HTTPException
from llm.interface import get_chat_response
from backend.app.schemas import ChatRequest, ChatResponse, FeedbackRequest, FeedbackResponse
from uuid import uuid4, UUID
from parser.nlp.toolkit import NLPToolkit

from backend.db.weaviate.connection import WeaviateManager
from config import WeaviateSettings
from openai import OpenAI
import instructor

InteractionStore = dict[UUID, dict]
FeedbackStore = list[dict]


def create_llm_client():
    raw = OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        timeout=5.0, 
    )
    llm_client = instructor.from_openai(raw, mode=instructor.Mode.JSON)
    return raw, llm_client


def create_weaviate_client():

    weaviate_settings = WeaviateSettings()
    weaviate_api_key = weaviate_settings.WEAVIATE_APIKEY_KEY

    weaviate_client = WeaviateManager(
        api_key=weaviate_api_key,
        host="127.0.0.1",
        native_embedding_url="http://127.0.0.1:8008/embed",
    )
    return weaviate_client

def verify_clients(raw_openai_client, weaviate_client, nlp_toolkit) -> None:

    try:
        raw_openai_client.models.list()
    except Exception as e:
        raise RuntimeError(f"LLM healthcheck failed: {e}") from e

    if not weaviate_client.is_healthy():
        raise RuntimeError("Weaviate healthcheck failed (is_healthy() is False)")

    if nlp_toolkit is None:
        raise RuntimeError("NLPToolkit is not initialized")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP

        raw, llm_client = create_llm_client()
        weaviate_client = create_weaviate_client()
        nlp_toolkit = NLPToolkit()

        verify_clients(raw, weaviate_client, nlp_toolkit)

        app.state.llm_client = llm_client
        app.state.weaviate_client = weaviate_client
        app.state.nlp_toolkit = nlp_toolkit

        yield

        weaviate_client.close()

app = FastAPI(title="WIKI RAG", version="0.1.0", lifespan=lifespan)

def get_llm_client(request: Request):
    try:
        return request.app.state.llm_client
    except AttributeError:
        raise HTTPException(status_code=503, detail="LLM client not initialized")


def get_weaviate_client(request: Request):
    try:
        return request.app.state.weaviate_client
    except AttributeError:
        raise HTTPException(status_code=503, detail="Weaviate client not initialized")


def get_nlp_toolkit(request: Request):
    try:
        return request.app.state.nlp_toolkit
    except AttributeError:
        raise HTTPException(status_code=503, detail="NLP toolkit not initialized")
    
def get_interactions_store(request: Request) -> InteractionStore:
    return request.app.state.interactions



@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(chat_reqest: ChatRequest,
         llm_client=Depends(get_llm_client),
            weaviate_client=Depends(get_weaviate_client),
            nlp_toolkit=Depends(get_nlp_toolkit),
            ):
    try:
        model_name = chat_reqest.model_name
        response_id = uuid4()

        answer, suggested_questions, route_type = get_chat_response(
            question=chat_reqest.question,     
            llm_client=llm_client,
            weaviate_client=weaviate_client,
            nlp_toolkit=nlp_toolkit,
            model_name=model_name,
        )
        chat_response = ChatResponse(answer=answer, suggested_prompts=suggested_questions, route=route_type, id=response_id)

        return chat_response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

# @app.post("/feedback", response_model=FeedbackResponse)
# def feedback
 