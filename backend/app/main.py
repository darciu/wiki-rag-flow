from contextlib import asynccontextmanager
from uuid import uuid4
import os 
import instructor
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from openai import OpenAI

from backend.app.schemas import (
    ChatRequest,
    ChatResponse,
    FeedbackRequest,
    FeedbackResponse,
)
from backend.db.weaviate.connection import WeaviateManager
from config import WeaviateSettings, OllamaSettings
from llm.interface import get_chat_response
from nlp.toolkit import NLPToolkit


def create_llm_client():
    ollama_settings = OllamaSettings()
    ollama_base_url = ollama_settings.OLLAMA_BASE_URL
    
    raw = OpenAI(
        base_url=ollama_base_url,
        api_key="ollama",
        timeout=5.0, 
    )
    llm_client = instructor.from_openai(raw, mode=instructor.Mode.JSON)
    return raw, llm_client


def create_weaviate_client():

    weaviate_settings = WeaviateSettings()
    weaviate_api_key = weaviate_settings.WEAVIATE_APIKEY_KEY
    weaviate_host = weaviate_settings.WEAVIATE_HOST
    embed_url = weaviate_settings.EMBEDDING_SERVER_URL


    weaviate_client = WeaviateManager(
        api_key=weaviate_api_key,
        host=weaviate_host,
        native_embedding_url=embed_url,
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
        app.state.chat_last_session = None
        app.state.app_run_id = uuid4()

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
    



@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(chat_reqest: ChatRequest,
         request: Request, response: Response,
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
        

        session_id = request.cookies.get("session_id")
        if not session_id:
            session_id = str(uuid4())
            response.set_cookie(key="session_id", value=session_id, httponly=True)

        chat_response = ChatResponse(answer=answer, suggested_prompts=suggested_questions, route=route_type, id=response_id,
                                     user_agent = request.headers.get("user-agent"),
                                     session_id = session_id,
                                     app_run_id = request.app.state.app_run_id,
                                     )

        # for feedback
        request.app.state.chat_last_session = {
            "chat_response_id": response_id,
            "chat_route": route_type
        }

        return chat_response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.post("/feedback", response_model=FeedbackResponse)
def post_feedback(feedback_request: FeedbackRequest, request: Request):

    chat_last_session = request.app.state.chat_last_session
    if not chat_last_session:
        raise HTTPException(
            status_code=400, 
            detail="Feedback cannot be submitted. No active chat response found or feedback already sent."
        )

    feedback_response = FeedbackResponse(
        rating=feedback_request.rating,
        chat_route=chat_last_session["chat_route"],
        chat_response_id=chat_last_session["chat_response_id"],
        user_agent = request.headers.get("user-agent"),
        session_id = request.cookies.get("session_id"),
        app_run_id = request.app.state.app_run_id
    )
    request.app.state.chat_last_session = None

    try:
        pass
    # save data to DB
    except:
        raise HTTPException(status_code=500, detail="Failed to save feedback to database.")

    return feedback_response
 