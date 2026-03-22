import logging
import os
from contextlib import asynccontextmanager
from typing import Any, cast
from uuid import uuid4

import instructor
import requests
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama
from openai import OpenAI
from openinference.instrumentation.langchain import LangChainInstrumentor
from openinference.instrumentation.openai import OpenAIInstrumentor
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from backend.app.schemas import (
    ChatRequest,
    ChatResponse,
    FeedbackRequest,
    FeedbackResponse,
)
from backend.db.weaviate.connection import WeaviateManager
from config import OllamaSettings, WeaviateSettings
from llm.graph import agent
from logger_config import setup_logging
from nlp.toolkit import NLPToolkit

setup_logging("backend")
logger = logging.getLogger(__name__)
logging.raiseExceptions = False


def create_instructor_client():
    ollama_settings = OllamaSettings()
    ollama_base_url = ollama_settings.OLLAMA_BASE_URL + "/v1"

    raw = OpenAI(
        base_url=ollama_base_url,
        api_key="ollama",
        timeout=30.0,
    )
    instructor_client = instructor.from_openai(raw, mode=instructor.Mode.JSON)
    return raw, instructor_client


def create_langchain_client():
    ollama_settings = OllamaSettings()
    ollama_base_url = ollama_settings.OLLAMA_BASE_URL

    langchain_client = ChatOllama(
        model="llama3.2", temperature=0.0, base_url=ollama_base_url
    )
    return langchain_client


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
        logger.exception(f"LLM healthcheck failed: {e}")
        raise RuntimeError(f"LLM healthcheck failed: {e}") from e

    if not weaviate_client.is_healthy():
        logger.error("Weaviate healthcheck failed (is_healthy() is False)")
        raise RuntimeError("Weaviate healthcheck failed (is_healthy() is False)")

    if nlp_toolkit is None:
        logger.error("NLPToolkit is not initialized")
        raise RuntimeError("NLPToolkit is not initialized")


def setup_phoenix_tracing():
    endpoint = os.getenv(
        "PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006/v1/traces"
    )

    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint))
    )
    trace.set_tracer_provider(tracer_provider)

    LangChainInstrumentor().instrument()
    OpenAIInstrumentor().instrument()

    logger.info(f"Phoenix tracing initialized. Sending traces to: {endpoint}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP

    setup_phoenix_tracing()

    raw_instructor, instructor_client = create_instructor_client()
    langchain_client = create_langchain_client()
    weaviate_client = create_weaviate_client()
    nlp_toolkit = NLPToolkit()

    logger.info("Warming up CrossEncoder.")
    try:
        nlp_toolkit.rank("test query", ["test context chunk"])
        logger.info("CrossEncoder has been successfully loaded")
    except Exception as e:
        logger.exception(f"Could not load CrossEncoder due to error: {e}")
        raise RuntimeError(f"Could not load CrossEncoder due to error: {e}") from e

    verify_clients(raw_instructor, weaviate_client, nlp_toolkit)

    app.state.instructor_client = instructor_client
    app.state.langchain_client = langchain_client
    app.state.weaviate_client = weaviate_client
    app.state.nlp_toolkit = nlp_toolkit
    app.state.chat_last_session = None
    app.state.app_run_id = uuid4()

    yield
    logger.info("Shutting down connection to Weaviate.")
    weaviate_client.close()


app = FastAPI(title="WIKI RAG", version="0.1.0", lifespan=lifespan)


def get_instructor_client(request: Request):
    try:
        return request.app.state.instructor_client
    except AttributeError as err:
        raise HTTPException(
            status_code=503, detail="Instructor client not initialized"
        ) from err


def get_langchain_client(request: Request):
    try:
        return request.app.state.langchain_client
    except AttributeError as err:
        logger.exception(
            "Failed to retrieve Langchain client: it is not initialized in the app state."
        )
        raise HTTPException(
            status_code=503, detail="Langchain client not initialized"
        ) from err


def get_weaviate_client(request: Request):
    try:
        return request.app.state.weaviate_client
    except AttributeError as err:
        logger.exception(
            "Failed to retrieve Weaviate client: it is not initialized in the app state."
        )
        raise HTTPException(
            status_code=503, detail="Weaviate client not initialized"
        ) from err


def get_nlp_toolkit(request: Request):
    try:
        return request.app.state.nlp_toolkit
    except AttributeError as err:
        logger.exception(
            "Failed to retrieve NLP Toolkit: it is not initialized in the app state."
        )
        raise HTTPException(
            status_code=503, detail="NLP toolkit not initialized"
        ) from err


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/models")
def get_installed_models():
    try:
        ollama_settings = OllamaSettings()
        ollama_base_url = ollama_settings.OLLAMA_BASE_URL

        response = requests.get(f"{ollama_base_url}/api/tags", timeout=10)
        response.raise_for_status()

        data = response.json()

        # only LLMs
        llm_models = []
        for model in data.get("models", []):
            name = model.get("name", "").lower()
            details = model.get("details", {})

            family = details.get("family", "").lower()
            families = [f.lower() for f in (details.get("families") or [])]

            if (
                "bert" in family
                or "nomic-bert" in family
                or any("bert" in f for f in families)
            ):
                continue

            if "embed" in name or "bge" in name or "minilm" in name:
                continue

            llm_models.append(model["name"])

        if not llm_models:
            llm_models = ["llama3.2"]

        return {"models": llm_models}
    except Exception as e:
        logger.error(f"Failed to fetch models from Ollama: {e}")
        return {"models": ["llama3.2"]}


@app.post("/chat", response_model=ChatResponse)
async def chat(
    chat_request: ChatRequest,
    request: Request,
    response: Response,
    instructor_client=Depends(get_instructor_client),  # noqa: B008
    langchain_client=Depends(get_langchain_client),  # noqa: B008
    weaviate_client=Depends(get_weaviate_client),  # noqa: B008
    nlp_toolkit=Depends(get_nlp_toolkit),  # noqa: B008
):
    try:
        session_id = request.cookies.get("session_id")
        if not session_id:
            session_id = str(uuid4())
            response.set_cookie(key="session_id", value=session_id, httponly=True)
        model_name = chat_request.model_name
        response_id = uuid4()

        config: RunnableConfig = {
            "configurable": {
                "thread_id": session_id,
                "model_name": model_name,
                "instructor_client": instructor_client,
                "weaviate_client": weaviate_client,
                "nlp_toolkit": nlp_toolkit,
                "langchain_client": langchain_client,
            }
        }

        initial_state = {
            "messages": [HumanMessage(content=chat_request.question)],
            "current_query": chat_request.question,
        }

        logger.info(f"Agent invoke with qustion: {chat_request.question}")
        result_agent_state = agent.invoke(cast(Any, initial_state), config=config)

        answer = result_agent_state["messages"][-1].content
        suggested_questions = result_agent_state.get("further_questions", [])

        chat_response = ChatResponse(
            answer=answer,
            suggested_prompts=suggested_questions,
            id=response_id,
            user_agent=request.headers.get("user-agent"),
            session_id=session_id,
            app_run_id=request.app.state.app_run_id,
        )

        # for feedback
        request.app.state.chat_last_session = {
            "chat_response_id": response_id,
        }

        return chat_response

    except Exception as err:
        logger.exception(f"Endpoint: /chat. Error has occured: {err}")
        raise HTTPException(status_code=500, detail=str(err)) from err


@app.post("/feedback", response_model=FeedbackResponse)
def post_feedback(feedback_request: FeedbackRequest, request: Request):

    chat_last_session = request.app.state.chat_last_session
    if not chat_last_session:
        logger.error(
            "Feedback cannot be submitted. No active chat response found or feedback already sent."
        )
        raise HTTPException(
            status_code=400,
            detail="Feedback cannot be submitted. No active chat response found or feedback already sent.",
        )

    feedback_response = FeedbackResponse(
        rating=feedback_request.rating,
        chat_route=chat_last_session["chat_route"],
        chat_response_id=chat_last_session["chat_response_id"],
        user_agent=request.headers.get("user-agent"),
        session_id=request.cookies.get("session_id"),
        app_run_id=request.app.state.app_run_id,
    )
    request.app.state.chat_last_session = None

    try:
        pass
    # save data to DB
    except Exception as err:
        logger.exception(f"Endpoint: /feedback. Error has occured: {err}")
        raise HTTPException(
            status_code=500, detail="Failed to save feedback to database."
        ) from err

    return feedback_response
