import os
from typing import List, Optional, Any, Tuple
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from parser.nlp.toolkit import NLPToolkit, NERModelName, KeywordsModelName, ChunkingModelName


app = FastAPI(title="NLP Toolkit Microservice")

NER_MODEL: NERModelName = os.getenv("NLP_NER_MODEL", "herbert")
KEYWORDS_MODEL: KeywordsModelName = os.getenv("NLP_KEYWORDS_MODEL", "keybert")
CHUNKING_MODEL: ChunkingModelName = os.getenv("NLP_CHUNKING_MODEL", "langchain")


print(f"Loading NLP Models: NER={NER_MODEL}, KW={KEYWORDS_MODEL}, CHUNK={CHUNKING_MODEL}...")
nlp_toolkit = NLPToolkit(
    ner_model_name=NER_MODEL,
    keywords_model_name=KEYWORDS_MODEL,
    chunking_model_name=CHUNKING_MODEL
)
print("NLP Models loaded successfully.")

class TextInput(BaseModel):
    texts: List[str]

class LemmatizeInput(TextInput):
    batch_size: int = 512

class ReadabilityInput(TextInput):
    batch_size: int = 100

class ChunkInput(TextInput):
    max_tokens: int

# --- Endpointy ---

@app.get("/health")
def health_check():
    return {"status": "ok", "models_loaded": True}

@app.post("/ner")
def extract_ner(payload: TextInput) -> List[Any]:
    try:
        return nlp_toolkit.extract_ner_entities(payload.texts)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/keywords")
def extract_keywords(payload: TextInput) -> List[Any]: 
    try:
        return nlp_toolkit.extract_keywords(payload.texts)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/lemmatize")
def lemmatize(payload: LemmatizeInput) -> List[str]:
    try:
        return nlp_toolkit.lemmatize(payload.texts, batch_size=payload.batch_size)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/readability")
def readability(payload: ReadabilityInput) -> List[float]:
    try:
        return nlp_toolkit.texts_readability_fog(payload.texts, batch_size=payload.batch_size)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chunk")
def chunk_texts(payload: ChunkInput) -> List[List[str]]:
    try:
        return nlp_toolkit.chunk_texts(payload.texts, max_tokens=payload.max_tokens)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))