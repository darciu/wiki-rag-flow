import asyncio
import gc

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from starlette.concurrency import run_in_threadpool

MODEL_NAME = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"

device = "mps" if torch.backends.mps.is_available() else "cpu"

app = FastAPI()
model = SentenceTransformer(MODEL_NAME, device=device)
model.eval()

sem = asyncio.Semaphore(1)
REQS_BETWEEN_CLEAN = 50
_req_counter = 0


class EmbedRequest(BaseModel):
    texts: list[str]
    normalize: bool = True


@app.get("/health")
def health():
    return {"status": "ok", "device": device, "model": MODEL_NAME}


def _encode_sync(texts: list[str], normalize: bool):
    with torch.inference_mode():
        emb = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
            batch_size=64,
            show_progress_bar=False,
        )
    return emb


def _maybe_clean():
    global _req_counter
    _req_counter += 1
    if _req_counter % REQS_BETWEEN_CLEAN != 0:
        return
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()


@app.post("/embed")
async def embed(req: EmbedRequest):
    if not req.texts:
        return {"vectors": [], "dim": 0}

    if len(req.texts) > 10000:
        raise HTTPException(413, "Too many texts in one request")

    async with sem:
        emb = await run_in_threadpool(_encode_sync, req.texts, req.normalize)
        _maybe_clean()

    return {"vectors": emb.tolist(), "dim": len(emb[0]) if len(emb) else 0}
