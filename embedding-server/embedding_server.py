from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import torch

MODEL_NAME = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"

app = FastAPI()

device = "mps" if torch.backends.mps.is_available() else "cpu"
model = SentenceTransformer(MODEL_NAME, device=device)

class EmbedRequest(BaseModel):
    texts: list[str]
    normalize: bool = True

@app.get("/health")
def health():
    return {"status": "ok", "device": device, "model": MODEL_NAME}

@app.post("/embed")
def embed(req: EmbedRequest):
    emb = model.encode(
        req.texts,
        convert_to_numpy=True,
        normalize_embeddings=req.normalize,
        batch_size=64,
        show_progress_bar=False,
    )
    return {"vectors": emb.tolist(), "dim": len(emb[0]) if len(emb) else 0}