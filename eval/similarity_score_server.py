from fastapi import FastAPI
from pydantic import BaseModel
import torch
from sentence_transformers import SentenceTransformer

MODEL_ID = "intfloat/e5-base-v2"

# Load model once at startup (GPU if available)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer(MODEL_ID, device=device)

app = FastAPI(title="Sentence Similarity API", version="1.0.0")


class SentencePair(BaseModel):
    s1: str
    s2: str


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID, "device": device}


@app.post("/similarity")
@torch.inference_mode()
def similarity(pair: SentencePair):
    # Encode and normalize embeddings
    embeddings = model.encode(
        [pair.s1, pair.s2],
        normalize_embeddings=True,
        convert_to_tensor=True,
        device=device
    )

    # Cosine similarity (dot product because normalized)
    similarity_score = torch.sum(embeddings[0] * embeddings[1]).item()

    return {
        "similarity": float(similarity_score),
        "s1": pair.s1,
        "s2": pair.s2,
        "model": MODEL_ID
    }
