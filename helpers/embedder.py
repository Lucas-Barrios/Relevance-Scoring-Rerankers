"""OpenAI embedding wrapper — text-embedding-3-small, 1536 dims."""
import os
from typing import List
from openai import OpenAI

MODEL = "text-embedding-3-small"
DIMENSION = 1536


def _client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set; skipping OpenAI embedding call.")
    return OpenAI(api_key=api_key)


def embed_texts(texts: List[str], model: str = MODEL) -> List[List[float]]:
    if not texts:
        return []
    response = _client().embeddings.create(input=texts, model=model)
    return [item.embedding for item in response.data]


def embed_query(query: str, model: str = MODEL) -> List[float]:
    return embed_texts([query], model)[0]
