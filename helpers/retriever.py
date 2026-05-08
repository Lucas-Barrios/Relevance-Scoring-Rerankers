"""Pinecone retrieval helpers."""
import os
from typing import List, Dict, Optional

INITIAL_TOP_K = 12
FINAL_TOP_K = 5


def get_index():
    from pinecone import Pinecone

    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("PINECONE_API_KEY is not set; skipping Pinecone retrieval call.")
    pc = Pinecone(api_key=api_key)
    return pc.Index(os.getenv("PINECONE_INDEX_NAME", "relevance-scoring-lab"))


def normalize_filter(metadata_filter: Optional[Dict]) -> Optional[Dict]:
    """Convert simple equality filters to Pinecone's explicit operator format."""
    if not metadata_filter:
        return None
    normalized = {}
    for key, value in metadata_filter.items():
        if key in {"$and", "$or"}:
            normalized[key] = [
                normalize_filter(item) if isinstance(item, dict) else item
                for item in value
            ]
        elif isinstance(value, dict):
            normalized[key] = value
        else:
            normalized[key] = {"$eq": value}
    return normalized


def retrieve(
    query_embedding: List[float],
    top_k: int = INITIAL_TOP_K,
    metadata_filter: Optional[Dict] = None,
) -> List[Dict]:
    index = get_index()
    kwargs: Dict = {"vector": query_embedding, "top_k": top_k, "include_metadata": True}
    if metadata_filter:
        kwargs["filter"] = normalize_filter(metadata_filter)

    results = index.query(**kwargs)

    chunks = []
    for m in results.matches:
        meta = dict(m.metadata)
        text = meta.pop("text", "")
        chunks.append({
            "chunk_id": m.id,
            "text": text,
            "score": m.score,
            "metadata": meta,
        })
    return chunks
