"""Reranking helpers: LLM scoring, Cohere Rerank, optional CrossEncoder."""
import os
from typing import List, Dict


def llm_score_relevance(
    query: str,
    candidates: List[Dict],
    client,
    top_n: int = 5,
    model: str = "gpt-4o-mini",
) -> List[Dict]:
    """Score each candidate 0–10 with an LLM and return top_n by score."""
    scored = []
    for c in candidates:
        prompt = (
            "Rate the relevance of the text below to the query on a scale of 0 to 10.\n"
            f"Query: {query}\n"
            f"Text: {c['text'][:600]}\n"
            "Return only a single integer between 0 and 10."
        )
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5,
            temperature=0,
        )
        try:
            score = float(resp.choices[0].message.content.strip())
        except ValueError:
            score = 0.0
        scored.append({**c, "llm_score": score})
    return sorted(scored, key=lambda x: x["llm_score"], reverse=True)[:top_n]


def cohere_rerank(
    query: str,
    candidates: List[Dict],
    top_n: int = 5,
    model: str = "rerank-v3.5",
) -> List[Dict]:
    """Rerank candidates using the Cohere Rerank API."""
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        print("COHERE_API_KEY is not set; returning baseline candidate order.")
        return candidates[:top_n]
    try:
        import cohere
    except ImportError:
        print("cohere is not installed; returning baseline candidate order.")
        return candidates[:top_n]

    co = cohere.ClientV2(api_key=api_key)
    docs = [c["text"] for c in candidates]
    if not docs:
        return []
    response = co.rerank(model=model, query=query, documents=docs, top_n=top_n)
    return [{**candidates[r.index], "rerank_score": r.relevance_score} for r in response.results]


def crossencoder_rerank(
    query: str,
    candidates: List[Dict],
    top_n: int = 5,
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
) -> List[Dict]:
    """Rerank using a local CrossEncoder. Requires sentence-transformers + torch."""
    try:
        from sentence_transformers import CrossEncoder
        model = CrossEncoder(model_name)
        pairs = [(query, c["text"]) for c in candidates]
        scores = model.predict(pairs)
        ranked = sorted(
            [{**c, "crossencoder_score": float(s)} for c, s in zip(candidates, scores)],
            key=lambda x: x["crossencoder_score"],
            reverse=True,
        )
        return ranked[:top_n]
    except ImportError:
        print("sentence-transformers not installed.")
        print("To enable: uncomment sentence-transformers and torch in requirements.txt, then pip install.")
        return candidates[:top_n]
