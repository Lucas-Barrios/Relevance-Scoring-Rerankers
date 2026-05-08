"""Manual evaluation metrics: Precision@K, MRR, NDCG@K."""
import json
import numpy as np
from typing import List, Dict


def load_eval_queries(path: str = "eval_queries.json") -> List[Dict]:
    with open(path) as f:
        return json.load(f)


def precision_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int = 5) -> float:
    if k <= 0:
        return 0.0
    hits = sum(1 for r in retrieved_ids[:k] if r in relevant_ids)
    return hits / k


def mrr(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    for i, r in enumerate(retrieved_ids, start=1):
        if r in relevant_ids:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved_ids: List[str], relevance_map: Dict[str, int], k: int = 5) -> float:
    dcg = sum(relevance_map.get(r, 0) / np.log2(i + 2) for i, r in enumerate(retrieved_ids[:k]))
    ideal = sorted(relevance_map.values(), reverse=True)[:k]
    idcg = sum(g / np.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_pipeline(
    pipeline_name: str,
    results_per_query: List[List[Dict]],
    eval_queries: List[Dict],
    k: int = 5,
) -> Dict:
    p_scores, mrr_scores, ndcg_scores = [], [], []
    for i, eq in enumerate(eval_queries):
        results = results_per_query[i] if i < len(results_per_query) else []
        relevant_ids = set(eq.get("relevant_chunk_ids", []))
        relevance_map = eq.get("relevance_map") or {r: 2 for r in relevant_ids}
        retrieved_ids = [r.get("chunk_id") or r.get("id") for r in results]
        p_scores.append(precision_at_k(retrieved_ids, list(relevant_ids), k))
        mrr_scores.append(mrr(retrieved_ids, list(relevant_ids)))
        ndcg_scores.append(ndcg_at_k(retrieved_ids, relevance_map, k))
    return {
        "pipeline": pipeline_name,
        "precision_at_5": round(float(np.mean(p_scores)), 3),
        "mrr": round(float(np.mean(mrr_scores)), 3),
        "ndcg_at_5": round(float(np.mean(ndcg_scores)), 3),
    }
