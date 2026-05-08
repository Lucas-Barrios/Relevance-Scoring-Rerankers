# Retrieval Evidence

## Purpose
This file summarizes the visible notebook evidence for baseline retrieval, metadata filtering, reranking, generated answers, and evaluation metrics. Full outputs remain in `relevance_scoring_rerankers.ipynb`.

## Evaluation Queries
| ID | Query | Filter |
|---|---|---|
| q1 | What AI practices are prohibited under the EU AI Act? | `category = regulation` |
| q2 | How does the EU AI Act define high-risk AI systems? | `category = regulation` |
| q3 | What transparency obligations apply to AI systems under EU law? | None |
| q4 | What are the key principles of trustworthy AI? | `source_type = podcast_transcript` |
| q5 | How should AI systems handle fundamental rights in the EU? | None |
| q6 | What conformity assessment procedures are required for high-risk AI? | `category = regulation` |

## Final Metrics
| Pipeline | P@5 | MRR | NDCG@5 |
|---|---:|---:|---:|
| Baseline | 0.933 | 0.917 | 0.895 |
| Cohere Rerank | 0.767 | 0.917 | 0.780 |
| LLM Relevance Scoring | 0.633 | 0.917 | 0.671 |

## Example Before/After Reranking
Query: `What AI practices are prohibited under the EU AI Act?`

| Rank | Baseline Result | Cohere Reranked Result |
|---:|---|---|
| 1 | `eu_ai_act_p51_c0` | `eu_ai_act_p51_c0` |
| 2 | `eu_ai_act_p12_c1` | `eu_ai_act_p9_c0` |
| 3 | `eu_ai_act_p9_c0` | `eu_ai_act_p12_c1` |
| 4 | `eu_ai_act_p45_c0` | `hleg_guidelines_en_p8_c1` |
| 5 | `hleg_guidelines_en_p4_c2` | `eu_ai_act_p45_c0` |

## Example RAG Answer
**Query:** What are the prohibited AI practices in the EU AI Act?

**Answer:** The generated answer identifies prohibited AI practices such as manipulative or subliminal techniques, exploitation of vulnerable groups, social scoring, profiling-based criminal-risk assessment without objective facts, untargeted facial-image scraping, and emotion inference in workplace or education contexts except for medical or safety reasons. The answer cites the EU AI Act context from Article 5.

## Interpretation
Baseline retrieval performed best on this small manually labeled dataset, with the highest Precision@5 and NDCG@5 and tied MRR. Reranking was still useful to test because legal and compliance RAG often requires high precision, especially when candidate sets are larger, noisier, or more heterogeneous. In this run, reranking added cost and latency without improving the labeled retrieval metrics.

