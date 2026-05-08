# LAB | Relevance Scoring and Rerankers for Trustworthy AI & EU AI Act

## Overview

This lab builds an advanced RAG pipeline over EU AI Act legal documents and Trustworthy AI podcast transcripts. It demonstrates the difference between baseline vector retrieval and reranked retrieval using Cohere Rerank and an optional local CrossEncoder, evaluated with Precision@5, MRR, and NDCG@5.

## Setup

```bash
git clone https://github.com/Lucas-Barrios/Relevance-Scoring-Rerankers.git
cd Relevance-Scoring-Rerankers
pip install -r requirements.txt
cp .env.example .env
# Edit .env and fill in your API keys
```

Place source files in `data/` — see [data/README.md](data/README.md) for details.

Open `relevance_scoring_rerankers.ipynb` in VS Code and run cells top to bottom.

## Required API Keys

| Key | Source |
|-----|--------|
| `OPENAI_API_KEY` | platform.openai.com |
| `PINECONE_API_KEY` | app.pinecone.io |
| `COHERE_API_KEY` | dashboard.cohere.com |

## Notebook Sections

| # | Section | Level |
|---|---------|-------|
| 0 | Setup and Environment | Core |
| 1 | Data Loading and Inspection | Core |
| 2 | Chunking and Metadata Tagging | Core |
| 3 | Embeddings and Pinecone Indexing | Core |
| 4 | Baseline Retrieval | Core |
| 5 | Metadata Filtering | Core |
| 6 | Query Enhancement | Core |
| 7 | LLM-Based Relevance Scoring | Advanced |
| 8 | Cohere Reranking | Advanced |
| 9 | CrossEncoder Reranking (local) | Advanced |
| 10 | Full RAG Pipeline with Reranking | Core |
| 11 | Manual Evaluation | Core |
| 12 | Results and Conclusion | Core |

## Evaluation Results

*(Fill in after running Section 11)*

| Pipeline | P@5 | MRR | NDCG@5 | Latency |
|----------|-----|-----|--------|---------|
| Baseline | — | — | — | — |
| LLM Scoring | — | — | — | — |
| Cohere Rerank | — | — | — | — |
| CrossEncoder | — | — | — | — |
