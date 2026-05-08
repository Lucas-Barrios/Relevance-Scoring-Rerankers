"""Text chunking and metadata tagging for PDFs and podcast transcripts."""
from pathlib import Path
from typing import List, Dict
from pypdf import PdfReader
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

CHUNK_SIZE = 2000   # characters (~500 tokens)
CHUNK_OVERLAP = 200


def chunk_pdf(
    path: str,
    doc_id: str,
    doc_title: str,
    category: str = "regulation",
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[Dict]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    reader = PdfReader(path)
    chunks = []
    chunk_index = 0

    source_type = "legal_document" if category == "regulation" else "pdf"

    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if not text.strip():
            continue
        for local_i, chunk_text in enumerate(splitter.split_text(text)):
            chunk_id = f"{doc_id}_p{page_num}_c{local_i}"
            parent_id = f"{doc_id}_p{page_num}"
            chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text,
                "metadata": {
                    "source_type": source_type,
                    "doc_id": doc_id,
                    "doc_title": doc_title,
                    "category": category,
                    "page": page_num,
                    "section": "",
                    "chunk_index": chunk_index,
                    "chunk_id": chunk_id,
                    "parent_id": parent_id,
                    "char_count": len(chunk_text),
                    "language": "en",
                },
            })
            chunk_index += 1

    return chunks


def chunk_transcript(
    path: str,
    doc_id: str,
    doc_title: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[Dict]:
    text = Path(path).read_text(encoding="utf-8")
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = []

    for i, chunk_text in enumerate(splitter.split_text(text)):
        chunk_id = f"{doc_id}_c{i}"
        parent_id = f"{doc_id}_segment_{i // 3}"
        chunks.append({
            "chunk_id": chunk_id,
            "text": chunk_text,
            "metadata": {
                "source_type": "podcast_transcript",
                "doc_id": doc_id,
                "doc_title": doc_title,
                "category": "podcast",
                "speaker": "unknown",
                "timestamp_start": 0,
                "timestamp_end": 0,
                "chunk_index": i,
                "chunk_id": chunk_id,
                "parent_id": parent_id,
                "char_count": len(chunk_text),
                "language": "en",
            },
        })

    return chunks
