"""
Simple in-memory RAG engine using keyword-based text retrieval.
Stores resume chunks and retrieves relevant context for interview questions.
"""

import math
from typing import Dict, List
from collections import Counter

# In-memory storage: session_id -> list of text chunks
_store: Dict[str, List[str]] = {}


def _tokenize(text: str) -> List[str]:
    """Simple tokenization: lowercase, split on non-alpha."""
    import re
    return re.findall(r'[a-zA-Z]+', text.lower())


def _compute_similarity(query_tokens: List[str], doc_tokens: List[str]) -> float:
    """Compute simple cosine similarity between token sets."""
    if not query_tokens or not doc_tokens:
        return 0.0

    query_counts = Counter(query_tokens)
    doc_counts = Counter(doc_tokens)

    all_tokens = set(query_counts.keys()) | set(doc_counts.keys())

    dot_product = sum(query_counts.get(t, 0) * doc_counts.get(t, 0) for t in all_tokens)
    q_norm = math.sqrt(sum(v ** 2 for v in query_counts.values()))
    d_norm = math.sqrt(sum(v ** 2 for v in doc_counts.values()))

    if q_norm == 0 or d_norm == 0:
        return 0.0

    return dot_product / (q_norm * d_norm)


def store_resume_chunks(session_id: str, text: str, chunk_size: int = 100):
    """Split resume text into chunks and store in memory."""
    words = text.split()
    chunks = []
    step = max(chunk_size, 1)

    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + step])
        if chunk.strip():
            chunks.append(chunk)

    if not chunks:
        chunks = [text]

    _store[session_id] = chunks


def retrieve_context(session_id: str, query: str, n_results: int = 3) -> str:
    """Retrieve relevant resume context for a query."""
    chunks = _store.get(session_id, [])
    if not chunks:
        return ""

    query_tokens = _tokenize(query)

    scored = []
    for chunk in chunks:
        doc_tokens = _tokenize(chunk)
        score = _compute_similarity(query_tokens, doc_tokens)
        scored.append((score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)

    top_chunks = [chunk for _, chunk in scored[:n_results]]
    return "\n".join(top_chunks)


def cleanup_session(session_id: str):
    """Remove session data."""
    _store.pop(session_id, None)


def get_full_resume_text(session_id: str) -> str:
    """Get the full resume text for a session."""
    chunks = _store.get(session_id, [])
    return " ".join(chunks)