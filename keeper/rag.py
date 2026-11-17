"""Small persistent, local vector index; no external database service required."""
from __future__ import annotations

import json
import math
import os
import sqlite3
from contextlib import closing
from pathlib import Path

import httpx

EMBED_MODEL = "bge-m3"


def data_dir() -> Path:
    path = Path(os.getenv("KEEPER_DATA_DIR", str(Path(__file__).resolve().parents[1] / "data")))
    path.mkdir(parents=True, exist_ok=True)
    return path


def database_path() -> Path:
    return data_dir() / "knowledge.sqlite3"


def embed(texts: list[str]) -> list[list[float]]:
    base = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    with httpx.Client(timeout=180) as client:
        response = client.post(base + "/api/embed", json={
            "model": EMBED_MODEL, "input": texts, "truncate": False,
            "keep_alive": "5m",
        })
        response.raise_for_status()
        result = response.json().get("embeddings", [])
    if len(result) != len(texts) or any(not v for v in result):
        raise RuntimeError("Ollama returned an invalid embedding batch")
    return result


def document_count() -> int:
    if not database_path().exists():
        return 0
    with closing(sqlite3.connect(database_path())) as db:
        return db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]


def cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Embedding dimensions differ; rebuild the PDF index")
    norm = math.sqrt(sum(x*x for x in left) * sum(y*y for y in right))
    return sum(x*y for x, y in zip(left, right)) / norm if norm else 0.0


def search(query: str, section: str = "scenario", k: int = 3) -> list[dict]:
    """Search only the requested section; returned text is GM-only context."""
    if section not in {"rules", "scenario", "handout", "character"}:
        raise ValueError("Unknown document section")
    if not database_path().exists():
        raise RuntimeError("PDF index is missing. Run python -m keeper.ingest --pdf PATH")
    query = query.strip()[:2000]
    if not query:
        return []
    vector = embed([query])[0]
    with closing(sqlite3.connect(database_path())) as db:
        db.row_factory = sqlite3.Row
        model = db.execute("SELECT value FROM metadata WHERE key='embedding_model'").fetchone()
        if not model or model[0] != EMBED_MODEL:
            raise RuntimeError("Embedding model changed; rebuild the PDF index")
        rows = db.execute("SELECT source,page,section,text,embedding FROM chunks WHERE section=?", (section,)).fetchall()
    results = []
    for row in rows:
        item = dict(row)
        item["score"] = cosine(vector, json.loads(item.pop("embedding")))
        results.append(item)
    return sorted(results, key=lambda item: item["score"], reverse=True)[:max(1, min(int(k), 5))]
