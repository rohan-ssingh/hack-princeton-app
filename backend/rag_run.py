# rag_run.py
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Any
import os
import re

from dotenv import load_dotenv
load_dotenv()

from load import Storage

ABSOLUTE_DEFAULT = "/Users/ankitwalishetti/Desktop/Ankit Programming/hack-princeton/backend/faiss_index"
FAISS_PATH = os.getenv("FAISS_INDEX_PATH") or ABSOLUTE_DEFAULT

def _ensure_index(path: str) -> None:
    p = Path(path)
    if not p.exists() or not p.is_dir():
        raise FileNotFoundError(f"FAISS_INDEX_PATH not a directory: {path}")
    req = [(p / "index.faiss"), (p / "index.pkl")]
    missing = [q.name for q in req if not q.exists()]
    if missing:
        raise FileNotFoundError(f"Missing files in {path}: {missing}")

def _date_range_list(start: Optional[str], end: Optional[str]) -> Optional[List[str]]:
    if not start and not end:
        return None
    return [start or "", end or ""]

def _make_summary(text: str, max_chars: int = 400) -> str:
    """Take first 2–3 sentences or cap to max_chars."""
    if not text:
        return "No summary returned."
    # split on sentence endings
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    s = " ".join(parts[:3])
    if len(s) > max_chars:
        s = s[:max_chars].rstrip() + "..."
    return s

def _build_sources(docs: List[Any]) -> List[Dict[str, str]]:
    sources = []
    for i, d in enumerate(docs, start=1):
        meta = getattr(d, "metadata", {}) or {}
        url = meta.get("url") or meta.get("link") or meta.get("source")
        title = meta.get("title") or meta.get("filename") or f"Doc {i}"
        if url:
            sources.append({"title": title, "url": url})
        else:
            # still include a title if no URL; optional
            sources.append({"title": title, "url": ""})
    return sources

def run_rag(
    prompt: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Dict[str, Any]:
    _ensure_index(FAISS_PATH)
    storage = Storage(path=FAISS_PATH, from_path=True)

    # Build date_range only if both are given
    date_range = [start_date or "", end_date or ""] if (start_date or end_date) else None

    # First attempt (with date_range if provided)
    result = storage.rag(question=prompt, schema=None, date_range=date_range)
    docs = result.get("documents", [])
    resp = result.get("response", "")

    # If we passed a date_range and got no docs, retry with no date filter
    if date_range and not docs:
        result = storage.rag(question=prompt, schema=None, date_range=None)
        docs = result.get("documents", [])
        resp = result.get("response", "")

    # Normalize response text
    if hasattr(resp, "model_dump_json"):
        resp_text = resp.model_dump_json()
    elif hasattr(resp, "json"):
        try:
            resp_text = resp.json()
        except Exception:
            resp_text = str(resp)
    else:
        resp_text = str(resp)

    summary = _make_summary(resp_text)
    content = resp_text if resp_text else "No content generated."
    feed_item = {
        "title": prompt,
        "summary": summary,
        "content": content,
        "sources": _build_sources(docs),
    }
    return feed_item

if __name__ == "__main__":
    try:
        item = run_rag("State of AI policy proposals in the last 90 days", start_date="2025-08-01")
        print("=== FEED ITEM ===")
        print("Title:", item["title"])
        print("Summary:\n", item["summary"], "\n")
        print("Article (first 800 chars):\n", item["content"][:800], "...\n")
        print("Sources:", item["sources"])
    except Exception as e:
        print("An error occurred:", e)