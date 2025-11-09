# rag_run.py
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Any
from load import Storage

# ----------------------------
# Config
# ----------------------------
# Explicit absolute path to your FAISS folder
FAISS_PATH = "/Users/ankitwalishetti/Desktop/Ankit Programming/hack-princeton/backend/faiss_index"

# Optional: date range parser
def parse_range(
    start: Optional[str] = None,  # e.g. "2024-01-01"
    end: Optional[str] = None,    # e.g. "2025-12-31"
) -> Optional[Tuple[datetime, datetime]]:
    if not start and not end:
        return None
    s = datetime.min if not start else datetime.fromisoformat(start)
    e = datetime.max if not end else datetime.fromisoformat(end)
    return (s, e)

# ----------------------------
# Feed item builder
# ----------------------------
def build_feed_item(
    title: str,
    summary: str,
    article: str,
    docs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    sources = []
    for d in docs:
        meta = d.get("metadata") or {}
        src = meta.get("url") or meta.get("link") or meta.get("source")
        src_title = meta.get("title") or meta.get("filename") or "Source"
        if src:
            sources.append({"title": src_title, "url": src})

    return {
        "title": title,
        "summary": summary,
        "content": article,
        "sources": sources,
    }

# ----------------------------
# Simple article generator
# ----------------------------
def stitch_article(prompt: str, docs: List[Dict[str, Any]], summary: str) -> str:
    bullet_sources = []
    for i, d in enumerate(docs[:8], start=1):
        meta = d.get("metadata") or {}
        src = meta.get("url") or meta.get("source") or ""
        title = meta.get("title") or meta.get("filename") or f"Doc {i}"
        bullet_sources.append(f"- {title}: {src}" if src else f"- {title}")

    body = [
        f"# {prompt}",
        "",
        "## Executive Summary",
        summary.strip(),
        "",
        "## Synthesis",
        "Below is a synthesis derived from top-ranked retrieved materials:",
        "",
    ]

    for i, d in enumerate(docs[:6], start=1):
        text = d.get("page_content") or d.get("content") or ""
        meta = d.get("metadata") or {}
        tag = meta.get("title") or meta.get("filename") or f"Doc {i}"
        if text:
            excerpt = "\n".join(text.strip().split("\n")[:6])
            body += [f"### {tag}", excerpt, ""]

    body += ["## Sources", *bullet_sources, ""]
    return "\n".join(body)

# ----------------------------
# Main RAG entry
# ----------------------------
def run_rag(
    prompt: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Dict[str, Any]:
    # Ensure the FAISS directory exists
    path_obj = Path(FAISS_PATH)
    if not (path_obj / "index.faiss").exists() or not (path_obj / "index.pkl").exists():
        raise FileNotFoundError(f"Missing FAISS index files inside {FAISS_PATH}")

    # Initialize Storage from the FAISS index folder
    storage = Storage(path=FAISS_PATH, from_path=True)

    # Query
    t_range = parse_range(start_date, end_date)
    result = storage.rag(prompt=prompt, time_range=t_range)

    docs = result.get("documents", [])
    summary = (result.get("summary") or "").strip() or "No summary returned."
    article = stitch_article(prompt, docs, summary)

    return build_feed_item(
        title=prompt,
        summary=summary,
        article=article,
        docs=docs,
    )

if __name__ == "__main__":
    prompt = "State of AI policy proposals in the last 90 days"
    item = run_rag(prompt, start_date="2025-08-01")
    print("=== FEED ITEM ===")
    print("Title:", item["title"])
    print("Summary:\n", item["summary"][:600], "...\n")
    print("Article (first 800 chars):\n", item["content"][:800], "...\n")
    print("Sources:", item["sources"])