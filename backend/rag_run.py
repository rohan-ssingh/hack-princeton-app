# rag_run.py
from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

# Load backend/.env if present
load_dotenv()

# Your Storage comes from load.py (unchanged)
from load import Storage


# ----------------------------
# Config & constants
# ----------------------------
FAISS_PATH = "faiss_index"

# # Try multiple keys for titles/urls/dates to be robust across corpora
META_TITLE_KEYS = ["title", "doc_title", "filename", "name", "heading", "document_title"]
# put near the top of rag_run.py

META_URL_KEYS   = [
    "source_url", "url", "document_url", "page_url", "permalink", "href", "link",
    "web_url", "bill_url", "pdf_url", "html_url", "source"
]
META_DATE_KEYS  = ["journal_date", "date", "published", "pub_date", "created", "updated_at"]
# we’ll custom-build title from file_name / bill_number / chamber / act_summary
# but keep a few generic fallbacks:
META_TITLE_FALLBACK_KEYS = ["title", "doc_title", "filename", "name", "heading", "document_title", "id", "doc_id", "slug"]

def _first_meta(meta: dict, keys: list[str]) -> str | None:
    for k in keys:
        v = meta.get(k)
        if v:
            return str(v)
    return None

def _format_date(d: str | None) -> str | None:
    if not d: return None
    from datetime import datetime
    try:
        return datetime.fromisoformat(d[:10]).date().isoformat()
    except Exception:
        return d

def _slug_to_title(s: str) -> str:
    """Turn 'VTHouseEnergyDigitalInfra_2025-05-15_09-03.html' into 'VTHouseEnergyDigitalInfra (2025-05-15 09:03)'."""
    import os, re
    base = os.path.basename(s)
    base = re.sub(r"\.[A-Za-z0-9]+$", "", base)          # drop extension
    pretty = base.replace("_", " ").strip()
    pretty = re.sub(r"(\d{4}-\d{2}-\d{2})[_-](\d{2})[-:](\d{2})", r"\1 \2:\3", pretty)
    return pretty if pretty else "Untitled"

def _compose_title(meta: dict, i: int) -> str:
    """
    Make a human-readable title:
      1) file_name (slug → title)
      2) chamber + bill_number
      3) act_summary (shortened)
      4) fallbacks in META_TITLE_FALLBACK_KEYS
      5) 'Doc i'
    """
    fn = meta.get("file_name")
    if fn:
        return _slug_to_title(str(fn))

    chamber = meta.get("chamber")
    bill    = meta.get("bill_number")
    if chamber or bill:
        parts = [p for p in [chamber, bill] if p]
        return " — ".join(parts)

    summary = meta.get("act_summary")
    if summary:
        s = str(summary).strip()
        return s if len(s) <= 80 else s[:77] + "…"

    fb = _first_meta(meta, META_TITLE_FALLBACK_KEYS)
    if fb:
        return fb

    return f"Doc {i}"

def _file_uri_if_path(s: str) -> str:
    """Convert existing local paths to file:// URIs; leave non-existing strings untouched."""
    try:
        from pathlib import Path
        p = Path(s)
        if p.exists():
            return p.resolve().as_uri()
        return s
    except Exception:
        return s

def _pick_top_docs_for_citations(docs, k: int = 10):
    return list(docs[:k])

def _build_cited_sources(docs_top: list) -> list[dict]:
    out = []
    for i, d in enumerate(docs_top, start=1):
        meta = getattr(d, "metadata", {}) or {}

        # URL: prefer source_url (your corpus), then other URL-like keys, then path-like fallback
        url = _first_meta(meta, META_URL_KEYS)
        if not url:
            # try path-ish keys and turn them into file:// if present
            path_like = _first_meta(meta, ["source", "file_path", "filepath", "path", "pdf_path"])
            if path_like:
                url = _file_uri_if_path(path_like)

        # Title: prefer smart composition using your fields
        title = _compose_title(meta, i)

        # Date
        date = _format_date(_first_meta(meta, META_DATE_KEYS))
        label = f"{title} ({date})" if date else title

        out.append({"S": f"S{i}", "title": label, "url": url or ""})
    return out

# ----------------------------
# Helpers
# ----------------------------
def _ensure_index(path: str) -> None:
    """Verify the FAISS folder and required files exist."""
    p = Path(path)
    if not p.exists() or not p.is_dir():
        raise FileNotFoundError(f"FAISS_INDEX_PATH not a directory: {path}")
    missing = [name for name in ("index.faiss", "index.pkl") if not (p / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing files in {path}: {missing}")


def _first_meta(meta: Dict[str, Any], keys: List[str]) -> Optional[str]:
    for k in keys:
        v = meta.get(k)
        if v is not None and v != "":
            return str(v)
    return None


def _format_date(d: Optional[str]) -> Optional[str]:
    if not d:
        return None
    # accept ISO dates or leave as-is if not parseable
    try:
        return datetime.fromisoformat(d[:10]).date().isoformat()
    except Exception:
        return d


def _build_sources(docs: List[Any], max_sources: int = 12) -> List[Dict[str, str]]:
    """Build a deduped, capped list of sources with best-effort title/date/url."""
    seen = set()
    out: List[Dict[str, str]] = []
    for i, d in enumerate(docs, start=1):
        meta = getattr(d, "metadata", {}) or {}
        url = _first_meta(meta, META_URL_KEYS) or ""
        title = _first_meta(meta, META_TITLE_KEYS) or f"Doc {i}"
        date = _format_date(_first_meta(meta, META_DATE_KEYS))
        key = (title.strip(), url.strip())
        if key in seen:
            continue
        seen.add(key)
        label = f"{title} ({date})" if date else title
        out.append({"title": label, "url": url})
        if len(out) >= max_sources:
            break
    return out


def _make_summary(text: str, max_chars: int = 400) -> str:
    """Take the first few sentences, capped to max_chars."""
    if not text:
        return "No summary returned."
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    s = " ".join(parts[:3]) or text.strip()
    if len(s) > max_chars:
        s = s[:max_chars].rstrip() + "..."
    return s


def _stitch(prompt: str, docs: List[Dict[str, Any]], summary: str) -> str:
    """Fallback article builder (no external LLM)."""
    bullets = []
    for i, d in enumerate(docs[:8], start=1):
        meta = getattr(d, "metadata", {}) or {}
        src = _first_meta(meta, META_URL_KEYS) or ""
        title = _first_meta(meta, META_TITLE_KEYS) or f"Doc {i}"
        bullets.append(f"- {title}: {src}" if src else f"- {title}")

    body = [
        f"# {prompt}",
        "",
        "## Executive Summary",
        summary or "No summary returned.",
        "",
        "## Synthesis",
        "Below is a synthesis derived from top-ranked retrieved materials:",
        "",
    ]
    for i, d in enumerate(docs[:6], start=1):
        text = (getattr(d, "page_content", None) or getattr(d, "content", None) or "").strip()
        meta = getattr(d, "metadata", {}) or {}
        tag = _first_meta(meta, META_TITLE_KEYS) or f"Doc {i}"
        if text:
            excerpt = "\n".join(text.split("\n")[:6])
            body += [f"### {tag}", excerpt, ""]
    body += ["## Sources", *bullets, ""]
    return "\n".join(body)


def _article_with_citations(prompt, summary, docs_top, cited_sources, model_hint=None):
    import os
    from textwrap import dedent
    try:
        from openai import OpenAI
    except Exception:
        return None
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return None

    # compact snippets from the same docs used for S1..Sk
    snippets = []
    for i, d in enumerate(docs_top, start=1):
        meta = getattr(d, "metadata", {}) or {}
        title = _first_meta(meta, META_TITLE_KEYS) or cited_sources[i-1]["title"]
        url = cited_sources[i-1]["url"]
        text = (getattr(d, "page_content", None) or getattr(d, "content", None) or "")[:1500]
        snippets.append({"sref": f"S{i}", "title": title, "url": url, "snippet": text})

    system = dedent("""
      You write concise, well-structured briefings with short headings.
      Use ONLY the provided summary and snippets. Insert inline citations like [S1], [S2] when you draw from snippets.
      If information is not present, say so. End with a 3–5 bullet "Key Takeaways".
    """).strip()

    user = {
        "task": prompt,
        "retrieval_summary": summary,
        "snippets": snippets,
        "sources": cited_sources,  # includes S labels
    }

    try:
        client = OpenAI(api_key=key)
        model = model_hint or os.getenv("RAG_MODEL") or "gpt-4o-mini"
        res = client.chat.completions.create(
            model=model,
            temperature=0.3,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": str(user)},
            ],
        )
        content = res.choices[0].message.content.strip()
    except Exception:
        return None

    # Append explicit References section so [S1] etc. are visible to readers
    refs_lines = ["", "## References"]
    for s in cited_sources:
        line = f"- [{s['S']}] {s['title']}"
        if s["url"]:
            line += f" — {s['url']}"
        refs_lines.append(line)
    return content + "\n" + "\n".join(refs_lines) + "\n"


# ----------------------------
# Core RAG
# ----------------------------
def run_rag(
    prompt: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    use_llm: bool = True,
    model_hint: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Runs retrieval-augmented generation using Storage.rag(question=..., date_range=[start,end])
    and returns a feed item:
        { "title", "summary", "content", "sources" }
    where "sources" is an S-mapped list:
        [{ "S": "S1", "title": "...", "url": "..." }, ...]
    """
    # 1) Ensure index + create Storage
    _ensure_index(FAISS_PATH)
    storage = Storage(path=FAISS_PATH, from_path=True)

    # 2) Query with optional date range; retry without it if it wipes results
    date_range = [start_date or "", end_date or ""] if (start_date or end_date) else None
    result = storage.rag(question=prompt, schema=None, date_range=date_range)
    docs = result.get("documents", [])

    #
    # # --- TEMP DEBUG: show what metadata keys exist in your top docs ---
    # try:
    #     print("\n[DEBUG] Top doc metadata keys & example values:")
    #     for i, d in enumerate(docs[:5], start=1):
    #         meta = getattr(d, "metadata", {}) or {}
    #         print(f"Doc {i} keys:", sorted(list(meta.keys())))
    #         # show a couple representative fields if present
    #         for k in (
    #         "title", "filename", "name", "source", "url", "link", "href", "page_url", "file_path", "source_url",
    #         "document_url"):
    #             if k in meta:
    #                 print(f"  {k} -> {meta[k]}")
    #     print()
    # except Exception as _e:
    #     pass
    # # --- END TEMP DEBUG ---


    resp = result.get("response", "")

    if date_range and not docs:
        # Fallback — run again with no date filter if nothing found
        result = storage.rag(question=prompt, schema=None, date_range=None)
        docs = result.get("documents", [])
        resp = result.get("response", "")

    # 3) Normalize response text to build a summary
    if hasattr(resp, "model_dump_json"):           # Pydantic v2 models
        resp_text = resp.model_dump_json()
    elif hasattr(resp, "json"):                    # Pydantic v1 models
        try:
            resp_text = resp.json()
        except Exception:
            resp_text = str(resp)
    else:
        resp_text = str(resp)

    summary = _make_summary(resp_text)

    # 4) Build a stable S1..Sk mapping from the same top-K docs we’ll cite/snippet
    docs_top = _pick_top_docs_for_citations(docs, k=10)
    cited_sources = _build_cited_sources(docs_top)   # [{'S':'S1','title':'..','url':'..'}, ...]

    # 5) Generate article with inline citations via LLM (or stitch fallback)
    content: Optional[str] = None
    if use_llm:
        content = _article_with_citations(
            prompt=prompt,
            summary=summary,
            docs_top=docs_top,
            cited_sources=cited_sources,
            model_hint=model_hint,
        )

    if not content:
        # Non-LLM fallback: stitch + append a References section that matches S-labels
        content = _stitch(prompt, docs_top, summary)
        refs = ["", "## References"] + [
            f"- [{s['S']}] {s['title']}" + (f" — {s['url']}" if s['url'] else "")
            for s in cited_sources
        ]
        content += "\n" + "\n".join(refs) + "\n"

    # 6) Return feed item (sources already carry S-labels)
    return {
        "title": prompt,
        "summary": summary,
        "content": content,
        "sources": cited_sources,
    }


if __name__ == "__main__":
    item = run_rag("State of AI policy proposals in the last 90 days")
    print("=== FEED ITEM ===")
    print("Title:", item["title"])
    print("Summary:\n", item["summary"], "\n")
    print("Article (first 800 chars):\n", item["content"][:800], "...\n")
    print("Sources:", item["sources"])