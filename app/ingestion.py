"""Ingestion pipeline: load -> split -> embed -> store.

Run standalone:  python -m app.ingestion
The API's POST /ingest endpoint reuses `ingest_text`.
"""
from pathlib import Path

import httpx
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from app import config
from app.vectorstore import get_vectorstore

SUPPORTED_EXTENSIONS = {".md", ".txt", ".html", ".htm"}

# --- Chunking strategy -------------------------------------------------------
# Step 1: split on markdown headings so each chunk stays inside ONE section
#         and remembers its heading path (e.g. "Request Body > Declare it").
# Step 2: if a section is still longer than CHUNK_SIZE, split it further with a
#         recursive splitter (paragraphs first, then lines, then words).
# Overlap keeps a sentence/code line that sits on a boundary visible in both chunks.
_header_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
    strip_headers=False,
)
_size_splitter = RecursiveCharacterTextSplitter(
    chunk_size=config.CHUNK_SIZE,
    chunk_overlap=config.CHUNK_OVERLAP,
    separators=["\n\n", "\n", " ", ""],
)


def html_to_markdown(html: str) -> str:
    """Very small HTML -> text converter that keeps headings as '#' lines."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    for level in (1, 2, 3):
        for h in soup.find_all(f"h{level}"):
            h.replace_with(f"\n\n{'#' * level} {h.get_text(strip=True)}\n\n")
    return soup.get_text("\n")


def fetch_url(url: str) -> tuple[str, str]:
    """Download a URL. Returns (source_name, markdown_or_text)."""
    r = httpx.get(url, timeout=30, follow_redirects=True)
    r.raise_for_status()
    body = r.text
    if "html" in r.headers.get("content-type", "") or body.lstrip().startswith("<"):
        body = html_to_markdown(body)
    return url, body


def decode_file(name: str, raw: bytes) -> str:
    text = raw.decode("utf-8")
    if Path(name).suffix.lower() in {".html", ".htm"}:
        text = html_to_markdown(text)
    return text


def chunk_document(source: str, text: str) -> list[Document]:
    sections = _header_splitter.split_text(text)
    chunks = _size_splitter.split_documents(sections)
    for i, chunk in enumerate(chunks):
        headings = [chunk.metadata.get(k) for k in ("h1", "h2", "h3")]
        chunk.metadata = {
            "source": source,
            "section": " > ".join(h for h in headings if h) or "(top)",
            "chunk_index": i,
        }
    return chunks


def ingest_text(source: str, text: str) -> int:
    """Index one document. Re-ingesting the same source replaces the old chunks."""
    chunks = chunk_document(source, text)
    if not chunks:
        raise ValueError(f"'{source}' has no text to index")
    store = get_vectorstore()
    store._collection.delete(where={"source": source})  # avoid duplicates
    store.add_documents(chunks, ids=[f"{source}::{c.metadata['chunk_index']}" for c in chunks])
    return len(chunks)


def list_documents() -> list[dict]:
    """Summarise what is in the store: one row per source with its chunk count."""
    data = get_vectorstore()._collection.get(include=["metadatas"])
    counts: dict[str, int] = {}
    for meta in data["metadatas"] or []:
        counts[meta["source"]] = counts.get(meta["source"], 0) + 1
    return [{"source": s, "chunks": n} for s, n in sorted(counts.items())]


def count_chunks() -> int:
    return get_vectorstore()._collection.count()


def ingest_directory(directory: Path = config.DOCS_DIR) -> None:
    files = sorted(p for p in directory.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS)
    if not files:
        raise SystemExit(f"No documents found in {directory}. Run: python scripts/fetch_docs.py")
    for path in files:
        n = ingest_text(path.name, decode_file(path.name, path.read_bytes()))
        print(f"indexed {path.name}: {n} chunks")


if __name__ == "__main__":
    ingest_directory()
