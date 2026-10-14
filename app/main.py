"""FastAPI service. Run with:  uvicorn app.main:app --reload"""
import json
import uuid
from datetime import datetime, timezone
from typing import Literal, Optional

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app import config
from app.ingestion import (
    SUPPORTED_EXTENSIONS,
    count_chunks,
    decode_file,
    fetch_url,
    ingest_text,
    list_documents,
)

app = FastAPI(
    title="RAG Technical Documentation Assistant",
    description="Self-corrective RAG (LangGraph) over technical documentation.",
    version="1.0.0",
)

# query_id -> question/answer, so /feedback can be tied to a real answer.
# In-memory on purpose (simple); a real system would use a database.
_QUERY_LOG: dict[str, dict] = {}


# ----------------------------- schemas ---------------------------------------
class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000, examples=["How do I declare a request body?"])


class Source(BaseModel):
    id: int
    source: str
    section: str
    snippet: str


class QueryResponse(BaseModel):
    query_id: str
    answer: str
    sources: list[Source]
    query_type: str
    rewritten_query: str
    retries: int
    trace: list[str]


class FeedbackRequest(BaseModel):
    query_id: str
    rating: Literal["up", "down"]
    comment: Optional[str] = Field(default=None, max_length=2000)


class IngestResponse(BaseModel):
    ingested: list[dict]
    errors: list[dict]


# ----------------------------- endpoints -------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "indexed_chunks": count_chunks()}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    question = req.question.strip()
    if len(question) < 3:
        raise HTTPException(status_code=422, detail="Question is too short.")
    if not config.GROQ_API_KEY:
        raise HTTPException(status_code=503, detail="GROQ_API_KEY is not set. See README.")
    if count_chunks() == 0:
        raise HTTPException(
            status_code=409,
            detail="No documents are indexed yet. Run `python -m app.ingestion` or POST /ingest.",
        )

    from app.graph import run_query  # imported here so /health works without LLM deps

    try:
        state = run_query(question)
    except Exception as exc:  # LLM / network failure
        raise HTTPException(status_code=502, detail=f"Upstream model error: {exc}") from exc

    query_id = uuid.uuid4().hex
    _QUERY_LOG[query_id] = {"question": question, "answer": state["answer"]}
    return QueryResponse(
        query_id=query_id,
        answer=state["answer"],
        sources=state.get("sources", []),
        query_type=state.get("query_type", "unknown"),
        rewritten_query=state.get("rewritten_query", question),
        retries=state.get("retries", 0),
        trace=state.get("trace", []),
    )


@app.post("/ingest", response_model=IngestResponse)
def ingest(
    files: list[UploadFile] = File(default=[]),
    urls: Optional[str] = Form(default=None, description="One URL per line or comma-separated"),
):
    url_list = [u.strip() for u in (urls or "").replace(",", "\n").splitlines() if u.strip()]
    if not files and not url_list:
        raise HTTPException(status_code=400, detail="Provide at least one file or URL.")

    ingested, errors = [], []

    for f in files:
        name = f.filename or "upload"
        suffix = "." + name.rsplit(".", 1)[-1].lower() if "." in name else ""
        if suffix not in SUPPORTED_EXTENSIONS:
            errors.append({"source": name, "error": f"unsupported type; use {sorted(SUPPORTED_EXTENSIONS)}"})
            continue
        try:
            text = decode_file(name, f.file.read())
            ingested.append({"source": name, "chunks": ingest_text(name, text)})
        except UnicodeDecodeError:
            errors.append({"source": name, "error": "file is not valid UTF-8 text"})
        except ValueError as exc:
            errors.append({"source": name, "error": str(exc)})

    for url in url_list:
        try:
            source, text = fetch_url(url)
            ingested.append({"source": source, "chunks": ingest_text(source, text)})
        except (httpx.HTTPError, ValueError) as exc:
            errors.append({"source": url, "error": str(exc)})

    if not ingested:
        raise HTTPException(status_code=422, detail={"message": "Nothing was ingested.", "errors": errors})
    return IngestResponse(ingested=ingested, errors=errors)


@app.get("/documents")
def documents():
    docs = list_documents()
    return {"count": len(docs), "total_chunks": sum(d["chunks"] for d in docs), "documents": docs}


@app.post("/feedback", status_code=201)
def feedback(req: FeedbackRequest):
    entry = _QUERY_LOG.get(req.query_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Unknown query_id (was it from this server session?).")
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "time": datetime.now(timezone.utc).isoformat(),
        "query_id": req.query_id,
        "rating": req.rating,
        "comment": req.comment,
        **entry,
    }
    with open(config.FEEDBACK_FILE, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
    return {"status": "saved"}
