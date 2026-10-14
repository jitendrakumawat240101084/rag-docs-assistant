# RAG Technical Documentation Assistant

A Retrieval-Augmented Generation (RAG) assistant that answers questions about technical documentation, built with a **self-corrective LangGraph workflow** and served through **FastAPI**.

The corpus is 5 pages of the official **FastAPI tutorial** (first steps, path parameters, query parameters, request body, dependencies). Ask it things like *"How do I declare a request body?"* and it answers with citations to the exact document and section.

## Architecture

```
START -> analyze_query -> retrieve -> grade_documents --- relevant chunks found ---> generate -> END
                             ^               |
                             |               +-- none, retries left ---> rewrite_query --+
                             +-----------------------------------------------------------+
                                             +-- none, retries used up -> fallback ("I don't know") -> END
```

| Node | What it does |
|---|---|
| `analyze_query` | LLM classifies the question (conceptual / how-to / troubleshooting / api-reference) and rewrites it into a search-friendly query. |
| `retrieve` | Similarity search in ChromaDB, returns top-k chunks with source metadata. |
| `grade_documents` | LLM grades **each** chunk as `relevant` / `irrelevant` against the original question. Irrelevant chunks are dropped. |
| `rewrite_query` | Used only when nothing relevant was found: asks the LLM for a different query and increments `retries`. |
| `generate` | LLM answers using only the relevant chunks and cites them as `[1]`, `[2]`... |
| `fallback` | After `MAX_RETRIES` failed rewrites, returns an honest "I don't know" instead of guessing. |

**Conditional edge** (`route_after_grading` in `app/graph.py`): relevant docs -> `generate`; none and `retries < MAX_RETRIES` -> `rewrite_query`; otherwise -> `fallback`.

### State schema

```python
class GraphState(TypedDict, total=False):
    question: str                       # original question, never modified
    rewritten_query: str                # query actually used for search
    query_type: str
    documents: list[Document]           # raw retrieved chunks
    relevant_documents: list[Document]  # chunks that passed grading
    retries: int                        # retry counter (drives the loop limit)
    answer: str
    sources: list[dict]
    trace: Annotated[list[str], operator.add]   # append-only log of the path taken
```

Key choices: the original `question` is kept separate from `rewritten_query`, so grading and generation always judge against what the user really asked. `retries` lives in the state, so the router can enforce the limit and the loop can never run forever. `trace` uses a reducer (`operator.add`) so each node *appends* to it; it is returned by the API for transparency and debugging.

## Project structure

```
app/
  config.py        settings (.env overrides)
  vectorstore.py   embeddings + ChromaDB
  ingestion.py     load -> chunk -> embed -> store
  graph.py         LangGraph workflow (state, nodes, routing)
  main.py          FastAPI endpoints
scripts/fetch_docs.py   downloads + cleans the corpus
data/docs/              the corpus (already included)
```

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env               # then put your GROQ_API_KEY in .env
```

Get a free API key at <https://console.groq.com>. Embeddings run locally with `sentence-transformers` (no key needed; the model downloads on first run).

## Run

```bash
# 1. (optional) re-download the corpus
python scripts/fetch_docs.py

# 2. index the documents into ChromaDB
python -m app.ingestion

# 3. start the API
uvicorn app.main:app --reload
```

Interactive docs: <http://127.0.0.1:8000/docs>

## API examples

**POST /query**
```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I declare a request body?"}'
```
```json
{
  "query_id": "9c1f...",
  "answer": "Declare a Pydantic model that inherits from `BaseModel` and use it as a parameter type in your path operation function [1][2]. ...",
  "sources": [
    {"id": 1, "source": "tutorial_body.md", "section": "Request Body > Create your data model", "snippet": "Then you declare your data model as a class that inherits from BaseModel..."}
  ],
  "query_type": "how-to",
  "rewritten_query": "declare request body Pydantic BaseModel FastAPI",
  "retries": 0,
  "trace": [
    "analyze_query -> type=how-to, query='declare request body Pydantic BaseModel FastAPI'",
    "retrieve -> 5 chunks",
    "grade_documents -> 3/5 relevant",
    "generate -> answer written"
  ]
}
```
(Output shown is illustrative; the exact wording comes from the LLM.)

**Out-of-scope question** returns the fallback after retries:
```json
{"answer": "I don't know. I could not find relevant information ...", "sources": [], "retries": 2}
```

**POST /ingest** (files and/or URLs)
```bash
curl -X POST http://127.0.0.1:8000/ingest \
  -F "files=@my_notes.md" \
  -F "urls=https://raw.githubusercontent.com/pydantic/pydantic/main/README.md"
```

**GET /documents**
```bash
curl http://127.0.0.1:8000/documents
# {"count": 5, "total_chunks": 60, "documents": [{"source": "tutorial_body.md", "chunks": 14}, ...]}
```

**POST /feedback** (use the `query_id` from a `/query` response)
```bash
curl -X POST http://127.0.0.1:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{"query_id": "9c1f...", "rating": "up", "comment": "Clear answer"}'
```

**Status codes:** `200` ok, `201` feedback saved, `400` nothing provided to ingest, `404` unknown `query_id`, `409` no documents indexed yet, `422` validation error / nothing ingested, `502` LLM provider failure, `503` API key missing.

## Design decisions and trade-offs

**Chunking.** Two-stage: first split on Markdown headings (`#`, `##`, `###`), then split any section longer than 1000 characters with a recursive splitter (paragraph -> line -> word) and 150 characters of overlap. Why: technical docs are organised by headings, so a chunk that stays inside one section stays about one topic, and every chunk carries its heading path (`Request Body > Create your data model`) as metadata, which makes citations precise. Overlap protects content that falls on a boundary. Code samples are kept inline in the text so a question about code can match them. Trade-off: character-based sizes are crude; a very long code block can still be cut in the middle.

**Embeddings.** `all-MiniLM-L6-v2` (sentence-transformers): free, runs on CPU, no API key, fast. Trade-off: weaker than large hosted embedding models on subtle technical queries; it is a one-line change in `.env` to swap.

**Vector store.** ChromaDB, persisted on disk. Simple and enough for a small corpus. Chunk IDs are `source::index` and re-ingesting a source first deletes its old chunks, so `/ingest` is idempotent (no duplicates).

**LLM.** Groq (`llama-3.3-70b-versatile`) at temperature 0: generous free tier and fast, which matters because grading makes one LLM call per chunk. Structured output (Pydantic models) forces the query analyser and the grader to return validated, fixed-format answers instead of free text.

**Grading.** One LLM call per chunk, run in parallel (max 5 at once), using the *original* question. Failed or empty grades count as "irrelevant", the safe default. Trade-off: more LLM calls = more latency and cost than grading all chunks in a single call, but per-chunk grading is simpler and more reliable.

**Self-correction.** On zero relevant chunks the system rewrites the query and retries, up to `MAX_RETRIES` (default 2). If it still finds nothing it says "I don't know" rather than hallucinate.

**Feedback storage.** Appended to `data/feedback.jsonl`, and `query_id`s are held in memory, so feedback only works for queries made in the current server session. Fine for a demo; production would use a database.

## Assumptions

- Documents are UTF-8 Markdown, text or HTML.
- English-only corpus and questions.
- A single shared index (no per-user collections) and no authentication.

## What I would improve with more time

1. **Hallucination check node** (Self-RAG style): verify the answer is supported by the chunks, and regenerate if not.
2. **Web-search fallback** (Tavily) before saying "I don't know".
3. **Conversation memory** so follow-up questions work within a session.
4. **Hybrid retrieval** (BM25 + vectors) and a **reranker**, which helps with exact identifiers like function names.
5. **Evaluation set**: a list of questions with expected sources to measure retrieval and grading quality (precision/recall) instead of eyeballing.
6. **Tests and persistence**: unit tests for the router and ingestion, a database for feedback, and request rate limiting.
7. **Streaming responses** and a small Streamlit UI.
