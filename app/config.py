"""Central settings. Everything tunable lives here (or in .env)."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DOCS_DIR = DATA_DIR / "docs"          # the corpus (markdown files)
CHROMA_DIR = DATA_DIR / "chroma"      # persisted vector store
FEEDBACK_FILE = DATA_DIR / "feedback.jsonl"

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

TOP_K = int(os.getenv("TOP_K", "5"))            # chunks retrieved per search
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))  # query rewrites before giving up

CHUNK_SIZE = 1000      # characters
CHUNK_OVERLAP = 150    # characters
COLLECTION_NAME = "tech_docs"
