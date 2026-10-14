"""Embedding model + ChromaDB, created once and reused."""
from functools import lru_cache

from app import config


@lru_cache(maxsize=1)
def get_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings

    # Runs locally and is free: no API key needed for embeddings.
    return HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def get_vectorstore():
    from langchain_chroma import Chroma

    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(config.CHROMA_DIR),
    )
