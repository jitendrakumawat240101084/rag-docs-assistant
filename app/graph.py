"""Self-corrective RAG workflow built with LangGraph.

    START -> analyze_query -> retrieve -> grade_documents --(conditional)--+
                                 ^                                         |
                                 |                    relevant found --> generate -> END
                                 |                    none, retries left -> rewrite_query
                                 +----------------------------------------+
                                                      none, no retries --> fallback -> END
"""
import operator
from functools import lru_cache
from typing import Annotated, Literal, TypedDict

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from app import config

# ----------------------------------------------------------------------------
# 1. STATE: the data that flows between nodes
# ----------------------------------------------------------------------------
class GraphState(TypedDict, total=False):
    question: str                       # original user question (never changes)
    rewritten_query: str                # the query actually sent to the vector store
    query_type: str                     # conceptual | how-to | troubleshooting | api-reference
    documents: list[Document]           # raw retrieved chunks
    relevant_documents: list[Document]  # chunks that passed grading
    retries: int                        # how many times we rewrote + re-retrieved
    answer: str
    sources: list[dict]
    # `operator.add` tells LangGraph to APPEND to this list on every update
    # instead of overwriting it. We use it to record the path taken.
    trace: Annotated[list[str], operator.add]


# ----------------------------------------------------------------------------
# Structured outputs (so the LLM must answer in a fixed, validated format)
# ----------------------------------------------------------------------------
class QueryAnalysis(BaseModel):
    query_type: Literal["conceptual", "how-to", "troubleshooting", "api-reference"] = Field(
        description="The kind of question being asked."
    )
    rewritten_query: str = Field(
        description="A search-optimised version of the question with key technical terms and synonyms."
    )


class Grade(BaseModel):
    score: Literal["relevant", "irrelevant"] = Field(
        description="Is the document useful for answering the question?"
    )


# ----------------------------------------------------------------------------
# Prompts
# ----------------------------------------------------------------------------
ANALYZE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You prepare user questions for a vector search over technical documentation. "
     "Classify the question and rewrite it as a clear search query: expand abbreviations, "
     "add relevant technical terms or synonyms, and remove filler words. "
     "Do not answer the question."),
    ("human", "Question: {question}"),
])

GRADE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You grade whether a retrieved documentation chunk is relevant to a user question. "
     "Answer 'relevant' if the chunk contains information that helps answer the question "
     "(keywords or meaning match). Answer 'irrelevant' otherwise. Be strict: a chunk that is "
     "only loosely on the same general topic is irrelevant."),
    ("human", "Question: {question}\n\nDocument chunk:\n{document}"),
])

REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "A vector search for the query below returned nothing useful. Write ONE new search query "
     "that approaches the question from a different angle (different wording, more general or more "
     "specific terms). Output only the query text."),
    ("human", "Original question: {question}\nPrevious query: {previous_query}"),
])

GENERATE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a technical documentation assistant. Answer the question using ONLY the numbered "
     "context chunks below. Rules:\n"
     "- Cite the chunks you use with their numbers in square brackets, like [1] or [2][3].\n"
     "- If the context only partly answers the question, answer the part you can and say what is missing.\n"
     "- Never invent facts that are not in the context.\n"
     "- Keep code samples in fenced code blocks. Be concise and clear."),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])


# ----------------------------------------------------------------------------
# Dependencies (created lazily so the module imports without API keys)
# ----------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_llm():
    from langchain_groq import ChatGroq

    return ChatGroq(model=config.LLM_MODEL, temperature=0, api_key=config.GROQ_API_KEY)


def get_store():
    from app.vectorstore import get_vectorstore

    return get_vectorstore()


# ----------------------------------------------------------------------------
# 2. NODES: each takes the state and returns ONLY the keys it updates
# ----------------------------------------------------------------------------
def analyze_query(state: GraphState) -> dict:
    chain = ANALYZE_PROMPT | get_llm().with_structured_output(QueryAnalysis)
    result = chain.invoke({"question": state["question"]})
    return {
        "rewritten_query": result.rewritten_query,
        "query_type": result.query_type,
        "retries": 0,
        "trace": [f"analyze_query -> type={result.query_type}, query='{result.rewritten_query}'"],
    }


def retrieve(state: GraphState) -> dict:
    docs = get_store().similarity_search(state["rewritten_query"], k=config.TOP_K)
    return {"documents": docs, "trace": [f"retrieve -> {len(docs)} chunks"]}


def grade_documents(state: GraphState) -> dict:
    docs = state["documents"]
    if not docs:
        return {"relevant_documents": [], "trace": ["grade_documents -> 0/0 relevant"]}
    chain = GRADE_PROMPT | get_llm().with_structured_output(Grade)
    grades = chain.batch(
        [{"question": state["question"], "document": d.page_content} for d in docs],
        config={"max_concurrency": 5},
    )
    # A failed/empty grade counts as irrelevant (safe default).
    relevant = [d for d, g in zip(docs, grades) if g is not None and g.score == "relevant"]
    return {
        "relevant_documents": relevant,
        "trace": [f"grade_documents -> {len(relevant)}/{len(docs)} relevant"],
    }


def rewrite_query(state: GraphState) -> dict:
    chain = REWRITE_PROMPT | get_llm()
    new_query = chain.invoke(
        {"question": state["question"], "previous_query": state["rewritten_query"]}
    ).content.strip()
    retries = state.get("retries", 0) + 1
    return {
        "rewritten_query": new_query,
        "retries": retries,
        "trace": [f"rewrite_query (retry {retries}) -> '{new_query}'"],
    }


def generate(state: GraphState) -> dict:
    docs = state["relevant_documents"]
    context = "\n\n".join(
        f"[{i}] (source: {d.metadata['source']} | section: {d.metadata['section']})\n{d.page_content}"
        for i, d in enumerate(docs, start=1)
    )
    answer = (GENERATE_PROMPT | get_llm()).invoke(
        {"context": context, "question": state["question"]}
    ).content
    sources = [
        {
            "id": i,
            "source": d.metadata["source"],
            "section": d.metadata["section"],
            "snippet": d.page_content[:200].replace("\n", " "),
        }
        for i, d in enumerate(docs, start=1)
    ]
    return {"answer": answer, "sources": sources, "trace": ["generate -> answer written"]}


def fallback(state: GraphState) -> dict:
    return {
        "answer": (
            "I don't know. I could not find relevant information about this in the indexed "
            "documentation, even after rewriting the search query. Try rephrasing the question "
            "or ingest more documents."
        ),
        "sources": [],
        "relevant_documents": [],
        "trace": ["fallback -> no relevant docs after retries"],
    }


# ----------------------------------------------------------------------------
# 3. CONDITIONAL EDGE: the routing decision after grading
# ----------------------------------------------------------------------------
def route_after_grading(state: GraphState) -> Literal["generate", "rewrite_query", "fallback"]:
    if state["relevant_documents"]:
        return "generate"
    if state.get("retries", 0) < config.MAX_RETRIES:
        return "rewrite_query"
    return "fallback"


# ----------------------------------------------------------------------------
# 4. GRAPH ASSEMBLY
# ----------------------------------------------------------------------------
def build_graph():
    g = StateGraph(GraphState)
    g.add_node("analyze_query", analyze_query)
    g.add_node("retrieve", retrieve)
    g.add_node("grade_documents", grade_documents)
    g.add_node("rewrite_query", rewrite_query)
    g.add_node("generate", generate)
    g.add_node("fallback", fallback)

    g.add_edge(START, "analyze_query")
    g.add_edge("analyze_query", "retrieve")
    g.add_edge("retrieve", "grade_documents")
    g.add_conditional_edges(
        "grade_documents",
        route_after_grading,
        {"generate": "generate", "rewrite_query": "rewrite_query", "fallback": "fallback"},
    )
    g.add_edge("rewrite_query", "retrieve")  # the self-correcting loop
    g.add_edge("generate", END)
    g.add_edge("fallback", END)
    return g.compile()


@lru_cache(maxsize=1)
def get_graph():
    return build_graph()


def run_query(question: str) -> GraphState:
    return get_graph().invoke({"question": question, "trace": []})
