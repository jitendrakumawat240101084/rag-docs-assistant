"""Download a small corpus (5 FastAPI tutorial pages) into data/docs/.

The raw MkDocs files contain two things that would hurt retrieval quality:
  1. Code samples are placeholders like `{* ../../docs_src/x.py hl[2] *}`.
     We replace each one with the real code in a fenced block.
  2. Headings carry anchors like `## Title { #title }`. We strip them.

Usage:  python scripts/fetch_docs.py
"""
import re
from pathlib import Path

import httpx

RAW = "https://raw.githubusercontent.com/fastapi/fastapi/master"
PAGES = [
    "tutorial/first-steps",
    "tutorial/path-params",
    "tutorial/query-params",
    "tutorial/body",
    "tutorial/dependencies/index",
]
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "docs"

INCLUDE_RE = re.compile(r"\{\*\s+(\S+?)(?:\s+[^*]*?)?\s*\*\}")
ANCHOR_RE = re.compile(r"[ \t]*\{[ \t]*#[\w\-]+[ \t]*\}[ \t]*$", re.MULTILINE)


def get(client: httpx.Client, url: str) -> str:
    r = client.get(url, timeout=30)
    r.raise_for_status()
    return r.text


def resolve_includes(client: httpx.Client, text: str) -> str:
    def repl(match: re.Match) -> str:
        rel = match.group(1).replace("../../", "")  # -> docs_src/....py
        try:
            code = get(client, f"{RAW}/{rel}").strip()
        except httpx.HTTPError:
            return ""  # skip samples we cannot fetch
        return f"```python\n{code}\n```"

    return INCLUDE_RE.sub(repl, text)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with httpx.Client(follow_redirects=True) as client:
        for page in PAGES:
            text = get(client, f"{RAW}/docs/en/docs/{page}.md")
            text = resolve_includes(client, text)
            text = ANCHOR_RE.sub("", text)
            name = page.replace("/", "_") + ".md"
            (OUT_DIR / name).write_text(text, encoding="utf-8")
            print(f"saved {name} ({len(text)} chars)")


if __name__ == "__main__":
    main()
