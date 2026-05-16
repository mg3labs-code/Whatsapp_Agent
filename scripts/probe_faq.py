"""Retrieval sanity check for the FAQ Pinecone index.

Run AFTER `python -m app.agents.faq_ingest` has populated the index. This
script issues a fixed list of realistic buyer queries against Pinecone and
prints the top-3 hits + similarity scores per query. Use it to spot bad
chunks before wiring up the FAQ agent (Task 3.4).

Usage:
    python -m scripts.probe_faq
    python -m scripts.probe_faq --top-k 5 --threshold 0.41
    python -m scripts.probe_faq --query "do you ship to nigeria"

The default queries cover all major topics + a few that SHOULD return no
high-confidence chunk (so the agent will escalate). Align --threshold with
FAQ_PINECONE_MIN_SCORE / app.agents.faq (production band ~0.40–0.42, default
0.41 from analyze_faq_thresholds). Scores well below that on a should-match
query usually mean chunk or embedding quality issues.
"""

from __future__ import annotations

import argparse
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

load_dotenv()

INDEX_NAME = os.getenv("PINECONE_INDEX", "wasa-faq")
EMBEDDING_MODEL = "text-embedding-3-small"

# Default sanity queries. The HUMAN_LABEL is what we expect to retrieve;
# 'should_escalate' marks queries that SHOULD return low-score / no relevant chunk.
DEFAULT_QUERIES: list[tuple[str, str, bool]] = [
    ("do you ship to nigeria", "shipping/countries", False),
    ("how long does delivery to uae take", "shipping/timelines", False),
    ("can i pay with letter of credit", "payment", False),
    ("are you a licensed pharmaceutical company", "regulatory/legal", False),
    ("what is your who-gmp status", "regulatory/quality", False),
    ("do you sell tramadol", "restricted/escalation", True),
    ("what is newlife medicare's head office address", "company", False),
    ("how much do you charge for a sample", "sample", False),
    ("what happens if my package gets stuck in customs", "shipping/customs", False),
    ("do i need a prescription", "prescription", False),
    ("what payment methods do you accept", "payment", False),
    ("do you supply antibiotics", "products", False),
    ("the medicine i received is damaged what do i do", "returns", False),
    ("can the ai guarantee delivery", "ai_rules", False),
    ("what is the best smartphone to buy", "off-topic/escalation", True),
]


def _embed(client: OpenAI, text: str) -> list[float]:
    resp = client.embeddings.create(input=text, model=EMBEDDING_MODEL)
    return resp.data[0].embedding


def _query(index, vector: list[float], top_k: int) -> list[dict]:
    resp = index.query(vector=vector, top_k=top_k, include_metadata=True)
    matches = getattr(resp, "matches", None)
    if matches is None and isinstance(resp, dict):
        matches = resp.get("matches", [])
    out: list[dict] = []
    for m in matches or []:
        score = getattr(m, "score", None) or (m.get("score") if isinstance(m, dict) else None)
        mid = getattr(m, "id", None) or (m.get("id") if isinstance(m, dict) else None)
        md = getattr(m, "metadata", None) or (m.get("metadata") if isinstance(m, dict) else None) or {}
        out.append({"id": mid, "score": score, "metadata": md})
    return out


def run_probe(queries: list[tuple[str, str, bool]], top_k: int, threshold: float) -> int:
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not pinecone_api_key or not openai_api_key:
        print("ERROR: PINECONE_API_KEY and OPENAI_API_KEY must be set.", file=sys.stderr)
        return 2

    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(INDEX_NAME)
    openai_client = OpenAI(api_key=openai_api_key)

    failures = 0
    for query_text, expected_topic, should_escalate in queries:
        print(f"\n=== {query_text!r}")
        print(f"    expected: {expected_topic} | should_escalate={should_escalate}")
        vec = _embed(openai_client, query_text)
        hits = _query(index, vec, top_k)
        if not hits:
            print("  (no matches)")
            if not should_escalate:
                failures += 1
                print("  FAIL: real query returned no chunks")
            continue

        top_score = hits[0]["score"] or 0
        for i, h in enumerate(hits):
            md = h["metadata"]
            q = (md.get("question") or "")[:80]
            topic = md.get("topic", "?")
            score = h["score"]
            print(f"  [{i + 1}] {score:.3f}  topic={topic:<12} Q={q}")

        passes_threshold = top_score >= threshold
        if should_escalate:
            if passes_threshold:
                print(f"  WARN: should_escalate=True but top score {top_score:.3f} >= {threshold}")
            else:
                print(f"  OK: top score {top_score:.3f} < {threshold} → agent would escalate.")
        else:
            if not passes_threshold:
                failures += 1
                print(f"  FAIL: top score {top_score:.3f} < {threshold} for a query that should match.")
            else:
                print(f"  OK: top score {top_score:.3f} >= {threshold}.")

    print("\n--- SUMMARY ---")
    print(f"Queries: {len(queries)}  Real-query failures (low score): {failures}")
    return 0 if failures == 0 else 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.41,
        help="Pass if top-1 score >= this (match FAQ_PINECONE_MIN_SCORE; typical band 0.40–0.42).",
    )
    parser.add_argument("--query", default=None, help="Run a single ad-hoc query instead of the default set.")
    args = parser.parse_args()

    if args.query:
        queries = [(args.query, "ad-hoc", False)]
    else:
        queries = DEFAULT_QUERIES

    sys.exit(run_probe(queries, top_k=args.top_k, threshold=args.threshold))


if __name__ == "__main__":
    main()
