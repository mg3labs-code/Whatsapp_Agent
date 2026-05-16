"""Score distribution for FAQ Pinecone retrieval — pick a similarity threshold.

Runs basic → complex buyer queries (plus a few that should *not* match well),
records top-1..top-k cosine scores, and prints percentiles + a suggested band.
Production FAQ agent uses FAQ_PINECONE_MIN_SCORE (default **0.41**, band **0.40–0.42**);
align this script after re-ingesting.

Requires PINECONE_API_KEY, OPENAI_API_KEY, populated index (same as probe_faq).

Usage:
    cd wasa
    python -m scripts.analyze_faq_thresholds
    python -m scripts.analyze_faq_thresholds --top-k 5
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

load_dotenv()

INDEX_NAME = os.getenv("PINECONE_INDEX", "wasa-faq")
EMBEDDING_MODEL = "text-embedding-3-small"


@dataclass(frozen=True)
class QueryCase:
    text: str
    tier: str  # basic | paraphrase | complex | edge | should_escalate
    note: str
    should_escalate: bool


# Ordered: easy → harder wording → multi-part → edge → off-topic / unsafe.
ANALYSIS_QUERIES: list[QueryCase] = [
    QueryCase("do you ship to nigeria", "basic", "country shipping", False),
    QueryCase("Nigeria export possible?", "paraphrase", "country shipping", False),
    QueryCase("how long does delivery to uae take", "basic", "transit", False),
    QueryCase("what is the transit time to Dubai if we order today", "complex", "transit", False),
    QueryCase("can i pay with letter of credit", "basic", "payment", False),
    QueryCase("LC payment terms for international buyers", "paraphrase", "payment", False),
    QueryCase("what payment methods do you accept", "basic", "payment", False),
    QueryCase("are you a licensed pharmaceutical company", "basic", "regulatory", False),
    QueryCase("WHO GMP certification status", "paraphrase", "quality", False),
    QueryCase("what is your who-gmp status", "basic", "quality", False),
    QueryCase("what is newlife medicare's head office address", "basic", "company", False),
    QueryCase("where is your registered office located", "paraphrase", "company", False),
    QueryCase("how much do you charge for a sample", "basic", "samples", False),
    QueryCase("what happens if my package gets stuck in customs", "basic", "customs", False),
    QueryCase("customs held our shipment at the border what is your policy", "complex", "customs", False),
    QueryCase("do i need a prescription", "basic", "prescription", False),
    QueryCase("do you supply antibiotics", "basic", "products", False),
    QueryCase("the medicine i received is damaged what do i do", "basic", "returns", False),
    QueryCase("can the ai guarantee delivery", "basic", "ai_rules", False),
    QueryCase("documentation needed for first order from Ghana", "complex", "documentation", False),
    QueryCase("bulk discount for recurring paracetamol orders", "complex", "pricing", False),
    QueryCase("do you sell tramadol", "edge", "restricted / expect low or escalate", True),
    QueryCase("Schedule H narcotics wholesale price list", "edge", "restricted", True),
    QueryCase("what is the best smartphone to buy", "should_escalate", "off-topic", True),
    QueryCase("write me a python quicksort implementation", "should_escalate", "off-topic", True),
    QueryCase("guarantee my order will clear customs in 48 hours no delays", "complex", "may be weak", False),
]


def percentile_nearest_rank(values: list[float], p: float) -> float | None:
    """Return the p-th percentile (0–100) using nearest-rank on a sorted copy."""
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    # Nearest-rank: position = ceil(p/100 * n)
    pos = max(1, min(len(xs), math.ceil(p / 100.0 * len(xs))))
    return xs[pos - 1]


def summarize_threshold_band(
    match_top1: list[float],
    escalate_top1: list[float],
    margin: float = 0.02,
) -> dict[str, float | str | None]:
    """Suggest a floor from score separation (heuristic, not a guarantee).

    - ``fewer_escalations``: use a low floor near match distribution bottom.
    - ``stricter_grounding``: keep floor above noisy off-topic / escalate scores.
    """
    if not match_top1:
        return {"error": "no match-labelled queries scored"}

    esc = escalate_top1 or [0.0]
    escalate_max = max(esc)
    match_min = min(match_top1)
    match_p10 = percentile_nearest_rank(match_top1, 10.0)
    match_p25 = percentile_nearest_rank(match_top1, 25.0)
    match_median = percentile_nearest_rank(match_top1, 50.0)

    # "Safe" floor: above worst escalator with a small margin (when separation exists).
    floor_from_escalation = escalate_max + margin

    # Fewer escalations: align with lower tail of legitimate queries.
    floor_from_recall = match_p10 if match_p10 is not None else match_min

    # If these disagree, the band is narrow or overlapping.
    low = max(floor_from_escalation, 0.0)
    high = min(match_p25 or match_median or match_min, 1.0)

    band_note: str
    if low <= high:
        band_note = f"overlap OK: try floor in [{low:.3f}, {high:.3f}] (tune inside for recall vs safety)"
    else:
        band_note = (
            f"ambiguous: escalate_max={escalate_max:.3f} vs match_p10={match_p10:.3f} — "
            "labels overlap; lower floor for recall OR improve chunks / add reranking"
        )

    return {
        "escalate_max": escalate_max,
        "match_min": match_min,
        "match_p10": match_p10,
        "match_p25": match_p25,
        "match_median": match_median,
        "floor_from_escalation": floor_from_escalation,
        "floor_from_recall_p10": floor_from_recall,
        "suggested_band_note": band_note,
    }


def _embed_batch(client: OpenAI, texts: list[str]) -> list[list[float]]:
    """One API round-trip for all query strings (order preserved)."""
    resp = client.embeddings.create(input=texts, model=EMBEDDING_MODEL)
    return [d.embedding for d in resp.data]


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
        out.append({"id": mid, "score": float(score) if score is not None else 0.0, "metadata": md})
    return out


def run_analysis(queries: list[QueryCase], top_k: int) -> int:
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not pinecone_api_key or not openai_api_key:
        print("ERROR: PINECONE_API_KEY and OPENAI_API_KEY must be set.", file=sys.stderr)
        return 2

    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(INDEX_NAME)
    client = OpenAI(api_key=openai_api_key)

    match_scores: list[float] = []
    escalate_scores: list[float] = []
    rows: list[tuple[QueryCase, list[float]]] = []

    vectors = _embed_batch(client, [qc.text for qc in queries])
    for qc, vec in zip(queries, vectors, strict=True):
        hits = _query(index, vec, top_k)
        scores = [h["score"] for h in hits[:top_k]] + [0.0] * max(0, top_k - len(hits))
        scores = scores[:top_k]
        top1 = scores[0] if scores else 0.0
        rows.append((qc, scores))
        if qc.should_escalate:
            escalate_scores.append(top1)
        else:
            match_scores.append(top1)

    # Per-query table
    print(f"Index={INDEX_NAME!r}  model={EMBEDDING_MODEL}  top_k={top_k}\n")
    w = max(len(qc.text) for qc in queries)
    for qc, scores in rows:
        esc = "ESC" if qc.should_escalate else "FAQ"
        score_str = "  ".join(f"{s:.3f}" for s in scores)
        print(f"{esc}  {qc.tier:<14}  {qc.text[:w]:<{w}}  | top1..k: {score_str}")
        print(f"     ({qc.note})")

    print("\n--- DISTRIBUTION (top-1 cosine) ---")
    p10 = percentile_nearest_rank(match_scores, 10.0)
    p25 = percentile_nearest_rank(match_scores, 25.0)
    p50 = percentile_nearest_rank(match_scores, 50.0)
    p10_s = f"{p10:.3f}" if p10 is not None else "n/a"
    p25_s = f"{p25:.3f}" if p25 is not None else "n/a"
    p50_s = f"{p50:.3f}" if p50 is not None else "n/a"
    print(
        f"should_match (n={len(match_scores)}): min={min(match_scores):.3f}  p10={p10_s}  "
        f"p25={p25_s}  median={p50_s}  max={max(match_scores):.3f}"
    )
    if escalate_scores:
        print(
            f"should_escalate (n={len(escalate_scores)}): min={min(escalate_scores):.3f}  "
            f"max={max(escalate_scores):.3f}  "
            f"(want agent floor > max for clean separation)"
        )

    summary = summarize_threshold_band(match_scores, escalate_scores)
    print("\n--- HEURISTIC THRESHOLD BAND ---")
    for k in (
        "escalate_max",
        "match_min",
        "match_p10",
        "match_p25",
        "match_median",
        "floor_from_escalation",
        "floor_from_recall_p10",
    ):
        v = summary.get(k)
        if isinstance(v, float):
            print(f"  {k}: {v:.3f}")
    print(f"\n{summary.get('suggested_band_note', '')}")

    # Pass/fail vs common floors
    print("\n--- ESCALATION RATE (if chunk used only when score > T; top-1) ---")
    for t in (0.35, 0.38, 0.40, 0.41, 0.42, 0.43, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70):
        miss = sum(1 for s in match_scores if s <= t)
        leak = sum(1 for s in escalate_scores if s > t) if escalate_scores else 0
        print(f"  T={t:.2f}  FAQ would escalate {miss}/{len(match_scores)}  |  "
              f"bad-query chunks passing filter: {leak}/{len(escalate_scores)}")

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Pinecone top scores for threshold tuning.")
    parser.add_argument("--top-k", type=int, default=5, help="Matches to fetch and display per query.")
    args = parser.parse_args()
    sys.exit(run_analysis(ANALYSIS_QUERIES, top_k=args.top_k))


if __name__ == "__main__":
    main()
