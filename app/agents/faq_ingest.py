"""FAQ ingestion into Pinecone.

Run after `python -m scripts.normalize_chunks` produces docs/faq_chunks.json:

    python -m app.agents.faq_ingest                # diff + upsert changed
    python -m app.agents.faq_ingest --dry-run      # show diff, no API calls
    python -m app.agents.faq_ingest --rebuild      # delete index contents and upsert all

What it does:
  1. Connects to Pinecone with PINECONE_API_KEY.
  2. Creates the `wasa-faq` index (1536-dim, cosine) if it doesn't exist.
  3. Loads canonical chunks from docs/faq_chunks.json (built by the pipeline).
  4. Fetches existing vector IDs + their stored `checksum` metadata.
  5. Diff: identifies new vs changed vs unchanged chunks by checksum.
  6. Embeds ONLY new + changed chunks with text-embedding-3-small.
  7. Upserts the new embeddings into Pinecone.

Idempotent: re-running with the same docs/faq_chunks.json embeds nothing and
prints "0 new, 0 changed, N unchanged".
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

INDEX_NAME = os.getenv("PINECONE_INDEX", "wasa-faq")
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
CHUNKS_PATH = Path("docs/faq_chunks.json")
STATE_PATH = Path("docs/_faq_ingest_state.json")

UPSERT_BATCH = 100


def _ensure_index(pc: Pinecone) -> None:
    existing = {idx["name"] for idx in pc.list_indexes()}
    if INDEX_NAME in existing:
        print(f"Index '{INDEX_NAME}' already exists — reusing.")
        return

    print(
        f"Creating index '{INDEX_NAME}' "
        f"(dim={EMBEDDING_DIM}, metric=cosine, cloud={PINECONE_CLOUD}, region={PINECONE_REGION})..."
    )
    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBEDDING_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
    )
    while not pc.describe_index(INDEX_NAME).status["ready"]:
        print("  ...waiting for index to become ready")
        time.sleep(2)
    print(f"Index '{INDEX_NAME}' is ready.")


def _load_chunks(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"Canonical chunk file not found at {path}. "
            f"Run `python -m scripts.normalize_chunks` first."
        )
    chunks = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(chunks, list) or not chunks:
        raise ValueError(f"{path} did not contain a non-empty JSON array.")
    required = {"id", "text", "checksum", "topic", "source"}
    missing = required - set(chunks[0].keys())
    if missing:
        raise ValueError(f"chunk records missing required fields: {missing}")
    return chunks


def _load_local_state(path: Path) -> dict[str, str]:
    """Load {id: checksum} from the local cache written after the last upsert.

    Pinecone serverless fetch() can be extremely slow on the free tier (multi
    minutes for 100 IDs), so we keep a local snapshot of what we last upserted
    and use that as the source of truth for the diff.
    """
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
    except Exception as exc:  # noqa: BLE001
        print(f"  WARNING: failed to read {path}: {exc}")
    return {}


def _save_local_state(path: Path, chunks: list[dict]) -> None:
    state = {c["id"]: c["checksum"] for c in chunks}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def _fetch_existing_checksums(index, ids: list[str]) -> dict[str, str]:
    """Verify-mode only: pull {id: checksum} directly from Pinecone.

    Use --verify to force this slow path. The default diff path uses the local
    cache from `docs/_faq_ingest_state.json`.
    """
    if not ids:
        return {}

    checksums: dict[str, str] = {}
    for i in range(0, len(ids), UPSERT_BATCH):
        batch_ids = ids[i:i + UPSERT_BATCH]
        try:
            resp = index.fetch(ids=batch_ids)
        except Exception as exc:  # noqa: BLE001
            print(f"  fetch() failed for batch starting at {i}: {exc}")
            continue
        vectors = getattr(resp, "vectors", None)
        if vectors is None and isinstance(resp, dict):
            vectors = resp.get("vectors")
        if not vectors:
            continue
        for vid, vec in vectors.items():
            md = getattr(vec, "metadata", None)
            if md is None and isinstance(vec, dict):
                md = vec.get("metadata")
            cs = md.get("checksum") if isinstance(md, dict) else None
            if cs:
                checksums[vid] = cs
    return checksums


def _diff(chunks: list[dict], existing: dict[str, str]) -> tuple[list[dict], list[dict], list[dict]]:
    """Classify each canonical chunk as new / changed / unchanged.

    Note: stale-vector cleanup is intentionally NOT done here. Pinecone serverless
    has no cheap way to list every vector ID. Use --rebuild for a clean slate.
    """
    new_chunks: list[dict] = []
    changed_chunks: list[dict] = []
    unchanged_chunks: list[dict] = []

    for c in chunks:
        prev = existing.get(c["id"])
        if prev is None:
            new_chunks.append(c)
        elif prev != c["checksum"]:
            changed_chunks.append(c)
        else:
            unchanged_chunks.append(c)

    return new_chunks, changed_chunks, unchanged_chunks


def _embed_batch(openai_client: OpenAI, texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts in one API call (cheaper than per-text)."""
    resp = openai_client.embeddings.create(input=texts, model=EMBEDDING_MODEL)
    return [d.embedding for d in resp.data]


def _to_vector(chunk: dict, embedding: list[float]) -> dict:
    """Convert a canonical chunk + embedding into a Pinecone upsert record."""
    md = {
        "text": chunk["text"],
        "topic": chunk["topic"],
        "source": chunk.get("source", ""),
        "source_page": chunk.get("source_page") or 0,
        "section": chunk.get("section", ""),
        "question": chunk.get("question", ""),
        "version": chunk.get("version", ""),
        "checksum": chunk["checksum"],
    }
    return {"id": chunk["id"], "values": embedding, "metadata": md}


def ingest(dry_run: bool = False, rebuild: bool = False, verify: bool = False) -> None:
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not pinecone_api_key or not openai_api_key:
        raise RuntimeError(
            "PINECONE_API_KEY and OPENAI_API_KEY must both be set in the environment."
        )

    chunks = _load_chunks(CHUNKS_PATH)
    print(f"Loaded {len(chunks)} canonical chunks from {CHUNKS_PATH}")

    pc = Pinecone(api_key=pinecone_api_key)
    openai_client = OpenAI(api_key=openai_api_key)

    _ensure_index(pc)
    index = pc.Index(INDEX_NAME)

    if rebuild:
        print("--rebuild: deleting all vectors in index first.")
        if not dry_run:
            try:
                index.delete(delete_all=True)
            except Exception as exc:  # noqa: BLE001
                print(f"  delete_all failed (continuing): {exc}")
            if STATE_PATH.exists():
                STATE_PATH.unlink()
        existing: dict[str, str] = {}
    elif verify:
        print(f"--verify: fetching checksums for {len(chunks)} IDs from Pinecone (slow)...")
        existing = _fetch_existing_checksums(index, [c["id"] for c in chunks])
        print(f"  {len(existing)} of {len(chunks)} already in index.")
    else:
        existing = _load_local_state(STATE_PATH)
        if existing:
            print(f"Loaded local checksum cache from {STATE_PATH} ({len(existing)} entries).")
        else:
            print(f"No local cache found. Treating index as empty (use --verify to confirm).")

    new_chunks, changed_chunks, unchanged_chunks = _diff(chunks, existing)
    print(f"\nDiff: {len(new_chunks)} new, {len(changed_chunks)} changed, "
          f"{len(unchanged_chunks)} unchanged")

    to_upsert = new_chunks + changed_chunks
    if dry_run:
        print("\n--dry-run: skipping embedding and upsert.")
        if to_upsert[:5]:
            print("Would upsert (first 5):")
            for c in to_upsert[:5]:
                print(f"  - {c['id']}  topic={c['topic']}  Q={c['question'][:60]}")
        return

    if not to_upsert:
        print("Nothing to embed. Index is up to date.")
    else:
        print(f"\nEmbedding {len(to_upsert)} chunks in batches of {UPSERT_BATCH}...")
        records: list[dict] = []
        for i in range(0, len(to_upsert), UPSERT_BATCH):
            batch = to_upsert[i:i + UPSERT_BATCH]
            texts = [c["text"] for c in batch]
            embeddings = _embed_batch(openai_client, texts)
            for c, emb in zip(batch, embeddings):
                records.append(_to_vector(c, emb))
            print(f"  embedded {min(i + UPSERT_BATCH, len(to_upsert))}/{len(to_upsert)}")

        print(f"\nUpserting {len(records)} vectors into '{INDEX_NAME}'...")
        for i in range(0, len(records), UPSERT_BATCH):
            index.upsert(vectors=records[i:i + UPSERT_BATCH])
            print(f"  upserted {min(i + UPSERT_BATCH, len(records))}/{len(records)}")

    _save_local_state(STATE_PATH, chunks)
    print(f"Saved checksum cache → {STATE_PATH}")

    time.sleep(3)
    try:
        stats = index.describe_index_stats()
        total = stats.get("total_vector_count") if isinstance(stats, dict) else getattr(stats, "total_vector_count", None)
        print(f"\nDone. Total vectors in '{INDEX_NAME}': {total}")
    except Exception as exc:  # noqa: BLE001
        print(f"\nDone. (describe_index_stats failed: {exc})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Idempotent FAQ ingestion into Pinecone.")
    parser.add_argument("--dry-run", action="store_true", help="Print the diff without embedding or upserting.")
    parser.add_argument("--rebuild", action="store_true", help="Delete all vectors first, then upsert everything.")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Fetch checksums directly from Pinecone (slow on free tier). Use to reconcile if the local cache drifts.",
    )
    args = parser.parse_args()
    ingest(dry_run=args.dry_run, rebuild=args.rebuild, verify=args.verify)


if __name__ == "__main__":
    main()
