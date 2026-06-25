"""Re-derive product attributes from real titles using an LLM.

Why this script exists:
The catalog   handed over has real titles/images/prices (eBay), but
the color, material, occasion, and prod_description columns were generated
by an earlier LLM run without looking at the title. As a result, a product
titled "Vintage 1998 Y2K Planet Hollywood Cut Workout Shirt White Tanktop"
might have color=Blue, material=Linen, occasion=Party — none of which
match the title.

This script reads each row, asks the LLM to extract the attributes from
the TITLE ALONE, and writes a corrected CSV. The original is preserved.

Features:
- Resumable: writes progress to a JSONL checkpoint after each batch. If
  you Ctrl-C or hit quota, re-run and it picks up from the last row done.
- Batched: processes N rows per LLM call to reduce overhead.
- Provider-agnostic: uses the same LLM client abstraction as the app.

Usage:
    python -m scripts.enhance_attributes
    python -m scripts.enhance_attributes --batch-size 10 --limit 500
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from app.config import DATA_DIR, PRODUCTS_CSV
from app.llm_client import LLMError, generate_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


CHECKPOINT_PATH = DATA_DIR / "enhancement_checkpoint.jsonl"
OUTPUT_CSV = DATA_DIR / "products_enhanced.csv"


BATCH_PROMPT = """\
You are a retail product analyst. For each numbered product TITLE below,
extract the most likely attributes from the title alone (ignore any other
context, do not invent details).

Return JSON only.

Allowed values:
- color: a single common color word (e.g. "red", "navy", "white"). Use "" if not clear.
- material: a single material if mentioned or strongly implied (e.g. "cotton", "denim", "wool", "leather"). Use "" if not clear.
- occasion: one of [casual, formal, party, wedding, sports, office wear, beach, outdoor]. Pick the best fit. Use "casual" as a default if unclear.
- description: one short sentence (max ~20 words) describing the product based on the title.

Output format:
{
  "items": [
    {"idx": 0, "color": "...", "material": "...", "occasion": "...", "description": "..."},
    ...
  ]
}

Products:
{block}
"""


def _format_batch(titles: list[str]) -> str:
    return "\n".join(f"{i}. {t}" for i, t in enumerate(titles))


def _load_checkpoint() -> dict[int, dict]:
    """Return {row_index: enhanced_attrs} from the checkpoint file."""
    if not CHECKPOINT_PATH.exists():
        return {}
    out: dict[int, dict] = {}
    with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                out[rec["row_idx"]] = rec
            except json.JSONDecodeError:
                continue
    return out


def _append_checkpoint(records: list[dict]) -> None:
    with open(CHECKPOINT_PATH, "a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def enhance(batch_size: int = 8, limit: Optional[int] = None) -> Path:
    df = pd.read_csv(PRODUCTS_CSV)
    n_total = len(df) if limit is None else min(limit, len(df))
    logger.info("Catalog has %d rows; processing %d.", len(df), n_total)

    done = _load_checkpoint()
    logger.info("Resuming with %d rows already done.", len(done))

    todo_indices = [i for i in range(n_total) if i not in done]
    logger.info("%d rows still to process.", len(todo_indices))

    for start in range(0, len(todo_indices), batch_size):
        batch_idx = todo_indices[start : start + batch_size]
        titles = [str(df.iloc[i]["Product_title"]) for i in batch_idx]
        prompt = BATCH_PROMPT.replace("{block}", _format_batch(titles))

        try:
            parsed = generate_json(prompt, temperature=0.0)
            items = parsed.get("items", [])
            if not isinstance(items, list):
                raise LLMError(f"Unexpected shape: {parsed!r}")
        except Exception as e:
            logger.warning(
                "Batch %d-%d failed (%s); skipping for now.",
                batch_idx[0],
                batch_idx[-1],
                repr(e),
            )
            continue

        records = []
        for item in items:
            local_idx = int(item.get("idx", -1))
            if 0 <= local_idx < len(batch_idx):
                row_idx = batch_idx[local_idx]
                records.append(
                    {
                        "row_idx": row_idx,
                        "Product_title": titles[local_idx],
                        "color": str(item.get("color", "")).strip(),
                        "material": str(item.get("material", "")).strip(),
                        "occasion": str(item.get("occasion", "")).strip(),
                        "prod_description": str(item.get("description", "")).strip(),
                    }
                )
        _append_checkpoint(records)
        for r in records:
            done[r["row_idx"]] = r
        logger.info("Progress: %d/%d", len(done), n_total)

    logger.info("Writing enhanced CSV.")
    for i, rec in done.items():
        if i >= len(df):
            continue
        for col in ("color", "material", "occasion", "prod_description"):
            if rec.get(col):
                df.at[i, col] = rec[col]

    df.iloc[:n_total].to_csv(OUTPUT_CSV, index=False)
    logger.info("Done. Output: %s", OUTPUT_CSV)
    return OUTPUT_CSV


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--limit", type=int, default=None, help="Process at most N rows.")
    args = p.parse_args()

    out = enhance(batch_size=args.batch_size, limit=args.limit)
    print(f"\nEnhanced catalog written to: {out}")
    print(
        "Switch the app to use it by setting in .env:\n"
        f"  PRODUCTS_CSV={out}\n"
        "Or rename it over the original to make it the default."
    )


if __name__ == "__main__":
    main()
