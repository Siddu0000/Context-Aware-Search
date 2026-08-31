"""Contrastive (MNRL) fine-tune of the embedding model on catalog-derived pairs."""

import argparse
import random
import sys

import pandas as pd
from transformers import TrainerCallback

from app.config import PRODUCTS_CSV, PROJECT_ROOT

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT_DIR = PROJECT_ROOT / "models" / "gte-small-cas-ft"


def _attr_text(row) -> str:
    """Attribute text for a product, minus the title (else the pair is identity)."""
    bits = []
    for col in ("bsns_vrtcl_name", "categ_lvl2_name", "color", "material", "occasion"):
        v = row.get(col)
        if isinstance(v, str) and v.strip():
            bits.append(v.strip())
    desc = row.get("prod_description")
    if isinstance(desc, str) and desc.strip():
        bits.append(desc.strip()[:300])
    return " ".join(bits)


def _query_phrase(row) -> str:
    """Short attribute phrase shaped like the intents the translator produces."""
    bits = []
    for col in ("color", "material", "categ_lvl2_name"):
        v = row.get(col)
        if isinstance(v, str) and v.strip():
            bits.append(v.strip().lower())
    return " ".join(bits)


def build_pairs(df: pd.DataFrame, n_pairs: int, seed: int) -> list:
    """Build (anchor, positive) pairs, deduplicated on BOTH sides."""
    rng = random.Random(seed)
    # sample evenly across verticals so no domain dominates the fine-tune
    per_vertical = max(1, n_pairs // (2 * max(1, df["bsns_vrtcl_name"].nunique())))
    pairs = []
    # dedup both sides: near-identical pairs in one batch collapse MNRL training
    seen_anchors, seen_positives = set(), set()
    for _, sub in df.groupby("bsns_vrtcl_name"):
        rows = sub.sample(min(len(sub), per_vertical * 4), random_state=seed)
        for _, row in rows.iterrows():
            title = row.get("Product_title")
            if not isinstance(title, str) or len(title) < 10:
                continue
            title_key = title.strip().lower()
            attr = _attr_text(row)
            if len(attr) > 20 and title_key not in seen_anchors and attr not in seen_positives:
                pairs.append((title, attr))
                seen_anchors.add(title_key)
                seen_positives.add(attr)
            q = _query_phrase(row)
            positive_b = f"{title} {attr}"[:400]
            if (
                len(q) > 8 and rng.random() < 0.5
                and q not in seen_anchors and positive_b not in seen_positives
            ):
                pairs.append((q, positive_b))
                seen_anchors.add(q)
                seen_positives.add(positive_b)
    rng.shuffle(pairs)
    return pairs[:n_pairs]


def _has_collapsed(model, sample_texts: list) -> bool:
    """True if probe embeddings are non-finite or collapsed to ~the same vector."""
    import numpy as np

    vecs = model.encode(sample_texts, convert_to_numpy=True, show_progress_bar=False)
    if not np.isfinite(vecs).all():
        print("  COLLAPSE CHECK: non-finite values in probe embeddings.")
        return True
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    unit = vecs / norms
    sims = unit @ unit.T
    off_diag = sims[~np.eye(len(sims), dtype=bool)]
    mean_sim = float(off_diag.mean())
    print(f"  COLLAPSE CHECK: mean pairwise cosine similarity across "
          f"{len(sample_texts)} distinct probe texts = {mean_sim:.4f} "
          f"(>{'0.97'} would indicate collapse).")
    return mean_sim > 0.97


class _AbortOnCollapse(TrainerCallback):
    """Stop training when loss plateaus at the MNRL collapse floor ln(batch_size)."""

    def __init__(self, batch_size: int, patience: int = 3, tol: float = 0.03):
        import math

        self.floor = math.log(batch_size)
        self.patience = patience
        self.tol = tol
        self.streak = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        loss = (logs or {}).get("loss")
        if loss is None:
            return control
        if abs(loss - self.floor) < self.tol:
            self.streak += 1
        else:
            self.streak = 0
        if self.streak >= self.patience:
            print(
                f"\n  ABORT: loss has sat within {self.tol} of the collapse "
                f"floor ln(batch_size)={self.floor:.3f} for {self.streak} "
                "consecutive logs — representation collapse detected. "
                "Stopping now instead of burning more CPU hours on a dead run."
            )
            control.should_training_stop = True
        return control


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="thenlper/gte-small")
    p.add_argument("--pairs", type=int, default=20000)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=32)
    # 2e-5 + 10% warmup collapsed on a short run; lower LR + longer warmup is the fix
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--warmup-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--max-seq", type=int, default=128,
        help="Token cap per text. CPU training cost scales with sequence "
        "length; titles+attributes fit in 128 tokens, and the default 512 "
        "made steps ~6 min on this laptop (65h/epoch — unusable).",
    )
    args = p.parse_args()

    import torch
    from datasets import Dataset
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.sentence_transformer import losses
    from sentence_transformers.sentence_transformer.trainer import (
        SentenceTransformerTrainer,
    )
    from sentence_transformers.sentence_transformer.training_args import (
        SentenceTransformerTrainingArguments,
    )

    print(f"Loading catalog from {PRODUCTS_CSV} ...")
    df = pd.read_csv(PRODUCTS_CSV)
    pairs = build_pairs(df, args.pairs, args.seed)
    print(f"Built {len(pairs):,} (anchor, positive) pairs "
          f"from {len(df):,} products.")

    # Trainer API only: this version's model.fit() shim diverged to NaN weights on CPU
    model = SentenceTransformer(args.base)
    model.max_seq_length = args.max_seq

    ds = Dataset.from_dict({
        "anchor": [a for a, _ in pairs],
        "positive": [b for _, b in pairs],
    })
    loss = losses.MultipleNegativesRankingLoss(model)
    steps = (len(pairs) // args.batch_size) * args.epochs
    train_args = SentenceTransformerTrainingArguments(
        output_dir=str(OUT_DIR),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_steps=max(10, int(steps * args.warmup_frac)),
        max_grad_norm=1.0,   # explicit, though it's also the HF default
        logging_steps=5,     # finer-grained so the collapse callback reacts fast
        save_strategy="no",  # we save once at the end, below
        report_to=[],
        use_cpu=not torch.cuda.is_available(),
    )
    print(f"Training {args.epochs} epoch(s), ~{steps} steps, "
          f"batch {args.batch_size}, lr {args.lr}, "
          f"warmup {train_args.warmup_steps} steps ...")
    trainer = SentenceTransformerTrainer(
        model=model, args=train_args, train_dataset=ds, loss=loss,
        callbacks=[_AbortOnCollapse(batch_size=args.batch_size)],
    )
    trainer.train()

    # check OUTPUT, not weights: a collapsed encoder can still have finite weights
    # probe spans verticals so a healthy model can't false-positive on one category
    probe = list(dict.fromkeys(
        pd.concat(
            [g.sample(min(len(g), 10), random_state=args.seed)["Product_title"]
             for _, g in df.groupby("bsns_vrtcl_name")]
        ).dropna().tolist()
    ))
    if _has_collapsed(model, probe):
        raise RuntimeError(
            "Training collapsed (degenerate or non-finite embeddings on the "
            "probe set). NOT saving. Retry with a lower --lr and/or higher "
            "--warmup-frac, and check --pairs is large enough for --batch-size."
        )

    model.save(str(OUT_DIR))
    print(f"\nSaved fine-tuned model to {OUT_DIR}")
    print("\nNext — measure it on the eval suite (retrieval-only, honest):")
    print("  EMBEDDING_MODEL=models/gte-small-cas-ft python -m eval.run_eval --no-rerank --tag ft_embed")
    print("  python -m eval.run_eval --no-rerank --tag base_embed   # baseline")


if __name__ == "__main__":
    main()
