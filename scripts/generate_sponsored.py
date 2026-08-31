"""Generate data/sponsored.json: 1:ratio catalog sample, stratified by sub-category."""

import argparse
import json
import random
from collections import Counter

import pandas as pd

from app.config import PRODUCTS_CSV, SPONSORED_CONFIG

SPONSORS = [
    "Acme Retail", "Nova Brands", "Peak Goods", "Vertex Co", "Harbor & Lane",
    "BlueOak", "Summit Supply", "Everline", "NorthStar", "Kettle & Co",
]
BID_MIN, BID_MAX = 1.0, 10.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratio", type=int, default=50,
                        help="Sponsored rate 1:ratio (default 50 = ~2%).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-per-subcat", type=int, default=0,
                        help="Force at least this many per sub-category (default 0 = strict ratio).")
    parser.add_argument("--out", default=str(SPONSORED_CONFIG))
    args = parser.parse_args()

    if not PRODUCTS_CSV.exists():
        raise SystemExit(f"Catalog not found: {PRODUCTS_CSV}")

    rng = random.Random(args.seed)
    df = pd.read_csv(PRODUCTS_CSV)
    if "parent_asin" not in df.columns:
        raise SystemExit("products.csv has no 'parent_asin' column.")

    subcat_col = "categ_lvl2_name" if "categ_lvl2_name" in df.columns else None
    cat_col = "bsns_vrtcl_name" if "bsns_vrtcl_name" in df.columns else None
    title_col = "Product_title" if "Product_title" in df.columns else None

    df = df.dropna(subset=["parent_asin"]).drop_duplicates(subset="parent_asin")

    group_key = df[subcat_col].fillna("Unknown") if subcat_col else pd.Series("All", index=df.index)

    picks = []
    for subcat, group in df.groupby(group_key):
        n = len(group)
        k = round(n / args.ratio)
        k = max(k, args.min_per_subcat)
        k = min(k, n)
        if k <= 0:
            continue
        chosen_idx = rng.sample(list(group.index), k)
        for i in chosen_idx:
            row = df.loc[i]
            # bid orders sponsored items within one pool; category/title are audit-only
            picks.append({
                "parent_asin": str(row["parent_asin"]),
                "sponsor": rng.choice(SPONSORS),
                "bid": round(rng.uniform(BID_MIN, BID_MAX), 1),
                "category": str(row[cat_col]) if cat_col else "",
                "sub_category": str(subcat),
                "title": (str(row[title_col])[:80] if title_col else ""),
            })

    rng.shuffle(picks)
    out = {"sponsored": picks}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    total = len(df)
    print(f"Catalog: {total:,} unique products")
    print(f"Sampled {len(picks)} sponsored (target 1:{args.ratio} = {total/args.ratio:.0f}) -> {args.out}")
    if cat_col:
        by_cat = Counter(p["category"] for p in picks)
        print("Per category:")
        for c, n in sorted(by_cat.items()):
            print(f"  {c}: {n}")
    n_subcats_covered = len({p["sub_category"] for p in picks})
    print(f"Sub-categories represented: {n_subcats_covered}")


if __name__ == "__main__":
    main()
