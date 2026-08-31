"""Load Amazon Reviews 2023 meta_*.jsonl dumps into data/products_amazon.csv."""

import argparse
import json
import logging
import random
import re
from pathlib import Path
from typing import Optional

import pandas as pd

from app.config import DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_CSV = DATA_DIR / "products_amazon.csv"

VERTICAL_MAP = {
    "meta_Amazon_Fashion": "Fashion",
    "meta_Clothing_Shoes_and_Jewelry": "Fashion",
    "meta_Electronics": "Electronics",
    "meta_Cell_Phones_and_Accessories": "Electronics",
    "meta_Grocery_and_Gourmet_Food": "Grocery",
    "meta_Beauty_and_Personal_Care": "Beauty",
    "meta_Home_and_Kitchen": "Home",
}


_COLOR_WORDS = [
    "black", "white", "red", "blue", "green", "yellow", "pink", "purple",
    "orange", "brown", "grey", "gray", "navy", "beige", "gold", "silver",
    "tan", "ivory", "maroon", "olive", "burgundy", "violet", "khaki",
    "teal", "turquoise", "coral", "magenta", "lavender", "mint", "charcoal",
    "cream", "rose", "indigo", "crimson", "ruby", "emerald",
]

_MATERIAL_WORDS = [
    "cotton", "polyester", "wool", "leather", "denim", "linen", "silk",
    "rayon", "nylon", "cashmere", "velvet", "satin", "spandex", "fleece",
    "suede", "canvas", "rubber", "elastic", "lace", "knit", "mesh",
    "chiffon", "polyurethane", "neoprene",
]

_OCCASION_PATTERNS = {
    "Sports": [r"\bathletic\b", r"\bsports?\b", r"\brunning\b", r"\bworkout\b",
               r"\btraining\b", r"\bgym\b", r"\byoga\b", r"\bcompression\b"],
    "Office Wear": [r"\boffice\b", r"\bbusiness\b", r"\bprofessional\b"],
    "Formal": [r"\bformal\b", r"\bdress shirt\b", r"\btuxedo\b"],
    "Wedding": [r"\bwedding\b", r"\bbridal\b", r"\bbridesmaid\b"],
    "Party": [r"\bparty\b", r"\bcocktail\b", r"\bevening\b"],
    "Beach": [r"\bbeach\b", r"\bswim\b", r"\bsandal\b", r"\bbikini\b"],
    "Casual": [r"\bcasual\b", r"\blounge\b", r"\bvintage\b", r"\beveryday\b",
               r"\bweekend\b"],
    "Outdoor": [r"\boutdoor\b", r"\bhiking\b", r"\bcamping\b"],
}

DEPT_NORMALIZE = {
    "womens": "Women's Clothing",
    "women's": "Women's Clothing",
    "women": "Women's Clothing",
    "ladies": "Women's Clothing",
    "mens": "Men's Clothing",
    "men's": "Men's Clothing",
    "men": "Men's Clothing",
    "girls": "Kids",
    "girls'": "Kids",
    "boys": "Kids",
    "boys'": "Kids",
    "kids": "Kids",
    "children": "Kids",
    "baby": "Kids",
    "unisex-adult": "Unisex Clothing",
    "unisex-child": "Kids",
    "unisex": "Unisex Clothing",
}


def _word_match(text: str, vocab: list[str]) -> str:
    """First vocab word that appears as a whole-word match in text."""
    t = (text or "").lower()
    for w in vocab:
        if re.search(rf"\b{re.escape(w)}\b", t):
            return w
    return ""


def _parse_price(raw) -> Optional[float]:
    """In the real dumps, price is null or float. Handle string just in case."""
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    s = str(raw).strip().lstrip("$").replace(",", "")
    if not s or s.lower() in {"none", "null"}:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _pick_image(images_field) -> str:
    """images is a LIST of dicts; prefer variant=MAIN, then hi_res > large > thumb."""
    if not isinstance(images_field, list) or not images_field:
        return ""
    main = [d for d in images_field if isinstance(d, dict) and d.get("variant") == "MAIN"]
    other = [d for d in images_field if isinstance(d, dict) and d.get("variant") != "MAIN"]
    for d in main + other:
        for key in ("hi_res", "large", "thumb"):
            url = d.get(key)
            if url:
                return url
    return ""


def _joined_description(desc_field) -> str:
    """description is a list of strings in the dump, not a single string."""
    if isinstance(desc_field, list):
        return " ".join(d for d in desc_field if d)
    if isinstance(desc_field, str):
        return desc_field
    return ""


def _joined_features(features_field) -> str:
    """features is a list of strings — bullet points from the listing."""
    if isinstance(features_field, list):
        return " ".join(f for f in features_field if f)
    return ""


def _extract_from_details(details_raw) -> dict:
    """details is already a dict in the real dumps; JSON string is an older format."""
    out = {"color": "", "material": "", "department": "", "brand": ""}
    if not details_raw:
        return out

    if isinstance(details_raw, str):
        try:
            d = json.loads(details_raw)
        except (json.JSONDecodeError, TypeError):
            return out
    elif isinstance(details_raw, dict):
        d = details_raw
    else:
        return out

    if not isinstance(d, dict):
        return out

    for k, v in d.items():
        kl = str(k).lower()
        sv = str(v).strip().lower()
        if "color" in kl and not out["color"]:
            out["color"] = sv
        if "material" in kl and not out["material"]:
            out["material"] = sv
        if "department" in kl and not out["department"]:
            out["department"] = sv
        if "brand" in kl and not out["brand"]:
            out["brand"] = sv
    return out


def _infer_category(title: str, dept: str, main_cat: str, vertical: str) -> str:
    """categ_lvl2_name. Department > title gender hints > main_category."""
    if dept:
        dept_l = dept.lower().strip()
        if dept_l in DEPT_NORMALIZE:
            return DEPT_NORMALIZE[dept_l]
        for key, val in DEPT_NORMALIZE.items():
            if key in dept_l:
                return val

    t = title.lower()
    if re.search(r"\bwomen'?s?\b|\bladies'?\b", t):
        return "Women's Clothing"
    if re.search(r"\bmen'?s?\b", t):
        return "Men's Clothing"
    if re.search(r"\b(girls?'?|boys?'?|kids|children|toddler|infant|baby|youth)\b", t):
        return "Kids"

    if vertical != "Fashion":
        return (main_cat or "").title()
    return "Unisex Clothing"


def _infer_occasion(blob: str) -> str:
    """First matching occasion label wins, or '' if none match."""
    t = (blob or "").lower()
    for label, patterns in _OCCASION_PATTERNS.items():
        for p in patterns:
            if re.search(p, t):
                return label
    return ""


def _convert_row(row: dict, vertical: str) -> dict:
    """Map one Amazon metadata record to our schema."""
    title = str(row.get("title") or "").strip()
    description = _joined_description(row.get("description"))
    features_text = _joined_features(row.get("features"))
    full_text = f"{title} {features_text} {description}".strip()

    details = _extract_from_details(row.get("details"))

    color = (
        details["color"]
        or _word_match(title, _COLOR_WORDS)
        or _word_match(features_text, _COLOR_WORDS)
        or _word_match(description, _COLOR_WORDS)
    )
    material = (
        details["material"]
        or _word_match(features_text, _MATERIAL_WORDS)
        or _word_match(title, _MATERIAL_WORDS)
        or _word_match(description, _MATERIAL_WORDS)
    )
    occasion = _infer_occasion(full_text)
    sub_category = _infer_category(
        title, details["department"], row.get("main_category") or "", vertical
    )

    prod_description = (features_text + " " + description).strip()

    return {
        "bsns_vrtcl_name": vertical,
        "categ_lvl2_name": sub_category,
        "Product_title": title,
        "img_url": _pick_image(row.get("images")),
        "color": color,
        "material": material,
        "occasion": occasion,
        "price": _parse_price(row.get("price")),
        "prod_description": prod_description[:1500],
        "average_rating": row.get("average_rating"),
        "rating_number": row.get("rating_number"),
        "store": details["brand"] or row.get("store") or "",
        "parent_asin": row.get("parent_asin") or "",
    }


def _reservoir_sample_jsonl(
    path: Path, n: int, seed: int = 42, progress_every: int = 100_000
) -> list[dict]:
    """One-pass sample of N rows, memory-bounded at O(N); n < 0 takes everything."""
    rng = random.Random(seed)
    reservoir: list[dict] = []
    seen = 0
    parse_errors = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                parse_errors += 1
                continue
            seen += 1

            if n < 0 or seen <= n:
                reservoir.append(row)
            else:
                j = rng.randint(0, seen - 1)
                if j < n:
                    reservoir[j] = row

            if seen % progress_every == 0:
                logger.info(
                    "  scanned %s lines from %s (reservoir=%d)",
                    f"{seen:,}",
                    path.name,
                    len(reservoir),
                )

    if parse_errors:
        logger.warning("Skipped %d malformed lines in %s.", parse_errors, path.name)
    logger.info(
        "Done scanning %s: %s lines total, %d kept.",
        path.name,
        f"{seen:,}",
        len(reservoir),
    )
    return reservoir


def _detect_vertical(file_path: Path) -> str:
    stem = file_path.stem
    return VERTICAL_MAP.get(stem, stem.replace("meta_", "").replace("_", " "))


def _find_meta_files(data_dir: Path) -> list[Path]:
    """Auto-detect meta_* files in data dir (extension optional)."""
    return sorted(p for p in data_dir.iterdir() if p.is_file() and p.name.startswith("meta_"))


def _process_file(path: Path, n_per: int, seed: int) -> list[dict]:
    vertical = _detect_vertical(path)
    logger.info("Reading %s as vertical=%r ...", path, vertical)
    raw_rows = _reservoir_sample_jsonl(path, n_per, seed=seed)
    converted: list[dict] = []
    for row in raw_rows:
        item = _convert_row(row, vertical)
        if item["Product_title"]:
            converted.append(item)
    logger.info("Converted %d valid rows from %s.", len(converted), path.name)
    return converted


def _print_coverage(df: pd.DataFrame) -> None:
    n = len(df)
    print()
    print("Coverage of extracted fields:")
    print(f"  title present:        {(df['Product_title'] != '').sum():>7,} / {n:,}")
    print(f"  img_url present:      {(df['img_url'] != '').sum():>7,} / {n:,}")
    print(f"  price present:        {df['price'].notna().sum():>7,} / {n:,}")
    print(f"  average_rating:       {df['average_rating'].notna().sum():>7,} / {n:,}")
    print(f"  color detected:       {(df['color'] != '').sum():>7,} / {n:,}")
    print(f"  material detected:    {(df['material'] != '').sum():>7,} / {n:,}")
    print(f"  occasion detected:    {(df['occasion'] != '').sum():>7,} / {n:,}")
    print(f"  sub-category set:     {(df['categ_lvl2_name'] != '').sum():>7,} / {n:,}")
    print(f"  prod_description:     {(df['prod_description'].str.len() > 0).sum():>7,} / {n:,}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--files", nargs="+", default=None,
        help="Paths to meta_*.jsonl files. Default: auto-detect in data/.",
    )
    parser.add_argument(
        "--n-per", type=int, default=20000,
        help="Items to sample per file. -1 means all. Default: 20000.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.files:
        files = [Path(p) for p in args.files]
    else:
        files = _find_meta_files(DATA_DIR)
        if not files:
            print(
                f"\nNo meta_* files found in {DATA_DIR}.\n"
                "Either move your JSONL files there, or pass them with --files."
            )
            return
        print(f"\nAuto-detected {len(files)} file(s) in {DATA_DIR}:")
        for f in files:
            sz_mb = f.stat().st_size / (1024 * 1024)
            print(f"  - {f.name}  ({sz_mb:,.1f} MB)")
        print()

    all_rows: list[dict] = []
    for path in files:
        if not path.exists():
            logger.warning("Skipping missing file: %s", path)
            continue
        all_rows.extend(_process_file(path, args.n_per, args.seed))

    if not all_rows:
        print("\nNo rows produced. Check your input files.")
        return

    df = pd.DataFrame(all_rows)
    df.to_csv(OUTPUT_CSV, index=False)

    print()
    print(f"Wrote {len(df):,} rows to {OUTPUT_CSV}")
    print()
    print("Verticals:")
    print(df["bsns_vrtcl_name"].value_counts().to_string())
    print()
    print("Top sub-categories:")
    print(df["categ_lvl2_name"].value_counts().head(10).to_string())
    _print_coverage(df)
    print()
    print("Next steps:")
    print(f"  cp {OUTPUT_CSV} data/products.csv")
    print("  # then restart uvicorn — embedding cache rebuilds once on first boot.")


if __name__ == "__main__":
    main()
