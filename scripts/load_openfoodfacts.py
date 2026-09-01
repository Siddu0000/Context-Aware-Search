"""Load Open Food Facts grocery rows with nutrition (ODbL, commercially usable)."""

import argparse
import logging
import time
from typing import Optional

import pandas as pd
import requests

from app.config import DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_CSV = DATA_DIR / "products_grocery.csv"
API_BASE = "https://world.openfoodfacts.org/api/v2/search"

REQUEST_FIELDS = [
    "code", "product_name", "brands", "categories_tags",
    "image_front_small_url", "ingredients_text",
    "energy-kcal_100g", "proteins_100g", "carbohydrates_100g", "fat_100g",
    "nutriscore_grade", "nova_group", "labels_tags",
]


def _convert_row(row: dict) -> Optional[dict]:
    name = (row.get("product_name") or "").strip()
    if not name:
        return None

    cat_tags = row.get("categories_tags") or []
    sub_category = cat_tags[-1].split(":")[-1].replace("-", " ") if cat_tags else ""

    def _num(v):
        try:
            return float(v) if v not in (None, "") else None
        except (TypeError, ValueError):
            return None

    calories = _num(row.get("energy-kcal_100g"))
    protein = _num(row.get("proteins_100g"))
    carbs = _num(row.get("carbohydrates_100g"))
    fat = _num(row.get("fat_100g"))

    if calories is None and protein is None and carbs is None and fat is None:
        return None

    return {
        "bsns_vrtcl_name": "Grocery",
        "categ_lvl2_name": sub_category,
        "Product_title": name,
        "img_url": row.get("image_front_small_url") or "",
        "color": "",
        "material": "",
        "occasion": "",
        "price": None,
        "prod_description": (row.get("ingredients_text") or "")[:1000],
        "store": (row.get("brands") or "").split(",")[0].strip() if row.get("brands") else "",
        "calories_per_100g": calories,
        "protein_g_per_100g": protein,
        "carbs_g_per_100g": carbs,
        "fat_g_per_100g": fat,
        "nutriscore": row.get("nutriscore_grade") or "",
        "nova_group": row.get("nova_group"),
        "barcode": row.get("code") or "",
    }


def fetch(n: int, country: Optional[str], page_size: int = 100) -> list[dict]:
    rows: list[dict] = []
    page = 1
    pages_cap = max(1, (n + page_size - 1) // page_size) * 4

    base_params = {
        "fields": ",".join(REQUEST_FIELDS),
        "page_size": page_size,
        "states_tags": "en:nutrition-facts-completed",
    }
    if country:
        base_params["countries_tags"] = country.lower()

    while len(rows) < n and page <= pages_cap:
        params = dict(base_params, page=page)
        logger.info("OFF API page %d (have %d/%d) ...", page, len(rows), n)
        try:
            resp = requests.get(
                API_BASE,
                params=params,
                timeout=30,
                headers={"User-Agent": "CAS-PoC/1.0"},
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            logger.warning("API call failed (%s); pausing 5s.", repr(e))
            time.sleep(5)
            continue

        products = data.get("products") or []
        if not products:
            logger.info("API returned no more products; stopping.")
            break
        for p in products:
            converted = _convert_row(p)
            if converted is not None:
                rows.append(converted)
                if len(rows) >= n:
                    break
        page += 1
        time.sleep(0.5)

    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=5000, help="Target row count.")
    p.add_argument("--country", default=None, help="Restrict to one country (e.g. india).")
    p.add_argument("--page-size", type=int, default=100)
    args = p.parse_args()

    rows = fetch(args.n, args.country, args.page_size)
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print()
    print(f"Wrote {len(df)} grocery rows to {OUTPUT_CSV}")
    if not df.empty:
        print("\nTop categories:")
        print(df["categ_lvl2_name"].value_counts().head(10).to_string())


if __name__ == "__main__":
    main()
