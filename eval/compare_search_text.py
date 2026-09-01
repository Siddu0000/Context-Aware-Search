"""Compare which catalog fields are concatenated into each product's search_text."""

import logging
import time
from typing import Tuple

from app import search as search_module
from eval.run_eval import evaluate

logging.basicConfig(level=logging.WARNING)


VARIANTS: dict[str, Tuple[str, ...]] = {
    "title_only": ("Product_title",),
    "title_plus_desc": ("Product_title", "prod_description"),
    "title_color": ("Product_title", "color"),
    "title_attrs": ("Product_title", "color", "material", "occasion"),
    "all_fields": (
        "bsns_vrtcl_name",
        "categ_lvl2_name",
        "Product_title",
        "prod_description",
        "color",
        "material",
        "occasion",
    ),
}


def run_for_variant(name: str, fields: Tuple[str, ...]):
    """Force a rebuild of the index using this field set, then run eval."""
    search_module._df = None
    search_module._embeddings = None
    search_module._loaded_fields = ()

    original = search_module.DEFAULT_SEARCH_FIELDS
    search_module.DEFAULT_SEARCH_FIELDS = fields
    try:
        t0 = time.perf_counter()
        out_path = evaluate(rerank_on=False, tag=f"chunk_{name}")
        return out_path, time.perf_counter() - t0
    finally:
        search_module.DEFAULT_SEARCH_FIELDS = original


def main():
    print("\n=== Search-text (chunking) comparison ===")
    for name, fields in VARIANTS.items():
        print(f"\n--- {name}: {fields} ---")
        try:
            out_path, elapsed = run_for_variant(name, fields)
            print(f"completed in {elapsed:.1f}s, results -> {out_path}")
        except Exception as e:
            print(f"FAILED for {name}: {e!r}")


if __name__ == "__main__":
    main()
