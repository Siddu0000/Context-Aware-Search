"""Clean the catalog and fill its gaps: real data first, synthetic last, every value tagged.

Stages (each writes to data/enrichment/, so any stage can be re-run alone):
  recover   re-scan the raw Amazon JSONL for `details` fields the loader dropped
            (flavor, specialty, compatible devices, ...) -> details_extra.jsonl
  rules     deterministic cleaning + rule-based attributes      -> rules.parquet
  price     price_estimate from the nearest PRICED neighbours    -> price.parquet
  llm       local open-source LLM writes descriptions/use-cases for rows with no
            description (resumable checkpoint, --limit for a pilot)
  assemble  merge everything into data/products_enriched.csv

Row order and count never change: Databricks uses row position as catalog_index.
Real columns are cleaned in place; anything inferred goes in its own column or is
tagged in a *_src column (raw | details | rule | llm | ""), so a reader can always
tell observed data from generated data.

Usage:
    python -m scripts.enrich_catalog recover rules price
    python -m scripts.enrich_catalog llm --limit 1000          # pilot
    python -m scripts.enrich_catalog llm                       # full run: DAYS on CPU, see LLM_MODEL
    python -m scripts.enrich_catalog assemble
"""

import argparse
import html
import json
import logging
import re
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from app.config import CACHE_DIR, DATA_DIR

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("enrich")

OUT_DIR = DATA_DIR / "enrichment"
DETAILS_PATH = OUT_DIR / "details_extra.jsonl"
RULES_PATH = OUT_DIR / "rules.parquet"
PRICE_PATH = OUT_DIR / "price.parquet"
LLM_CHECKPOINT = OUT_DIR / "llm_checkpoint.jsonl"
ENRICHED_CSV = DATA_DIR / "products_enriched.csv"
# Always the BASE catalog: PRODUCTS_CSV may point the app at the enriched copy,
# and enriching that again would compound inferences
BASE_CSV = DATA_DIR / "products.csv"

RAW_FILES = {
    "Fashion": "meta_Amazon_Fashion.jsonl",
    "Electronics": "meta_Electronics.jsonl",
    "Grocery": "meta_Grocery_and_Gourmet_Food.jsonl",
}

JUNK = {"", "unk", "unknown", "none", "nan", "null", "n/a", "na", "-", "[]", "{}", "not applicable"}

# ---------------------------------------------------------------- vocabularies
# Ordered: multi-word and specific shades first so "rose gold" isn't read as "rose"
COLOR_FAMILIES = [
    ("multicolor", r"multi[- ]?colou?r(ed)?|rainbow|assorted colou?rs?"),
    ("clear", r"clear|transparent"),
    ("gold", r"rose gold|gold(en)?"),
    ("silver", r"silver|chrome"),
    ("navy", r"navy"),
    ("grey", r"gr[ae]y|charcoal|graphite|slate|gunmetal"),
    ("black", r"black|jet|onyx"),
    ("white", r"white|ivory|off[- ]white"),
    ("beige", r"beige|khaki|tan|nude|sand|taupe|cream|camel"),
    ("brown", r"brown|chocolate|coffee|mocha|espresso"),
    ("burgundy", r"burgundy|maroon|wine"),
    ("red", r"red|crimson|scarlet"),
    ("pink", r"pink|rose|blush|fuchsia|magenta"),
    ("purple", r"purple|violet|lavender|lilac|plum"),
    ("blue", r"blue|teal|turquoise|aqua|cyan|cobalt"),
    ("green", r"green|olive|sage|mint|emerald"),
    ("yellow", r"yellow|mustard"),
    ("orange", r"orange|coral|peach"),
]
COLOR_RE = [(fam, re.compile(rf"\b(?:{pat})\b", re.I)) for fam, pat in COLOR_FAMILIES]

MATERIALS = [
    ("faux leather", r"faux leather|pu leather|vegan leather"),
    ("leather", r"leather"),
    ("cashmere", r"cashmere"),
    ("wool", r"wool|merino"),
    ("linen", r"linen"),
    ("silk", r"silk"),
    ("denim", r"denim"),
    ("cotton", r"cotton"),
    ("polyester", r"polyester"),
    ("nylon", r"nylon"),
    ("spandex", r"spandex|elastane|lycra"),
    ("rayon", r"rayon|viscose"),
    ("modal", r"modal"),
    ("bamboo", r"bamboo"),
    ("fleece", r"fleece"),
    ("velvet", r"velvet"),
    ("satin", r"satin"),
    ("chiffon", r"chiffon"),
    ("lace", r"lace"),
    ("suede", r"suede"),
    ("canvas", r"canvas"),
    ("mesh", r"mesh"),
    ("acrylic", r"acrylic"),
    ("down", r"down[- ]filled|goose down|duck down"),
    ("rubber", r"rubber"),
    ("silicone", r"silicone"),
    ("stainless steel", r"stainless steel"),
    ("aluminum", r"alumin(i)?um"),
    ("titanium", r"titanium"),
    ("plastic", r"plastic|abs|polycarbonate|tpu"),
    ("glass", r"tempered glass|glass"),
    ("wood", r"wood(en)?|walnut|oak"),
    ("metal", r"metal|alloy|steel"),
]
MATERIAL_RE = [(m, re.compile(rf"\b(?:{pat})\b", re.I)) for m, pat in MATERIALS]

# Fashion keeps the loader's occasion vocabulary; first match wins
# Lounge before formal ("dressing gown" is a robe), beach before office ("bathing
# suit"); bare "suit"/"tennis"/"waterproof" removed (swimsuits, tennis bracelets)
OCCASION_RULES = [
    ("lounge", r"pajama|pyjama|sleepwear|lounge(wear)?|bath ?robe|robe|dressing gown|nightgown|slipper"),
    ("wedding", r"wedding|bridal|bridesmaid|groomsm[ae]n"),
    ("formal", r"formal|ball gown|evening gown|gown|tuxedo|evening dress|black tie"),
    ("party", r"party|cocktail|club ?wear|sequin|prom"),
    ("beach", r"beach|swim(wear|suit)?|bikini|bathing suit|board ?shorts|cover[- ]?up"),
    ("office wear", r"office|business|work ?wear|blazer|suit jacket|pant ?suit|dress (shirt|pants)"),
    ("sports", r"gym|workout|running|yoga|athletic|sports? bra|activewear|training|jogger|golf|"
               r"tennis (shoes?|skirt|dress)"),
    ("outdoor", r"hiking|outdoor|camping|trail|rain ?(coat|jacket|boots?)"),
    ("casual", r"casual|t[- ]?shirt|tee\b|hoodie|sweatshirt|jeans|sneakers?"),
]
OCCASION_RE = [(o, re.compile(rf"\b(?:{pat})\b", re.I)) for o, pat in OCCASION_RULES]

USE_CASE_RULES = {
    "Electronics": [
        ("gaming", r"gaming|gamer|xbox|playstation|ps[345]\b|nintendo|switch controller"),
        ("photography", r"camera|lens|tripod|gimbal|photo|dslr|mirrorless"),
        ("home entertainment", r"\btv\b|television|projector|soundbar|home theat(er|re)|streaming|roku|fire stick"),
        ("car", r"\bcar\b|vehicle|dash ?cam|automotive"),
        ("fitness", r"fitness|smart ?watch|heart rate|pedometer|running"),
        ("audio", r"headphone|earbud|earphone|speaker|audio|microphone|\bmic\b|amplifier"),
        ("smart home", r"smart home|alexa|smart plug|security camera|doorbell|google home"),
        ("travel", r"travel|power bank|luggage"),
        ("office", r"office|printer|keyboard|mouse|webcam|monitor|laptop|desk|scanner|ink"),
        ("phone accessory", r"phone case|screen protector|iphone|galaxy|phone charger|phone holder"),
        ("outdoor", r"outdoor|camping|hiking|solar|flashlight"),
    ],
    "Grocery": [
        ("breakfast", r"cereal|oatmeal|oats|granola|pancake|waffle|breakfast"),
        ("beverage", r"coffee|\btea\b|juice|soda|drink|water|cocoa mix|latte|espresso|kombucha"),
        ("baking", r"flour|baking|yeast|cake mix|frosting|sprinkles|vanilla extract|brownie mix"),
        ("dessert", r"candy|candies|chocolate|cookie|gummy|gummies|\bgum\b|dessert|toffee|caramel"),
        ("snack", r"chips|crackers|nuts|jerky|snack|popcorn|trail mix|pretzel|\bbars?\b"),
        ("cooking", r"sauce|spice|seasoning|\boil\b|vinegar|pasta|rice|broth|condiment|marinade|salt|pepper"),
    ],
}
USE_CASE_RE = {v: [(u, re.compile(rf"\b(?:{p})\b", re.I)) for u, p in rules]
               for v, rules in USE_CASE_RULES.items()}
USE_CASE_VOCAB = {v: [u for u, _ in rules] for v, rules in USE_CASE_RULES.items()}
OCCASION_VOCAB = [o for o, _ in OCCASION_RULES]

DIETARY_RULES = [
    ("gluten free", r"gluten[- ]?free|no gluten|celiac"),
    ("vegan", r"vegan|plant[- ]based"),
    ("vegetarian", r"vegetarian"),
    ("organic", r"organic"),
    ("keto", r"keto"),
    ("sugar free", r"sugar[- ]?free|no (added )?sugar|zero sugar|unsweetened"),
    ("non-gmo", r"non[- ]?gmo"),
    ("kosher", r"kosher"),
    ("halal", r"halal"),
    ("dairy free", r"dairy[- ]?free|lactose[- ]?free|non[- ]dairy"),
    ("low carb", r"low[- ]?carb"),
    ("high protein", r"high[- ]?protein|protein bar|protein powder"),
    ("caffeine free", r"caffeine[- ]?free|decaf"),
    ("nut free", r"nut[- ]?free|peanut[- ]?free"),
    ("low sodium", r"low[- ]?sodium|no salt"),
]
DIETARY_RE = [(d, re.compile(rf"\b(?:{p})\b", re.I)) for d, p in DIETARY_RULES]

GENDER_RE = [
    ("kids", re.compile(r"\b(kids?|girls?|boys?|toddler|baby|infant|youth|children'?s?)\b", re.I)),
    ("women", re.compile(r"\b(women'?s?|womens|ladies|female|maternity|her)\b", re.I)),
    ("men", re.compile(r"\b(men'?s?|mens|male|his)\b", re.I)),
    ("unisex", re.compile(r"\bunisex\b", re.I)),
]

# details keys that carry no shopping meaning (logistics, ids, dates)
_USELESS_KEY = re.compile(
    r"date|dimension|model number|upc|asin|rank|discontinued|weight|part number|"
    r"batteries|country of origin|global trade|\bean\b|isbn|review|package quantity|"
    r"manufacturer|item model|domestic shipping|international shipping|department",
    re.I,
)
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


# --------------------------------------------------------------------- helpers
def clean_text(value) -> str:
    """Unescape HTML, drop tags, NFKC-normalise, collapse whitespace; junk -> ''."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    s = html.unescape(str(value))
    s = _TAG_RE.sub(" ", s)
    s = unicodedata.normalize("NFKC", s)
    s = _WS_RE.sub(" ", s).strip()
    return "" if s.lower() in JUNK else s


def first_match(rules, text: str) -> str:
    for label, rx in rules:
        if rx.search(text):
            return label
    return ""


def all_matches(rules, text: str) -> list:
    return [label for label, rx in rules if rx.search(text)]


def color_family(value: str) -> str:
    """Canonical color for a free-text color value, '' if it names none."""
    return first_match(COLOR_RE, value) if value else ""


def material_family(value: str) -> str:
    return first_match(MATERIAL_RE, value) if value else ""


# --------------------------------------------------------------- stage: recover
_ASIN_RE = re.compile(r'"parent_asin":\s*"([^"]+)"')


def _spec_items(details) -> dict:
    """Shopping-relevant details as {Key: short value}; logistics and ids dropped."""
    if isinstance(details, str):
        try:
            details = json.loads(details)
        except json.JSONDecodeError:
            return {}
    if not isinstance(details, dict):
        return {}
    out = {}
    for k, v in details.items():
        key = clean_text(k).strip(" :")
        if not key or _USELESS_KEY.search(key):
            continue
        if isinstance(v, (list, tuple)):
            v = ", ".join(str(x) for x in v if x)
        elif isinstance(v, dict):
            continue
        val = clean_text(v)
        if val and len(val) <= 120:
            out[key] = val
    return out


def stage_recover(df: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    found = 0
    with open(DETAILS_PATH, "w", encoding="utf-8") as out:
        for vertical, fname in RAW_FILES.items():
            path = DATA_DIR / fname
            if not path.exists():
                logger.warning("Missing %s -- skipping %s.", path, vertical)
                continue
            wanted = set(df.loc[df["bsns_vrtcl_name"] == vertical, "parent_asin"].astype(str))
            t0, n = time.perf_counter(), 0
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    n += 1
                    m = _ASIN_RE.search(line)
                    # Parse only the ~1 in 5 lines we sampled: json.loads is the slow part
                    if not m or m.group(1) not in wanted:
                        continue
                    row = json.loads(line)
                    det = row.get("details") or {}
                    rec = {
                        "parent_asin": m.group(1),
                        "specs": _spec_items(det),
                        "brand": clean_text((det.get("Brand") if isinstance(det, dict) else "") or ""),
                        "maker": clean_text((det.get("Manufacturer") if isinstance(det, dict) else "") or ""),
                        "department": clean_text((det.get("Department") if isinstance(det, dict) else "") or ""),
                    }
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    wanted.discard(m.group(1))
                    found += 1
            logger.info("%s: scanned %s lines in %.0fs; %s rows not found in raw file.",
                        vertical, f"{n:,}", time.perf_counter() - t0, f"{len(wanted):,}")
    logger.info("recover: %s records -> %s", f"{found:,}", DETAILS_PATH)


def _load_details() -> pd.DataFrame:
    if not DETAILS_PATH.exists():
        raise SystemExit("Run the 'recover' stage first.")
    return pd.read_json(DETAILS_PATH, lines=True, dtype={"parent_asin": str})


def _specs_text(specs: dict, limit: int = 300) -> str:
    """'Key: value; ...' capped at `limit` chars, whole items only."""
    parts, used = [], 0
    for k, v in (specs or {}).items():
        item = f"{k}: {v}"
        if used + len(item) > limit:
            break
        parts.append(item)
        used += len(item) + 2
    return "; ".join(parts)


# ----------------------------------------------------------------- stage: rules
def stage_rules(df: pd.DataFrame) -> None:
    det = _load_details().drop_duplicates("parent_asin").set_index("parent_asin")
    asin = df["parent_asin"].astype(str)
    specs = asin.map(det["specs"]).apply(lambda s: s if isinstance(s, dict) else {})
    out = pd.DataFrame(index=df.index)
    out["catalog_index"] = np.arange(len(df))
    vert = df["bsns_vrtcl_name"].astype(str)

    title = df["Product_title"].map(clean_text)
    desc = df["prod_description"].map(clean_text)
    # A "description" that only repeats the title says nothing new
    desc = desc.where(desc.str.lower() != title.str.lower(), "")
    out["Product_title"] = title
    out["prod_description"] = desc
    out["desc_src"] = np.where(desc.str.len() > 0, "raw", "")
    out["specs"] = specs.map(_specs_text)
    spec_text = specs.map(lambda s: " ".join(f"{k} {v}" for k, v in s.items()))
    # Title + specs describe THIS product; descriptions drift ("great with coffee",
    # "waterproof case sold separately"), so they are only a fallback where reliable
    primary = (title + " " + spec_text).str.lower()
    blob = (primary + " " + desc.str[:600].str.lower())

    # store: listing brand, then details Brand / Manufacturer
    store = df["store"].map(clean_text)
    fill_store = asin.map(det["brand"]).fillna("").where(lambda s: s != "", asin.map(det["maker"]).fillna(""))
    out["store"] = store.where(store != "", fill_store.fillna(""))
    out["store_src"] = np.where(store != "", "raw", np.where(out["store"] != "", "details", ""))

    # category: fill the 2% gap from department / title, else the vertical's catch-all
    cat = df["categ_lvl2_name"].map(clean_text)
    dept = asin.map(det["department"]).fillna("").str.lower()
    gender_hint = (dept + " " + title.str.lower()).map(lambda t: first_match(GENDER_RE, t))
    fashion_cat = gender_hint.map({"women": "Women's Clothing", "men": "Men's Clothing",
                                   "kids": "Kids", "unisex": "Unisex Clothing"}).fillna("Unisex Clothing")
    default_cat = np.select([vert == "Electronics", vert == "Grocery"], ["All Electronics", "Grocery"],
                            default=fashion_cat)
    out["categ_lvl2_name"] = cat.where(cat != "", default_cat)
    out["category_src"] = np.where(cat != "", "raw", "rule")

    # gender (fashion only): category is authoritative, then department, then title
    cat_l = out["categ_lvl2_name"].str.lower()
    g = np.select(
        [cat_l.str.contains("women"), cat_l.str.contains(r"\bmen"), cat_l.str.contains("kid|girl|boy|baby"),
         cat_l.str.contains("unisex")],
        ["women", "men", "kids", "unisex"], default="")
    g = pd.Series(g, index=df.index).where(lambda s: s != "", gender_hint)
    out["gender"] = g.where(vert == "Fashion", "")

    # color: canonical family from the observed value, else from title/specs (never grocery)
    raw_color = df["color"].map(clean_text)
    fam = raw_color.map(color_family)
    observed = fam.where(fam != "", raw_color.where(raw_color.str.len().between(1, 20), ""))
    inferred = (title + " " + spec_text).map(color_family)
    color = observed.where(observed != "", inferred)
    is_food = vert == "Grocery"
    out["color"] = color.where(~is_food, "")
    out["color_src"] = np.where(is_food, "", np.where(observed != "", "raw", np.where(inferred != "", "rule", "")))

    # material: food has none; otherwise observed value, then title/specs/description
    raw_mat = df["material"].map(clean_text)
    mfam = raw_mat.map(material_family)
    m_obs = mfam.where(mfam != "", raw_mat.where(raw_mat.str.len().between(1, 25), ""))
    # Electronics descriptions name materials of OTHER things ("glass screen"), so title+specs only
    m_inf = primary.map(material_family)
    m_desc = blob.map(material_family)
    m_inf = m_inf.where((m_inf != "") | (vert != "Fashion"), m_desc)
    mat = m_obs.where(m_obs != "", m_inf)
    out["material"] = mat.where(~is_food, "")
    out["material_src"] = np.where(is_food, "", np.where(m_obs != "", "raw", np.where(m_inf != "", "rule", "")))

    # occasion is a FASHION attribute; electronics/grocery get use_case instead
    raw_occ = df["occasion"].map(clean_text).str.lower()
    occ_inf = primary.map(lambda t: first_match(OCCASION_RE, t))
    occ_inf = occ_inf.where(occ_inf != "", blob.map(lambda t: first_match(OCCASION_RE, t)))
    is_fashion = vert == "Fashion"
    occ = raw_occ.where(raw_occ != "", occ_inf)
    out["occasion"] = occ.where(is_fashion, "")
    out["occasion_src"] = np.where(~is_fashion, "", np.where(raw_occ != "", "raw", np.where(occ_inf != "", "rule", "")))

    use = pd.Series("", index=df.index)
    for v, rules in USE_CASE_RE.items():
        mask = vert == v
        use[mask] = primary[mask].map(lambda t, r=rules: first_match(r, t))
    out["use_case"] = use
    out["use_case_src"] = np.where(use != "", "rule", "")

    diet = blob.map(lambda t: ", ".join(all_matches(DIETARY_RE, t)))
    out["dietary"] = diet.where(is_food, "")

    # price: only real, positive prices stay in `price`
    price = pd.to_numeric(df["price"], errors="coerce")
    out["price"] = price.where(price > 0)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(RULES_PATH, index=False)
    _coverage_report(df, out)
    logger.info("rules -> %s", RULES_PATH)


def _missing(s: pd.Series) -> pd.Series:
    if s.dtype.kind in "fi":
        return s.isna()
    return s.isna() | s.astype(str).str.strip().str.lower().isin(JUNK)


def _coverage_report(before: pd.DataFrame, after: pd.DataFrame) -> None:
    print("\nCoverage (non-empty share), before -> after, per vertical:")
    cols = ["prod_description", "color", "material", "occasion", "use_case", "dietary",
            "gender", "specs", "store", "categ_lvl2_name", "price"]
    for v in sorted(before["bsns_vrtcl_name"].unique()):
        mb, ma = before["bsns_vrtcl_name"] == v, after.index.isin(before.index[before["bsns_vrtcl_name"] == v])
        cells = []
        for c in cols:
            a = 1 - _missing(after.loc[ma, c]).mean() if c in after else float("nan")
            b = 1 - _missing(before.loc[mb, c]).mean() if c in before else 0.0
            cells.append(f"{c}={b:.0%}->{a:.0%}")
        print(f"  {v:12s} " + "  ".join(cells))


# ----------------------------------------------------------------- stage: price
def _cached_embeddings() -> np.ndarray:
    """The app's gte-small cache for the CURRENT products.csv (no re-encode)."""
    from app.embeddings import get_default_embedder
    from app.search import DEFAULT_SEARCH_FIELDS, _file_hash

    # Same key app/search.py builds for the base catalog, without trusting cfg.PRODUCTS_CSV
    key = "_".join([_file_hash(BASE_CSV), get_default_embedder().model_name.replace("/", "_"),
                    "-".join(sorted(DEFAULT_SEARCH_FIELDS))])
    path = CACHE_DIR / f"embeddings_{key}.npy"
    if not path.exists():
        raise SystemExit(f"No embedding cache at {path}: build the BASE catalog's cache with "
                         "PRODUCTS_CSV=data/products.csv python -m scripts.encode_catalog "
                         "(the API now serves the enriched catalog, a different cache).")
    emb = np.load(path).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
    return emb


def _knn_estimate(q: np.ndarray, ref: np.ndarray, ref_price: np.ndarray, k: int,
                  chunk: int = 2048, exclude_self: np.ndarray | None = None):
    """Median price of the k most similar priced products, plus their mean similarity."""
    est = np.empty(len(q), dtype=np.float32)
    sim = np.empty(len(q), dtype=np.float32)
    for s in range(0, len(q), chunk):
        scores = q[s:s + chunk] @ ref.T
        if exclude_self is not None:
            rows = np.arange(scores.shape[0])
            scores[rows, exclude_self[s:s + chunk]] = -np.inf
        top = np.argpartition(-scores, k, axis=1)[:, :k]
        est[s:s + chunk] = np.median(ref_price[top], axis=1)
        sim[s:s + chunk] = np.take_along_axis(scores, top, axis=1).mean(axis=1)
    return est, sim


def stage_price(df: pd.DataFrame, k: int = 10, min_sim: float = 0.90) -> None:
    rules = pd.read_parquet(RULES_PATH)
    emb = _cached_embeddings()
    if len(emb) != len(df):
        raise SystemExit("Embedding cache does not match products.csv row count.")
    est_all = np.full(len(df), np.nan, dtype=np.float32)
    sim_all = np.full(len(df), np.nan, dtype=np.float32)
    rng = np.random.default_rng(42)
    print(f"\nPrice estimate = median of the {k} most similar PRICED products in the same "
          f"vertical; kept only if their mean similarity >= {min_sim}.")
    print("Hold-out check on real prices (leave-one-out, 3,000 per vertical):")
    for v in sorted(df["bsns_vrtcl_name"].unique()):
        idx = np.flatnonzero(df["bsns_vrtcl_name"].values == v)
        price = rules["price"].values[idx]
        priced = idx[~np.isnan(price)]
        unpriced = idx[np.isnan(price)]
        ref, ref_price = emb[priced], rules["price"].values[priced].astype(np.float32)

        # Honest accuracy: estimate real prices we already know, excluding the item itself
        sample = rng.choice(len(priced), size=min(3000, len(priced)), replace=False)
        e, s = _knn_estimate(ref[sample], ref, ref_price, k, exclude_self=sample)
        keep = s >= min_sim
        truth = ref_price[sample]
        ape = np.abs(e - truth) / truth
        print(f"  {v:12s} kept {keep.mean():5.1%} | median abs error {np.median(ape[keep]):5.1%} | "
              f"within ±25%: {(ape[keep] <= .25).mean():5.1%} | within ±50%: {(ape[keep] <= .5).mean():5.1%}")

        e, s = _knn_estimate(emb[unpriced], ref, ref_price, k)
        est_all[unpriced] = np.where(s >= min_sim, e, np.nan)
        sim_all[unpriced] = s
    out = pd.DataFrame({"catalog_index": np.arange(len(df)),
                        "price_estimate": np.round(est_all, 2), "price_est_sim": np.round(sim_all, 3)})
    out.to_parquet(PRICE_PATH, index=False)
    n = np.isfinite(est_all).sum()
    print(f"\nprice_estimate filled for {n:,} of {np.isnan(rules['price'].values).sum():,} unpriced rows.")
    logger.info("price -> %s", PRICE_PATH)


# ------------------------------------------------------------------- stage: llm
# Apache-2.0 (the 3B sibling is research-only). Measured on this CPU (Core Ultra 7,
# fp32, 2026-10-08): 1.5B 0.19 rows/s, 0.5B 0.58 rows/s; torch int8 dynamic
# quantization produced garbage for both, so it is not offered.
LLM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

# Short + plain text on purpose: a 1.5B model broke the JSON format in 5 of 8 pilot
# replies, its labels were unreliable (phone case -> "gaming"), and on CPU the
# prompt costs as much as the answer. Labels therefore come from the rules only.
LLM_SYSTEM = ("You write one-sentence product descriptions for an online store. Use only the "
              "facts given. Never invent sizes, numbers, materials, colors, brands or uses.")
LLM_USER = ("Title: {title}\nCategory: {category}\nSpecs: {specs}\n"
            "Write one factual sentence (max 25 words): what it is and who or what it is for.")

_NUM_RE = re.compile(r"\d+(?:\.\d+)?")


def _llm_rows(rules: pd.DataFrame, scope: str) -> pd.DataFrame:
    """Rows still lacking a usable description; 'title-only' also skips rows with specs."""
    need = rules["prod_description"].str.len() < 40
    if scope == "title-only":
        # Specs are embedded as their own column, so these rows already carry real facts
        need &= rules["specs"] == ""
    return rules.loc[need, ["catalog_index", "Product_title", "categ_lvl2_name", "specs"]]


def _grounded(text: str, source: str) -> str:
    """'' if `text` is grounded in `source`, else the first invented number/color/material."""
    src = source.lower()
    for n in _NUM_RE.findall(text):
        if n not in src:
            return f"number {n}"
    for label, rx in COLOR_RE + MATERIAL_RE:
        if rx.search(text) and not rx.search(src):
            return f"attribute {label}"
    return ""


def _first_sentence(raw: str) -> str:
    """The model's first line, without labels or wrapping quotes."""
    line = next((l for l in raw.strip().splitlines() if l.strip()), "")
    line = re.sub(r"^\W*(description|product description)\W*:\s*", "", line, flags=re.I)
    return clean_text(line.strip().strip('"').strip())


def stage_llm(df: pd.DataFrame, limit: int | None, batch_size: int, model_name: str,
              threads: int, max_new_tokens: int, scope: str) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rules = pd.read_parquet(RULES_PATH)
    rows = _llm_rows(rules, scope)
    done = set()
    if LLM_CHECKPOINT.exists():
        with open(LLM_CHECKPOINT, encoding="utf-8") as f:
            done = {json.loads(l)["catalog_index"] for l in f if l.strip()}
    # Seeded shuffle: a pilot of N rows covers every vertical
    rows = rows.sample(frac=1.0, random_state=7)
    rows = rows[~rows["catalog_index"].isin(done)]
    if limit is not None:
        rows = rows.head(max(0, limit - len(done)))
    logger.info("llm: %s rows to generate (%s already done, scope=%s) with %s", f"{len(rows):,}",
                f"{len(done):,}", scope, model_name)
    if rows.empty:
        return

    torch.set_num_threads(threads)
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.padding_side = "left"  # decoder-only batching needs left padding
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32).eval()

    t0, n_ok, n_rej = time.perf_counter(), 0, 0
    with open(LLM_CHECKPOINT, "a", encoding="utf-8") as ckpt:
        for s in range(0, len(rows), batch_size):
            batch = rows.iloc[s:s + batch_size]
            texts = [tok.apply_chat_template(
                [{"role": "system", "content": LLM_SYSTEM},
                 {"role": "user", "content": LLM_USER.format(
                     title=r.Product_title[:200], category=r.categ_lvl2_name, specs=r.specs or "(none)")}],
                tokenize=False, add_generation_prompt=True) for r in batch.itertuples()]
            enc = tok(texts, return_tensors="pt", padding=True)
            with torch.inference_mode():
                gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                     pad_token_id=tok.pad_token_id or tok.eos_token_id)
            replies = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            for r, raw in zip(batch.itertuples(), replies):
                desc = _first_sentence(raw)
                why = _grounded(desc, f"{r.Product_title} {r.specs} {r.categ_lvl2_name}") if desc else "empty"
                n_words = len(desc.split())
                if not why and not 4 <= n_words <= 45:
                    why = f"{n_words} words"
                n_ok += not why
                n_rej += bool(why)
                ckpt.write(json.dumps({"catalog_index": int(r.catalog_index), "model": model_name,
                                       "description": "" if why else desc, "rejected": why,
                                       "raw": raw[:300]}, ensure_ascii=False) + "\n")
            ckpt.flush()
            done_n = s + len(batch)
            rate = done_n / (time.perf_counter() - t0)
            logger.info("llm: %s/%s rows | %.2f rows/s | eta %.1fh | kept %s, rejected %s",
                        f"{done_n:,}", f"{len(rows):,}", rate, (len(rows) - done_n) / rate / 3600,
                        n_ok, n_rej)


# -------------------------------------------------------------- stage: assemble
def stage_assemble(df: pd.DataFrame) -> None:
    rules = pd.read_parquet(RULES_PATH)
    out = df.copy()
    for col in ["Product_title", "prod_description", "store", "categ_lvl2_name", "color",
                "material", "occasion", "price"]:
        out[col] = rules[col].values
    for col in ["specs", "gender", "use_case", "dietary", "desc_src", "store_src", "category_src",
                "color_src", "material_src", "occasion_src", "use_case_src"]:
        out[col] = rules[col].values

    out["price_estimate"] = np.nan
    if PRICE_PATH.exists():
        pe = pd.read_parquet(PRICE_PATH)
        # Never overwrite a real price: the estimate only exists where price is missing
        out["price_estimate"] = np.where(out["price"].isna(), pe["price_estimate"].values, np.nan)

    out["gen_description"] = ""
    if LLM_CHECKPOINT.exists():
        recs = pd.read_json(LLM_CHECKPOINT, lines=True).drop_duplicates("catalog_index", keep="last")
        recs = recs.set_index("catalog_index")
        pos = recs.index.values
        out.loc[pos, "gen_description"] = recs["description"].fillna("").values
        has = recs["description"].fillna("") != ""
        out.loc[recs.index[has], "desc_src"] = "llm"

    assert len(out) == len(df) and (out["parent_asin"].values == df["parent_asin"].values).all(), \
        "row order changed -- catalog_index would break"
    out.to_csv(ENRICHED_CSV, index=False)
    _coverage_report(df, out)
    est = out["price_estimate"].notna().sum()
    print(f"\nprices: {out['price'].notna().sum():,} real, {est:,} estimated (shown as ≈, flagged), "
          f"{(out['price'].isna() & out['price_estimate'].isna()).sum():,} still unknown")
    print(f"gen_description: {(out['gen_description'] != '').sum():,} rows")
    logger.info("assemble -> %s (%s rows)", ENRICHED_CSV, f"{len(out):,}")


# ------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stages", nargs="+", choices=["recover", "rules", "price", "llm", "assemble"])
    p.add_argument("--limit", type=int, default=None, help="llm: stop after N rows in total (pilot).")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--max-new-tokens", type=int, default=48)
    p.add_argument("--scope", choices=["title-only", "all"], default="title-only",
                   help="llm: 'title-only' skips rows whose specs already give real facts.")
    p.add_argument("--model", default=LLM_MODEL)
    p.add_argument("--k", type=int, default=10, help="price: neighbours per estimate.")
    p.add_argument("--min-sim", type=float, default=0.90, help="price: min mean neighbour similarity.")
    a = p.parse_args()

    df = pd.read_csv(BASE_CSV, low_memory=False)
    logger.info("catalog: %s rows from %s", f"{len(df):,}", BASE_CSV)
    for stage in a.stages:
        t0 = time.perf_counter()
        if stage == "recover":
            stage_recover(df)
        elif stage == "rules":
            stage_rules(df)
        elif stage == "price":
            stage_price(df, k=a.k, min_sim=a.min_sim)
        elif stage == "llm":
            stage_llm(df, a.limit, a.batch_size, a.model, a.threads, a.max_new_tokens, a.scope)
        elif stage == "assemble":
            stage_assemble(df)
        logger.info("stage %s took %.0fs", stage, time.perf_counter() - t0)


if __name__ == "__main__":
    main()
