"""Exclusions the shopper STATED: "a summer dress that isn't floral", "not black", "no
sequins or lace", "isn't too short" -- rows the pool must lose before anything ranks it.

2026-11-19, the Databricks App review (query bank tab 2): "Find me a summer dress that
isn't floral, isn't too short, and is suitable for a brunch date" served floral dresses.
An embedding cannot read "not" (a floral sundress IS close to that text), and the reranker
caps a violator only inside its RERANK_INPUT_K window: the rest of the pool is shown
unjudged. Reproduced locally (gte-small, qwen3.8-27b): 7 of the 15 window rows were floral
(the reranker held them off page 1) and 7 floral dresses sat on pages 2-3 from the
unjudged tail. The catalog has the supply: 2,553 of 13,539 Fashion dress titles say
"floral" and 466 "flower", the other ~10,500 neither.

- requested(query) -> frozenset of keys, read from a CLOSED vocabulary (patterns, colours,
  materials, trims, cuts, lengths) and only right after a negation cue ("isn't", "not",
  "no", "non-", "without", "nothing", "avoid", "anything but", "don't want" ...), with
  "or" / "and" / "," lists ("no sequins or lace"). Anything else is left to the reranker:
  "no added sugar", "non-stick", "non-alcoholic", "not flagship-priced", "no-show socks"
  read nothing. "too short" is the mini length, "too long" the maxi one.
- matches(row, keys): the row's title (or colour / material column) shows an excluded key.
  Grocery rows never match: a food negation is a dietary constraint the reranker judges
  ("no peanuts" must not drop "Peanut-Free" snacks).
- constraint(keys): the deterministic text the reranker is sent (never the shopper's words).

Pure re (no pandas, no I/O, no LLM), English-only and hand-maintained like product_type.py
and the query guard; tables in tests/test_exclusions.py. main.py acts on it inside
TYPE_AWARE_RANKING, flat pages only (a bundle's components are the translator's list).
"""

import re
from functools import lru_cache
from typing import FrozenSet, Optional

# key -> (query words, title words, what the reranker is told). Query words are read only
# after a negation cue; title words decide a row. Every pattern is wrapped in \b...\b.
_TERMS = {
    # patterns
    "floral": (r"florals?|flowers?|flowered|flowery|botanicals?",
               r"florals?|flowers?(?![\s-]+girls?)|flowered|flowery|botanicals?|blossoms?|"
               r"dais(?:y|ies)|ditsy|hibiscus|sunflowers?|roses|rose prints?",
               "floral prints (flowers, botanical, blossom)"),
    "striped": (r"stripes?|striped|pinstripes?|pinstriped",
                r"stripes?|striped|pinstripes?|pinstriped", "stripes"),
    "plaid": (r"plaids?|tartans?|gingham|checks|checked|checkered|chequered",
              r"plaids?|tartans?|gingham|checked|checkered|chequered", "plaid or check patterns"),
    "polka_dot": (r"polka[\s-]?dots?|dotted|dots",
                  r"polka[\s-]?dots?|dotted|dot prints?", "polka dots"),
    "animal_print": (r"animal prints?|leopard(?: prints?)?|cheetah|zebra|snake ?skin|"
                     r"snake prints?|tiger prints?|cow prints?",
                     r"animal prints?|leopard|cheetah|zebra|snake ?skin|snake prints?|"
                     r"tiger prints?|cow prints?", "animal prints (leopard, zebra, snake)"),
    "camo": (r"camo|camouflage", r"camo|camouflage", "camouflage"),
    "tie_dye": (r"tie[\s-]?dyed?", r"tie[\s-]?dyed?", "tie-dye"),
    "paisley": (r"paisley", r"paisley", "paisley"),
    "graphic": (r"graphics?|graphic prints?|logos?|slogans?",
                r"graphics?|logos?|slogans?", "graphics, logos or slogans"),
    "patterned": (r"prints?|printed|patterns?|patterned",
                  r"prints?|printed|patterns?|patterned|florals?|flowers?(?![\s-]+girls?)|"
                  r"stripes?|striped|plaids?|tartans?|gingham|checkered|polka[\s-]?dots?|"
                  r"leopard|zebra|camo|camouflage|paisley|tie[\s-]?dyed?|geometric",
                  "any print or pattern (solid colours only)"),
    # trims
    "sequin": (r"sequins?|sequined|sequinned|glitter|glittery|sparkly|sparkles?",
               r"sequins?|sequined|sequinned|glitter|glittery|sparkly|sparkles?",
               "sequins or glitter"),
    "lace": (r"lace|lacy", r"lace(?![\s-]*ups?)|lacy", "lace"),
    "ruffle": (r"ruffles?|ruffled|frills?|frilly", r"ruffles?|ruffled|frills?|frilly", "ruffles"),
    "fringe": (r"fringes?|fringed|tassels?", r"fringes?|fringed|tassels?", "fringe or tassels"),
    "embroidered": (r"embroidery|embroidered", r"embroidery|embroidered", "embroidery"),
    "beaded": (r"beads?|beaded|rhinestones?", r"beads?|beaded|rhinestones?",
               "beading or rhinestones"),
    "hooded": (r"hoods?|hooded", r"hoods?|hooded|hoodies?", "hoods"),
    # cuts
    "sleeveless": (r"sleeveless", r"sleeveless", "sleeveless cuts"),
    "strapless": (r"strapless", r"strapless", "strapless cuts"),
    "backless": (r"backless|open[\s-]back", r"backless|open[\s-]back", "backless cuts"),
    "off_shoulder": (r"off[\s-](?:the[\s-])?shoulders?|cold[\s-]shoulders?",
                     r"off[\s-](?:the[\s-])?shoulders?|cold[\s-]shoulders?",
                     "off-shoulder cuts"),
    "cropped": (r"crop|cropped|crop tops?", r"crop|cropped", "cropped cuts"),
    "low_cut": (r"low[\s-]cut|plunging|plunge|deep[\s-]v(?:[\s-]neck)?",
                r"low[\s-]cut|plunging|plunge|deep[\s-]v(?:[\s-]neck)?",
                "low-cut or plunging necklines"),
    "sheer": (r"see[\s-]through|sheer|mesh|transparent", r"see[\s-]through|sheer|mesh",
              "sheer, mesh or see-through fabric"),
    # lengths: "isn't too short" / "not too long" (never a bare "short": "short sleeve")
    "mini": (r"(?:too|so|very|overly)[\s-]+short(?![\s-]+sleeves?)|minis?|micro",
             r"mini|micro", "mini length (too short)"),
    "maxi": (r"(?:too|so|very|overly)[\s-]+long(?![\s-]+sleeves?)|maxis?|floor[\s-]length",
             r"maxi|floor[\s-]length", "maxi or floor length"),
}

# Colours: a title word, the seller's colour column or the load-time colour family
_COLOURS = ("black", "white", "red", "blue", "navy", "green", "yellow", "pink", "purple",
            "orange", "brown", "grey", "gray", "beige", "khaki", "gold", "silver", "neon")
for _c in _COLOURS:
    if _c == "gray":
        continue
    _TERMS[_c] = (r"gr[ae]y" if _c == "grey" else _c,
                  r"gr[ae]y" if _c == "grey" else (r"gold(?:en)?" if _c == "gold" else _c),
                  "the colour " + _c)

# Materials: a title word or the one-value material column
_MATERIALS = {
    "leather": r"leather", "suede": r"suede", "wool": r"wool|woolen|woollen",
    "polyester": r"polyester|poly", "nylon": r"nylon", "acrylic": r"acrylic",
    "silk": r"silk|silky", "satin": r"satin", "velvet": r"velvet|velour",
    "denim": r"denim", "fur": r"fur|furry", "fleece": r"fleece", "cotton": r"cotton",
    "linen": r"linen", "spandex": r"spandex|elastane|lycra",
}
for _m, _p in _MATERIALS.items():
    _TERMS[_m] = (_p, _p, _m)
_MATERIAL_KEYS = frozenset(_MATERIALS)
_COLOUR_KEYS = frozenset(c for c in _COLOURS if c != "gray")
KEYS = frozenset(_TERMS)

# A negation, read as a whole word: what follows it is what the shopper does NOT want.
# "non" also reads "non-floral" (the hyphen is a separator below)
_CUE_RE = re.compile(
    r"\b(?:is ?n'?t|is not|are ?n'?t|are not|was ?n'?t|was not|not|no|non|without|w/o|"
    r"nothing|never|avoid(?:ing)?|except(?: for)?|other than|anything but|but not|minus|"
    r"free of|free from|don'?t want|do not want|don'?t like|do not like|hate|dislike|"
    r"no more)\b")
# Words between a cue and the term ("nothing in black", "without any lace", "not made of
# polyester", "isn't too floral"): skipped, any number of them
_SEP_RE = re.compile(r"[\s-]*")
_FILLER_RE = re.compile(
    r"(?:[\s-]+(?:a|an|any|the|in|with|of|made of|made from|like|that|those|these|anything|"
    r"something|being|too|so|very|overly|really|super|kind of|sort of)\b)*[\s-]+")
# A list after the first term: "no sequins or lace", "not floral, striped or plaid", "no
# leather and no suede"
_JOIN_RE = re.compile(r"\s*(?:,|/|&|\+|\bor\b|\band\b|\bnor\b)\s*(?:(?:no|non|not)\b)?")
_TERM_RE = re.compile(
    "|".join(rf"(?P<{key}>\b(?:{words})\b)" for key, (words, _, _) in _TERMS.items()))
_TITLE_RE = {key: re.compile(rf"\b(?:{title})\b") for key, (_, title, _) in _TERMS.items()}


def _norm(text) -> str:
    """Casefolded, curly apostrophe straightened, whitespace collapsed; non-str -> ''."""
    if not isinstance(text, str):
        return ""
    return " ".join(text.casefold().replace("’", "'").split())


def _term_at(q: str, pos: int) -> Optional[re.Match]:
    """The vocabulary term at `pos` -- right after the separator ("too short" is a term
    of its own, before "too" can be skipped) or after the filler words ("too floral" is
    "floral", "nothing in black" is "black")."""
    m = _TERM_RE.match(q, _SEP_RE.match(q, pos).end())
    if m:
        return m
    gap = _FILLER_RE.match(q, pos)
    return _TERM_RE.match(q, gap.end()) if gap else None


@lru_cache(maxsize=4096)
def requested(query) -> FrozenSet[str]:
    """The keys the shopper said they do NOT want, from a closed vocabulary; empty when
    none. Pure and cached (the query text is never logged: the caller logs counts)."""
    q = _norm(query)
    if not q:
        return frozenset()
    keys = set()
    for cue in _CUE_RE.finditer(q):
        pos = cue.end()
        while True:
            term = _term_at(q, pos)
            if term is None:
                break
            keys.add(term.lastgroup)
            pos = term.end()
            join = _JOIN_RE.match(q, pos)
            if not join or join.end() == pos:
                break
            pos = join.end()
    return frozenset(keys)


def _text(value) -> str:
    return value.casefold() if isinstance(value, str) else ""


def matches(row: dict, keys) -> bool:
    """Whether the row shows one of `keys`: its title or colour column (a colour key also
    its load-time colour family, a material key also its material column). Grocery rows
    never match."""
    if not keys:
        return False
    if _text(row.get("bsns_vrtcl_name")).strip() == "grocery":
        return False
    hay = _text(row.get("Product_title")) + " | " + _text(row.get("color"))
    for key in keys:
        if key not in _TITLE_RE:
            continue
        if _TITLE_RE[key].search(hay):
            return True
        if key in _MATERIAL_KEYS and _TITLE_RE[key].search(_text(row.get("material"))):
            return True
        if key in _COLOUR_KEYS and _text(row.get("color_family")) == key:
            return True
    return False


def constraint(keys) -> dict:
    """{'type': 'exclude', 'value': ...}: what the reranker is told, from the vocabulary's
    own labels in a fixed order (never the shopper's words)."""
    labels = [_TERMS[k][2] for k in _TERMS if k in keys]
    return {"type": "exclude",
            "value": ("NOT " + "; NOT ".join(labels)
                      + ": a product that shows any of them violates it")}
