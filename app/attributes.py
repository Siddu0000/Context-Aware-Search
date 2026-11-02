"""Per-department attribute facets for the filter rail (2026-10-27, Oct 27 ask 5): an
Electronics page filters by Category / Item type / Works with / Condition, a Grocery
page by Category / Dietary.

Before this, every refinement past Department, Reviews, Price, Brands and Colour was
Fashion-only (product_type's columns), so an Electronics or Grocery pool had nothing
to narrow by. One registry entry = one rail section = one declared csv /search param
(`category`, `item_type`, `works_with`, `condition`, `dietary`). The UI draws whatever
`facets.attributes` describes for the focus department, so a new section needs a
registry entry and its /search param, no UI code.

- Values are a FIXED vocabulary (app/catalog_taxonomy.py's keys and labels), never
  seller text: an unknown value is a 422 before any LLM call, and every label is ours.
- The columns are derived at load by catalog_taxonomy.annotate_catalog (both backends;
  never embedded, never written to the CSV), so no re-embed and no 01/02 re-run.
- One param may serve two departments with different vocabularies (`category`:
  Electronics shop groups, Grocery categories). Their keys are disjoint
  (tests/test_api_filters.py): a key alone says which department it filters, and the
  UI remembers a label by its key.
- A row of another department, or with no value, never matches (as `ptype`).
  match "any" keeps a row with ANY ticked value; "all" (dietary) one with EVERY ticked
  value -- a shopper's restrictions all apply. Sections AND together, as every filter.

Pure (no pandas, no LLM, no I/O). app/main.py calls parse_filters (validation),
passes (_passes, post-cache: 0 LLM calls), facet_sections (_facets) and brand_name
(the Brands facet and the `store` filter).
"""

from collections import Counter
from typing import Callable, Iterable, NamedTuple, Optional

from app import catalog_taxonomy as ct


class Facet(NamedTuple):
    param: str             # the /search csv param, and the section's id in the rail
    label: str             # the section heading
    department: str        # the vertical whose rows carry it (bsns_vrtcl_name)
    column: str            # the load-time column it reads (a str, or a tuple of keys)
    vocab: dict            # key -> label: the whole vocabulary, in display order
    match: str = "any"     # "any": a row with ANY ticked key; "all": with EVERY one
    hint: str = ""         # the heading's tooltip
    key_of: Optional[dict] = None   # column value -> key (rail groups); None: the value


# Item type: what an Electronics row IS. "service" (protection plans, 396 rows) is
# never offered: nobody shops the rail for one
_ITEM_TYPES = ("device", "accessory", "part", "consumable")
# Both disclaimers say what the data is: title words, and enrichment regexes with no
# negation handling (catalog_taxonomy DIETARY; .claude/rules/safety.md "no claim the
# data cannot back": gluten and nut free are allergy-relevant)
_AS_WORDED = "As the listing words it"
_DIET_HINT = "As the listing words it, not a certification"

FACETS = (
    # The fine Electronics keys merge into shop groups ("computers" + "monitors" ->
    # "Computers & monitors"); "service" is in no group, so it is no value
    Facet("category", "Category", ct.ELECTRONICS, ct.CATEGORY_COLUMN,
          {key: label for key, (label, _keys) in ct.ELECTRONICS_RAIL_GROUPS.items()},
          key_of=dict(ct.ELECTRONICS_GROUP_OF)),
    Facet("item_type", "Item type", ct.ELECTRONICS, ct.ROLE_COLUMN,
          {role: ct.ROLE_LABELS[role] for role in _ITEM_TYPES}),
    Facet("works_with", "Works with", ct.ELECTRONICS, ct.WORKS_WITH_COLUMN,
          dict(ct.WORKS_WITH_LABELS)),
    Facet("condition", "Condition", ct.ELECTRONICS, ct.CONDITION_COLUMN,
          dict(ct.CONDITION_LABELS), hint=_AS_WORDED),
    Facet("category", "Category", ct.GROCERY, ct.CATEGORY_COLUMN,
          {key: label for key, (label, _keys) in ct.GROCERY_RAIL_GROUPS.items()},
          key_of={k: group for group, (_label, keys) in ct.GROCERY_RAIL_GROUPS.items()
                  for k in keys}),
    Facet("dietary", "Dietary", ct.GROCERY, ct.DIET_COLUMN,
          {value: ct.DIETARY_LABELS[value] for value in ct.DIETARY_VALUES},
          match="all", hint=_DIET_HINT),
)
PARAMS = tuple(dict.fromkeys(f.param for f in FACETS))
_BY_PARAM = {p: tuple(f for f in FACETS if f.param == p) for p in PARAMS}
# casefolded department -> ((registry position, facet), ...): facet_sections reads a row
# only against its own department's sections
_BY_DEPARTMENT = {dept: tuple((i, f) for i, f in enumerate(FACETS)
                              if f.department.casefold() == dept)
                  for dept in {f.department.casefold() for f in FACETS}}
# A param's csv cap, on the joined csv (a repeated param counts in all), checked by
# parse_filters for GET /search (a 422) and plain callers alike: a list param gets no
# per-value Query(max_length=). It is above the 300 of the other csv params because
# every key a param accepts, sent at once, must fit (tests): `category` holds both
# departments' keys
# (21 Electronics groups + 18 Grocery categories, 394 characters as a csv), so no valid
# request is ever refused for its length and no section needs a max_picked
MAX_CHARS = 500
MAX_VALUES_SHOWN = 30
PRIMARY_SHARE = 0.5


def vocabulary(param: str) -> dict:
    """Every key `param` accepts -- the union over its departments -- with its label."""
    out = {}
    for f in _BY_PARAM.get(param, ()):
        out.update(f.vocab)
    return out


def _fold(value) -> str:
    return " ".join(value.casefold().split()) if isinstance(value, str) else ""


def parse_filters(raw: dict) -> dict:
    """{param: csv string, list of csv strings or None} -> {param: [keys]} for the params
    that are set, each key casefolded, whitespace-collapsed and listed once (the order
    sent). A list is the REPEATED form (GET /search?dietary=vegan&dietary=keto) and
    means its comma-join, so both spellings filter alike. Raises ValueError (main.py: a
    422, before any cache read or LLM call) for an unknown key -- one only another
    param knows included --, a value over MAX_CHARS (the joined csv: the only cap a
    plain caller meets) or anything but strings. A param of no registry entry is a
    caller bug, also a ValueError."""
    active = {}
    for param, value in raw.items():
        if param not in _BY_PARAM:
            raise ValueError(f"{param} is not an attribute filter")
        if value is None:
            continue
        if isinstance(value, (list, tuple)) and all(isinstance(v, str) for v in value):
            value = ",".join(value)
        if not isinstance(value, str):
            raise ValueError(f"{param} must be a comma-separated string or a list of them")
        if len(value) > MAX_CHARS:
            raise ValueError(f"{param} is longer than {MAX_CHARS} characters")
        keys = list(dict.fromkeys(k for k in (_fold(v) for v in value.split(",")) if k))
        allowed = vocabulary(param)
        unknown = [k for k in keys if k not in allowed]
        if unknown:
            raise ValueError(
                f"Unknown {param} {unknown!r}; expected one of {', '.join(allowed)}")
        if keys:
            active[param] = keys
    return active


def row_keys(row: dict, facet: Facet, department: Optional[str]) -> frozenset:
    """The facet's keys on this row: empty for a row of another department or with no
    value. A tuple / list column (works_with, diet_tags; a list once it went through
    JSON) holds several; a comma-separated string is read the same way. Anything
    outside the vocabulary is dropped (a fixed vocabulary, never seller text)."""
    if not department or _fold(department) != _fold(facet.department):
        return frozenset()
    return _keys(row, facet)


def _keys(row: dict, facet: Facet) -> frozenset:
    """row_keys for a row already known to be of the facet's department."""
    raw = row.get(facet.column)
    if isinstance(raw, str):
        raw = raw.split(",")
    elif not isinstance(raw, (list, tuple, set, frozenset)):
        return frozenset()          # None, NaN, absent
    keys = set()
    for value in raw:
        key = _fold(value)
        if facet.key_of is not None:
            key = facet.key_of.get(key)
        if key in facet.vocab:
            keys.add(key)
    return frozenset(keys)


def passes(row: dict, active: dict, department: Optional[str]) -> bool:
    """Whether the row passes every attribute filter in `active` (main.py's filter dict;
    keys that are not PARAMS are ignored). `department` = main._department(row)."""
    for param in PARAMS:
        wanted = active.get(param)
        if not wanted:
            continue
        facets = [f for f in _BY_PARAM[param] if _fold(f.department) == _fold(department)]
        have = set()
        for f in facets:
            have |= row_keys(row, f, department)
        if not have:
            return False            # another department's row, or no value
        if any(f.match == "all" for f in facets):
            if not set(wanted) <= have:
                return False
        elif not set(wanted) & have:
            return False
    return True


def facet_sections(rows: Iterable[dict], on_target: Callable[[dict], bool],
                   department: Callable[[dict], Optional[str]]) -> tuple:
    """(sections, primary_department) over the UNFILTERED pool, counted in ROWS.

    A value is offered only with an on-target row (main._on_target), like every other
    facet; `n` counts every row carrying it, so it is what ticking it alone returns.
    Values by count, then vocabulary order, at most MAX_VALUES_SHOWN (n_values = all
    offered); a section with none is left out. Sections keep the registry order, every
    department's: the UI shows the focus department's.

    primary_department = the department of at least PRIMARY_SHARE of the on-target
    rows (None when none has it; on an exact tie the department seen first in the
    pool's relevance order). The UI shows the primary department's sections, and
    Fashion's own only when it is Fashion or None: a pool 86% Grocery ("gift for a
    coffee lover") showed Fashion's Category and Style from its 10 Fashion rows."""
    counts = [Counter() for _ in FACETS]
    offered = [set() for _ in FACETS]
    departments = Counter()
    n_on = 0
    for row in rows:
        hit = on_target(row)
        name = department(row)
        if hit:
            n_on += 1
            if name:
                departments[name] += 1
        for i, f in _BY_DEPARTMENT.get(_fold(name), ()):   # only its own department's
            for key in _keys(row, f):
                counts[i][key] += 1
                if hit:
                    offered[i].add(key)
    sections = []
    for i, f in enumerate(FACETS):
        if not offered[i]:
            continue
        order = {key: n for n, key in enumerate(f.vocab)}
        keys = sorted(offered[i], key=lambda k: (-counts[i][k], order[k]))
        sections.append({
            "param": f.param, "label": f.label, "department": f.department,
            "match": f.match, "hint": f.hint,
            "values": [{"key": k, "label": f.vocab[k], "n": counts[i][k]}
                       for k in keys[:MAX_VALUES_SHOWN]],
            "n_values": len(keys),
        })
    primary = None
    if n_on and departments:
        top, n = departments.most_common(1)[0]   # stable: the first seen wins a tie
        if n / n_on >= PRIMARY_SHARE:
            primary = top
    return sections, primary


def brand_name(row: dict) -> Optional[str]:
    """The name the Brands facet lists and the `store` filter matches: the row's `brand`
    (catalog_taxonomy.brand_of: the store, or the maker the specs name where the store
    is a chip vendor -- "intel" sold a Lenovo laptop, so "laptop for college" listed
    intel, amd and nvidia as brands), else its `store`. Off Electronics `brand` IS the
    store, so Fashion and Grocery are unchanged. The card byline reads the same
    (ui_common.brand_name), so a ticked brand keeps the cards that print it."""
    brand = row.get("brand")
    if isinstance(brand, str) and brand.strip():
        return brand
    store = row.get("store")
    return store if isinstance(store, str) else None
