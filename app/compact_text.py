"""Compact embedding text for rows the local embedder would cut (2026-11-20).

Each product is embedded as ONE string (app/search.py `_build_search_text`), and the local
embedder reads only its first EMBED_MAX_SEQ_LEN tokens (128 for gte-small: 126 of
content). On the enriched catalog 40% of rows were longer, and the cut fell on the TAIL:
the description and then the enriched tags (colour, material, use_case, dietary, specs)
were dropped mid-word. This module rewrites ONLY such rows, by rules, so that every
attribute fits and the description keeps its most informative sentences:

1. department and category (once), then the title without the parts that only repeat it;
2. every tag value (colour, material, occasion, gender, use_case, dietary) that says
   something new -- for a comma list, only its new parts;
3. spec values that say something new (the key only where the value needs it: a bare
   number, a code, a value whose key gives it its role -- 'allergen information dairy';
   the key alone for a yes flag; "compatible with ..." for device lists), minus packaging
   noise, colours that are no colour and seller warnings;
4. description segments (sentences and feature bullets) in their own order, without
   seller boilerplate (a run too long to drop whole loses only the words around it; a
   short one keeps a number + unit or a fabric blend stated in front of its phrase),
   repeated runs of words, or segments mostly already said, while they fit; one last
   segment that does not fit is clipped at a word boundary (never ending on a stop word,
   a number cut from its unit or a word cut from its 'free'), and after that only
   complete sentences are added; with room left, a sentence that only looked said
   ('the same hot sauce ... zesty jalapeno flavor') adds its new words.

Extractive and deterministic: it never writes a word the listing did not contain (it
only removes words and symbols, and adds "compatible with" before a device list). No
model: the caller passes `count_tokens`. scripts/compact_search_text.py runs it once
over the catalog and writes the sidecar search.load_index reads when SEARCH_TEXT_COMPACT
is on; the sidecar and cache helpers at the end hash files, nothing else does I/O. A row
that already fits keeps its text byte for byte, so its cached vector is reused. Bump
COMPACT_VERSION when a released output changes: it names the sidecar and the cache.
"""

import hashlib
import re
import unicodedata
from pathlib import Path
from typing import Callable, List, Mapping, Optional, Sequence, Set, Tuple

COMPACT_VERSION = "1"

HEAD_FIELDS = ("bsns_vrtcl_name", "categ_lvl2_name")
TAG_FIELDS = ("color", "material", "occasion", "gender", "use_case", "dietary")

_WORD_RE = re.compile(r"[a-z0-9]+")
_WS_RE = re.compile(r"\s+")

# Emoji, dingbats, arrows, geometric and other symbols. Most are [UNK] to gte-small, and a
# glyph glued to a word ('💖material') turns the whole word into one [UNK]; the rest
# (★ ♥ ● √ ...) are decoration. They become spaces; in a description they are bullets.
# Outside the range: '°', '™', '℃', and the bounds '≤' / '≥', which gte-small reads as
# tokens and which say something ('≤ 5 v' is not '5 v'): _clean spaces them out instead,
# since glued ('cri≥95') they split the words around them into pieces.
_SYMBOLS = "←-≣≦-⯿\U0001f000-\U0001faff️‍⃣"
_SYMBOL_RE = re.compile(f"[{_SYMBOLS}]+")
# A key's ':' right before a bound goes too ('insertion loss: ≤0.3db' -> 'insertion loss ≤
# 0.3db'); a ':' between two figures stays ('swr ≤ 2: ≥ 1300 khz')
_BOUND_RE = re.compile(r"(?:(?:(?<=[^\W\d_])|(?<=\)))\s*:)?\s*([≤≥])\s*")

# A description is Amazon's feature bullets and description text joined with spaces
# (scripts/load_amazon_data.py), so most bullet boundaries carry no full stop. Split on
# sentence ends, ';', bullet glyphs, an opening 【 / [ and an ALL-CAPS bullet header
# ("SOFT FABRIC: ...") -- in the ORIGINAL case, before lowercasing. A header starts after
# a lowercase word, a digit or punctuation, never inside the header itself
# ("MINERALS AND ELECTROLYTES FOR TASTE:" is one header). A closing 】 / ] becomes a
# space, so a bracketed header stays with the text it heads.
_SEGMENT_RE = re.compile(
    r"(?<=[.!?])\s+"
    r"|\s*;\s*"
    r"|\s*[•|" + _SYMBOLS + r"]+\s*"
    r"|\s*[【\[]\s*"
    r"|(?<=[a-z0-9.!?)])\s+(?=[A-Z][A-Z0-9&/'+ -]{2,40}:)"
)
_CLOSE_RE = re.compile(r"\s*[】\]]\s*")
# a leading dash bullet goes, a minus sign before a number stays ('-40°C to 85°C')
_LEAD_RE = re.compile(r"^[\s:,;]*(?:[-–—]+(?!\d)\s*)?[\s:,;]*")

# Title parts: Amazon titles repeat themselves across '|' / ';' / ',' / ' - ' parts. A
# comma between two digits is a number ('30,000', '4,5,6'), never a part boundary.
_TITLE_SPLIT_RE = re.compile(r"(\s*\|\s*|\s*;\s*|(?<!\d)\s*,\s*|\s*,\s*(?!\d)|\s+-\s+)")
_LONE_CHAR_RE = re.compile(r"(?<![\w'’])[a-z0-9](?![\w'’])")

_STOP = frozenset(
    "a an and are as at be by can for from has have in into is it its of on or our that the "
    "their this to was with you your will which these those than then also more most very "
    "just all any each other such only so".split()
)
# Fused with the next word for 'already said?' checks: 'no added sugar' is not 'added
# sugar', 'not braided' is not 'braided'
_NEGATORS = frozenset("no not non without zero never".split())
_POSSESSIVE_RE = re.compile(r"['’]s\b")

# Number + unit, written one way for 'already said?' checks: '512gb' = '512 GB',
# '12 fl oz' = '12 fluid ounces', '3.70 ml' = '3.7 milliliters'
_UNITS = (
    (r"ghz|gigahertz", "ghz"), (r"mhz|megahertz", "mhz"), (r"khz|kilohertz", "khz"),
    (r"hz|hertz", "hz"),
    (r"fl\.?\s*oz|fluid\s+ounces?", "floz"), (r"ml|millilit(?:er|re)s?", "ml"),
    (r"l|lit(?:er|re)s?", "l"), (r"oz|ounces?", "oz"), (r"lbs?|pounds?", "lb"),
    (r"mg|milligrams?", "mg"), (r"kg|kilograms?", "kg"), (r"g|grams?", "g"),
    (r"tb|terabytes?", "tb"), (r"gb|gigabytes?", "gb"), (r"mb|megabytes?", "mb"),
    (r"mah", "mah"), (r"w|watts?", "w"), (r"v|volts?", "v"),
    (r"in|inch(?:es)?|\"|”", "in"), (r"ft|feet|foot", "ft"),
    (r"cm|centimet(?:er|re)s?", "cm"), (r"mm|millimet(?:er|re)s?", "mm"),
)
_UNIT_RE = re.compile(
    r"(?<![a-z0-9.])(\d+(?:\.\d+)?|\.\d+)\s*-?\s*("
    + "|".join(f"(?:{p})" for p, _ in _UNITS) + r")(?![a-z0-9])")
_UNIT_CANON = [(re.compile(f"(?:{p})"), canon) for p, canon in _UNITS]
# A frequency also keeps a bare number key: '144hz' in a title and 'refresh rate 144 hz'
# in a spec are one fact, and a text that says '1600' alone still says it. A decimal one
# keeps it fused ('2.4ghz' -> '2p4ghz', '2p4'), never '2' and '4', which hid a '4 ports'
# spec. Measured (review round 4): a reshuffle, not a net gain -- 246 over-budget rows
# change, 212 of them Computers; the port and processor counts it stops hiding push the
# description's tail out (distinct content words +478 / -489)
_BARE_TOO = frozenset({"ghz", "mhz", "khz", "hz"})
# '20,000 mah' = '20000 mah' (else every 'N,000 mah' read as '0mah'); never inside a
# decimal or a list ('usb 3.0,100w', '13,12,11')
_THOUSANDS_RE = re.compile(r"(?<![\d.,])\d{1,3}(?:,\d{3})+(?!\d)(?!,\d)")
# What may follow a number in the source, so that a clip never ends on the bare number
_TRAIL_UNIT_RE = re.compile(
    "(?:" + "|".join(p for p, _ in _UNITS)
    + r"|hours?|hrs?|minutes?|mins?|seconds?|secs?|days?|weeks?|months?|years?"
    r"|pcs|pieces?|packs?|count|ct|percent|%)", re.I)
# A spec value of 'N count' (Amazon's Unit Count), dropped when N is already said
_COUNT_VALUE_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(?:count|ct)\s*$", re.I)

# Seller boilerplate: service promises, trademark lines, ordering prompts, program
# explainers. A short segment saying one of these goes whole; in a long segment only the
# clauses saying it go (a 50-word spec run that ends in 'money back guarantee' keeps its
# specs).
_BOILERPLATE_RE = re.compile(
    r"registered trademarks?|all rights reserved|buy with confidence"
    r"|if you (?:have|are not|encounter) any"
    r"|feel free to (?:contact|reach|ask|message|email|let us)"
    r"|(?:do not|don'?t) hesitate"
    r"|\bcontact us\b|customer (?:service|support)|we will (?:reply|respond|get back)"
    r"|within 24 hours|money[- ]back|100% satisf|satisfaction guarantee"
    r"|\badd to cart\b|\border now\b|\bbuy now\b"
    r"|\bclick (?:the |on (?:the )?)?(?:add to cart|buy|order)\b"
    r"|\bclick (?:on )?(?:the )?[\"“]?(?:name of our|our (?:shop|store)|seller|brand name"
    r"|customi[sz]e)|\babove (?:the )?title\b|feel free to (?:talk|chat) with us"
    r"|always check the actual product label|amazon renewed|amazon-qualified"
    r"|\bairmail\b|\bestimated delivery\b|\bshipping information\b|\bdate of shipment\b"
    r"|^please\b|size chart"
    # (F2 covers 25 of F1's 45 rows above; F1 still changes 20 more: review round 4)
    r"|customi[sz]e now|ask a question|name of (?:our|the) (?:shop|store)"
    r"|contact (?:the )?seller(?! to (?:verify|confirm|check))"
    r"|give us a call|more for the money|(?:purchase|shop) with confidence"
    r"|we stand behind our products"
)
_BOILERPLATE_CI_RE = re.compile(_BOILERPLATE_RE.pattern, re.I)
_BOILERPLATE_MAX_WORDS = 25
_CLAUSE_RE = re.compile(r"(?<=[,.;!?])\s+")
# In a clause too long to drop whole (a bullet run with no ', '), only the words around
# each boilerplate phrase go: this many before it and after it
_CUT_BEFORE, _CUT_AFTER = 3, 8

# Spec items that are packaging noise or misleading on most listings
_SPEC_DROP_KEYS = frozenset({"other display features"})
_COUNT_KEYS = frozenset({"unit count", "number of items", "number of pieces",
                         "item package quantity"})
_ONE_COUNT_RE = re.compile(r"^1(?:\.0+)?\s*(?:count|ct)?$", re.I)
_BOOL_RE = re.compile(r"^(?:yes|no|true|false|n/?a)$", re.I)
_YES_RE = re.compile(r"^(?:yes|true)$", re.I)
# For a yes flag the KEY is the fact ('Is Waterproof: Yes' -> 'waterproof'), except on
# these keys, where 'Yes' says nothing
_YES_JUNK_KEYS = frozenset({
    "brand", "warranty description", "certification", "size", "included components",
    "is product expirable", "contains liquid contents", "connectivity technology"})
_FLAG_PREFIX_RE = re.compile(r"^(?:is|has|supports)\s+", re.I)
_COMPAT_KEYS = frozenset({"compatible devices", "compatible phone models"})
# A compat value that is a seller warning ('Not for iPhone 12', 'should check ...') is
# no compatibility claim
_COMPAT_WARNING_RE = re.compile(r"\b(?:should|not|won'?t|doesn'?t|except)\b", re.I)
# Keys that give the value its role: 'dairy' alone reads as an ingredient, 'dvd' on a
# router as the product, '240 volts' twice as one figure
_ROLE_KEYS = frozenset({"allergen information", "input voltage", "output voltage",
                        "included components"})
_COLOR_KEYS = frozenset({"color", "colour", "color name", "colour name"})
# Colour values that are no colour (the tag and the spec alike)
_NO_COLOR_RE = re.compile(
    r"(?:one|random|picture)?\s*colou?rs?|default|as shown|(?:see|as) (?:the )?pictures?"
    r"|random|no colou?r|(?:the )?same as (?:the |our )?pictures?(?: shown?)?|as pictured"
    r"|colou?r ?\d+")
_JUNK_VALUE_RE = re.compile(r"attribute not applicable", re.I)
# 'Included Components' that name nothing: a count, 'not null', or paperwork alone
# (a count of 1-3 digits; a longer number is a part number)
_COMPONENT_JUNK_RE = re.compile(r"^(?:\d{1,3}(?:\.\d+)?|(?:not )?null|none|n/?a|inc|items?)$")
_DOC_WORDS = frozenset(
    "user users owner owners owner's owners' manual manuals instruction instructions "
    "instructional warranty card cards guide guides quick start booklet booklets operating "
    "installation operation and & paper".split())
_DOC_ANCHORS = frozenset(
    "manual manuals instruction instructions instructional warranty guide guides booklet "
    "booklets".split())
# 'N% <fabric or material>': a blend is a fact even inside a boilerplate sentence
_BLEND_RE = re.compile(                 # (?<!\d): linear on a long run of digits
    r"(?<!\d)\d+(?:\.\d+)?\s*%\s*(?:cotton|polyester|poly|spandex|elastane|lycra|nylon|polyamide|wool"
    r"|merino|cashmere|alpaca|mohair|silk|linen|hemp|jute|rayon|viscose|modal|lyocell|tencel"
    r"|acrylic|bamboo|leather|suede|denim|fleece|microfiber|vinyl|pvc|polypropylene"
    r"|polyurethane|pu|silicone|rubber|latex|plastic|metal|steel|stainless|aluminum|aluminium"
    r"|titanium|copper|brass|gold|silver|sterling|ceramic|glass|wood|natural|organic|pure)\b")
# A size-tolerance note before a boilerplate phrase states no fact ('there may be 1-3cm
# error', 'it allows 1 to 3cm deviation')
_TOLERANCE_RE = re.compile(
    r"^\W*(?:there\s+may\s*be|it\s+allows|(?:measurement\s+)?(?:errors?|deviations?)\b)", re.I)
# A full stop or '!' / '?' with no space after it, which the segmenter does not split
_TIGHT_BREAK_RE = re.compile(r"(?<=[a-z)][.!?])(?=[A-Z])")
_FACT_KEY_RE = re.compile(r"^\s*(?:materials?|fabrics?)\s*[:\-]\s*\w", re.I)
_KEY_QUALIFIER_RE = re.compile(r"\s*\([^)]*\)")

_RUN = 6  # a run of this many words already in the text is a repeat


def _blank(value) -> str:
    """str(value), '' for None and every NaN / NA (float, numpy, pandas)."""
    if value is None:
        return ""
    try:
        if value != value:             # NaN of any float type
            return ""
    except TypeError:                  # pd.NA: its truth value is ambiguous
        return ""
    text = str(value)
    return "" if text in ("<NA>", "NaT") else text


def _fold(text: str) -> str:
    """Lowercase with accents removed, as the uncased tokenizer reads it ('café' = 'cafe')."""
    decomposed = unicodedata.normalize("NFD", text.lower())
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def _clean(text: str) -> str:
    """The text as embedded: lowercase, symbols out, bounds spaced, single spaces."""
    text = _BOUND_RE.sub(r" \1 ", _SYMBOL_RE.sub(" ", _blank(text).lower()))
    return _WS_RE.sub(" ", text).strip()


def _stem(word: str) -> str:
    """Plural folding for 'already said?' checks, and nothing else: dresses / dress,
    shoes / shoe, sizes / size, batteries / battery, cookies / cookie, boxes / box --
    while cape / cap and note / not stay apart."""
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("ie"):
        return word[:-2] + "y"
    if len(word) > 4 and word.endswith(("sses", "xes", "ches", "shes", "zzes")):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def content_words(text: str) -> List[str]:
    """Words for the build's retention report: accents folded, stop words and single
    characters dropped, plurals folded."""
    return [_stem(w) for w in _WORD_RE.findall(_fold(_blank(text)))
            if len(w) > 1 and w not in _STOP]


def _unit(match) -> str:
    number = ("%f" % float(match.group(1))).rstrip("0").rstrip(".") or "0"   # '3.70' -> '3.7'
    unit = match.group(2)
    for rx, canon in _UNIT_CANON:
        if rx.fullmatch(unit):
            bare = f"{number.replace('.', 'p')} " if canon in _BARE_TOO else ""
            return f" {number.replace('.', 'p')}{canon} {bare}"
    return match.group(0)


def _fold_thousands(text: str) -> str:
    return _THOUSANDS_RE.sub(lambda m: m.group(0).replace(",", ""), text)


def said_keys(text: str) -> List[str]:
    """What a text says, for 'is this already in the text?': content words plus single
    letters and digits ('xbox series s' is not 'xbox series x'; 'usb ports 4'), a
    negator fused with the next word, number + unit written one way (thousands commas
    folded), possessive 's dropped, plurals folded."""
    t = _UNIT_RE.sub(_unit, _fold_thousands(_POSSESSIVE_RE.sub("", _fold(_blank(text)))))
    out: List[str] = []
    negated = False
    for w in _WORD_RE.findall(t):
        if w in _NEGATORS:
            negated = True
            continue
        if w in _STOP:
            continue
        key = _stem(w)
        out.append("no_" + key if negated else key)
        negated = False
    return out


def _norm_phrase(text: str) -> str:
    return " ".join(_WORD_RE.findall(_fold(text)))


def _phrase_in(phrase: str, text: str) -> bool:
    p = _norm_phrase(phrase)
    return bool(p) and f" {p} " in f" {_norm_phrase(text)} "


def dedup_title(title: str) -> str:
    """The title without the parts that only repeat it. Whole parts between '|', ';',
    ',' or ' - ' go, never single words (word-level dedup turned 'gluten free | dairy
    free' into 'gluten | dairy'). A part without a number goes when everything it says
    was said earlier ('not braided' and 'xbox series s' say something new); a part with
    a number only when the same phrase already stands in the title, number intact
    ('lamp 30 degree, 30,000 hours' keeps both), and never when it is only numbers.
    Nothing redundant: the title comes back byte for byte."""
    parts = _TITLE_SPLIT_RE.split(title)
    seen: Set[str] = set()
    out: List[str] = []
    for i in range(0, len(parts), 2):
        part = parts[i]
        keys = said_keys(part)
        # a number or a standalone letter ('galaxy tab a', 'usb-a'; never a possessive 's)
        # is judged by its exact phrase: the word test drops single letters and stop words
        if any(c.isdigit() for c in part) or _LONE_CHAR_RE.search(part):
            phrase = _WS_RE.sub(" ", part).strip()
            kept = _WS_RE.sub(" ", "".join(out))
            redundant = bool(re.search(r"[a-z]{2}", phrase)) and bool(re.search(
                r"(?<![\w.,+\-/])" + re.escape(phrase) + r"(?![\w.,+\-/%])", kept))
        else:
            redundant = bool(keys) and set(keys) <= seen
        if not redundant and part.strip():
            if out and i > 0:
                out.append(parts[i - 1])
            out.append(part)
        seen.update(keys)
    return "".join(out).strip()


def segments(description: str) -> List[str]:
    """Sentences and feature bullets of a description, in order, stripped. A part with no
    word of its own ('x 5 in.' after 'Dimensions: 10 in.', a lone emoji) joins the part
    before it, and so does a number cut at a decimal point ('13. 3"')."""
    text = _CLOSE_RE.sub(" ", _blank(description))
    out: List[str] = []
    for part in _SEGMENT_RE.split(text):
        part = _LEAD_RE.sub("", part or "").rstrip(" :,;")
        if not part:
            continue
        if out and (not content_words(part)
                    or (re.search(r"\d\.$", out[-1]) and part[:1].isdigit())):
            out[-1] = out[-1] + " " + part
        else:
            out.append(part)
    return out


def is_boilerplate(segment: str) -> bool:
    return bool(_BOILERPLATE_RE.search(_blank(segment).lower()))


def _cut_boilerplate(clause: str) -> str:
    """A clause too long to drop whole, minus _CUT_BEFORE words before and _CUT_AFTER
    words after each boilerplate phrase: Amazon bullets are joined with spaces, so a
    '... 3 Year Limited Warranty - Synergy Digital is a Registered Trademark Mini Battery
    ...' run is one clause, and dropping it whole lost the specs around the phrase."""
    spans = [(m.start(), m.group()) for m in re.finditer(r"\S+", clause)]
    keep = [True] * len(spans)
    for m in _BOILERPLATE_CI_RE.finditer(clause):    # offsets in the clause as written
        first = max((i for i, (s, _) in enumerate(spans) if s <= m.start()), default=0)
        last = max((i for i, (s, _) in enumerate(spans) if s < max(m.end(), m.start() + 1)),
                   default=first)
        hi = min(last + _CUT_AFTER + 1, len(spans))
        if hi - 1 > last and _norm_phrase(spans[hi - 1][1]) in _NEGATORS:
            hi -= 1                                  # keep 'not backwards compatible ...' whole
        for i in range(max(first - _CUT_BEFORE, 0), hi):
            keep[i] = False
    return " ".join(w for (_, w), k in zip(spans, keep) if k)


def _fact_prefix(text: str) -> str:
    """The words before a SHORT boilerplate unit's first phrase, kept when they state a
    number + unit or 'N% <fabric>' ('65% polyester, 35% cotton, 30 days money back' ->
    '65% polyester, 35% cotton'), ending on the last fact, without a dangling end. Not a
    size table before 'size chart' (two or more sizes: a blend still counts), nor a
    tolerance note ('there may be 1-3cm error')."""
    m = _BOILERPLATE_CI_RE.search(text)
    if not m:
        return ""
    if _TOLERANCE_RE.match(text[:m.start()]):
        return ""
    size_chart = (m.group().lower() == "size chart"
                  and len(_UNIT_RE.findall(_fold(text[:m.start()]))) >= 2)
    words = text[:m.start()].split()
    while words and (_norm_phrase(words[-1]) in _STOP or _norm_phrase(words[-1]) in _NEGATORS
                     or not _norm_phrase(words[-1])
                     or (words[-1].startswith("(") and ")" not in words[-1])):
        words.pop()
    pre = " ".join(words)
    # a trailing 'N days' / 'N%' is the guarantee's own lead-in ('30 days money back')
    pre = re.sub(r"(?:\s+\d+\s*(?:-\s*)?(?:days?|months?|years?|%))+\s*$", "", pre)
    folded = _fold(pre)
    if not ((((not size_chart) and _UNIT_RE.search(folded)) or _BLEND_RE.search(folded))
            and len(content_words(pre)) >= 2):
        return ""
    words = pre.split()

    def n_facts(ws):
        t = _fold(" ".join(ws))
        return (0 if size_chart else len(_UNIT_RE.findall(t))) + len(_BLEND_RE.findall(t))

    total = n_facts(words)
    while len(words) > 1 and n_facts(words[:-1]) == total:
        words.pop()
    pre = " ".join(words).rstrip(" ,;:")
    return pre if len(content_words(pre)) >= 2 else ""


def _fact_part(part: str) -> bool:
    folded = _fold(part)
    return bool((_UNIT_RE.search(folded) or _BLEND_RE.search(folded) or _FACT_KEY_RE.match(part))
                and len(content_words(part)) >= 2)


def _strip_short(unit: str) -> str:
    """A short boilerplate unit: its fact prefix -- or, when a full stop with no space
    after it hid a sentence break ('...Button,Two Styles ... Choose.Material: Soft'), the
    parts that state a fact (a number + unit, a blend, 'Material: ...')."""
    parts = [p for p in _TIGHT_BREAK_RE.split(unit) if p.strip()]
    if len(parts) < 2:
        return _fact_prefix(unit)
    kept = [_fact_prefix(p) if is_boilerplate(p) else (p if _fact_part(p) else "") for p in parts]
    return " ".join(k for k in kept if k) if any(kept) else _fact_prefix(unit)


def strip_boilerplate(segment: str) -> str:
    """A short boilerplate segment goes, but for a fact before its phrase (_fact_prefix); a
    long one loses only its boilerplate clauses, and a clause too long to drop whole only
    the words around its boilerplate."""
    if not is_boilerplate(segment):
        return segment
    if len(segment.split()) <= _BOILERPLATE_MAX_WORDS:
        return _strip_short(segment)
    kept = []
    for clause in _CLAUSE_RE.split(segment):
        if not is_boilerplate(clause):
            kept.append(clause)
        elif len(clause.split()) > _BOILERPLATE_MAX_WORDS:
            kept.append(_cut_boilerplate(clause))
        else:
            kept.append(_strip_short(clause))
    return " ".join(c for c in kept if c).strip()


def _adds_words(text: str, seen: Set[str], min_new: int = 1, max_seen_share: float = 1.0) -> bool:
    keys = said_keys(text)
    if not keys:
        return False
    new = [k for k in keys if k not in seen]
    return len(set(new)) >= min_new and (1 - len(new) / len(keys)) <= max_seen_share


def is_placeholder_colour(value) -> bool:
    """A colour value that names no colour: no Latin letter ('14', '#31-31') or a
    placeholder ('default', 'random color', 'see picture'). The colour tag and the colour
    spec both skip it; the build report counts it apart."""
    clean = _clean(value)
    return not re.search(r"[a-z]", clean) or bool(_NO_COLOR_RE.fullmatch(clean))


def _junk_component(part: str) -> bool:
    p = part.strip()
    if not p or _COMPONENT_JUNK_RE.match(p):
        return True
    words = re.findall(r"[a-z0-9'&]+", p)
    return bool(words) and all(w in _DOC_WORDS for w in words) and any(
        w in _DOC_ANCHORS for w in words)


def _spec_items(specs: str) -> List[Tuple[str, str]]:
    """'Key: value; Key: value' (scripts/enrich_catalog.py _specs_text) -> [(text to
    embed, value to judge)]. The value alone when it names itself ('powder', '12
    volts'); 'key value' when it does not ('number of usb ports 4') or when the key gives
    it its role ('allergen information dairy', 'input voltage 240 volts'); 'compatible
    with <devices>' for a device list (a cable is not the phone); the key of a yes flag
    ('Is Waterproof: Yes' -> 'waterproof'). Dropped: no / n/a flags, a unit count of 1,
    'other display features' (mostly 'wireless' on wired goods), a colour that is no
    colour ('14', 'default', 'see picture'), a compat value that is a seller warning,
    'attribute not applicable'."""
    out = []
    for item in _blank(specs).split("; "):
        item = item.strip()
        if not item:
            continue
        key, sep, value = item.partition(":")
        if not sep:
            out.append((item, item))
            continue
        key = _KEY_QUALIFIER_RE.sub("", key).strip()
        k, value = key.lower(), value.strip()
        if not value or k in _SPEC_DROP_KEYS or _JUNK_VALUE_RE.search(value):
            continue
        if k == "included components" and all(
                _junk_component(p) for p in _clean(value).split(",")):
            continue
        if _BOOL_RE.match(value):
            if _YES_RE.match(value) and k not in _YES_JUNK_KEYS:
                flag = _FLAG_PREFIX_RE.sub("", key).rstrip("?").strip()
                if flag:
                    out.append((flag, flag))
            continue
        if k in _COUNT_KEYS and _ONE_COUNT_RE.match(value):
            continue
        if k in _COLOR_KEYS and is_placeholder_colour(value):
            continue
        if k in _COMPAT_KEYS:
            if not _COMPAT_WARNING_RE.search(value):
                out.append((f"compatible with {value}", value))
        elif k in _ROLE_KEYS:
            out.append((f"{key} {value}", value))
        elif re.search(r"[A-Za-z]{3}", value):
            out.append((value, value))
        else:
            out.append((f"{key} {value}", value))
    return out


def _is_sentence(segment: str) -> bool:
    """Ends like a sentence and says two things. A list number ('detailed information:
    1.') is not a sentence end."""
    return (bool(re.search(r"[.!?]$", segment))
            and not re.search(r"(?<![\w.,])\d{1,2}[.!?]$", segment)
            and len(content_words(segment)) >= 2)


def _clip(text: str, count_tokens: Callable[[str], int], room: int) -> str:
    """The longest prefix of `text`, cut at a word boundary, that fits in `room` tokens."""
    words = text.split()
    lo, hi = 0, len(words)
    while lo < hi:                       # largest k with count(words[:k]) <= room
        mid = (lo + hi + 1) // 2
        if count_tokens(" ".join(words[:mid])) <= room:
            lo = mid
        else:
            hi = mid - 1
    return " ".join(words[:lo])


def _trim_clip(clipped: str, text: str) -> str:
    """A clip of `text` without a dangling end: trailing stop words and negators ('for
    a', 'no'), a number whose unit was cut off ('9.3' | 'oz'), a word whose 'free' was
    cut off ('gluten' | 'free'). Only words go, so it stays a prefix of `text`."""
    words, source = clipped.split(), text.split()
    k = len(words)
    while k:
        last = _norm_phrase(words[k - 1])
        nxt = source[k].strip(",;:.!?()") if k < len(source) else ""
        # ('10' | 'in 1' goes too: cut from its 'in 1' the number says nothing; review round 4)
        unit = bool(_TRAIL_UNIT_RE.fullmatch(nxt))
        number = re.fullmatch(r"\d+(?:[.,]\d+)*", words[k - 1].strip(",;:"))
        # '10 x' | '5': a dimension cut in two (never a lone 'x' in general: 'iphone x')
        times = (words[k - 1].lower() in ("x", "×") and k > 1
                 and re.fullmatch(r"\d+(?:[.,]\d+)*", words[k - 2].strip(",;:"))
                 and nxt[:1].isdigit())
        if (not last or last in _STOP or last in _NEGATORS or times
                or re.fullmatch(r"-?free", nxt.lower()) or (number and unit)):
            k -= 1
            continue
        break
    return " ".join(words[:k])


def _number_said(value: str, seen: Set[str]) -> bool:
    """The N of an 'N count' value is already in the text, bare or with a unit."""
    m = _COUNT_VALUE_RE.match(value)
    if not m:
        return False
    n = ("%f" % float(m.group(1))).rstrip("0").rstrip(".") or "0"   # '18.00' -> '18'
    if "." in n:
        p = n.replace(".", "p")
        return any(k.startswith(p) and k[len(p):].isalpha() for k in seen)
    return n in seen or any(k.startswith(n) and k[len(n):].isalpha() for k in seen)


def compact_text(
    row: Mapping,
    count_tokens: Callable[[str], int],
    budget: int,
    clip_last: bool = True,
    min_clip_tokens: int = 6,
) -> str:
    """The compacted, lowercased embedding text of one catalog row, within `budget`
    tokens (`count_tokens` counts content tokens, no [CLS]/[SEP]). Pieces are joined
    with single spaces; a wordpiece tokenizer splits on whitespace first, so a piece's
    count is the same alone as inside the joined text (checked on gte-small)."""
    pieces: List[str] = []
    used = 0
    seen: Set[str] = set()
    stream: List[str] = []               # normalised words of the text so far
    grams: Set[Tuple[str, ...]] = set()  # its runs of _RUN words

    def remember(piece: str) -> None:
        start = max(len(stream) - _RUN + 1, 0)
        stream.extend(w for w in (_norm_phrase(t) for t in piece.split()) if w)
        for i in range(start, len(stream) - _RUN + 1):
            grams.add(tuple(stream[i:i + _RUN]))

    def add(piece: str, force: bool = False) -> bool:
        nonlocal used
        piece = _clean(piece)
        if not piece:
            return False
        n = count_tokens(piece)
        if not force and used + n > budget:
            return False
        pieces.append(piece)
        used += n
        seen.update(said_keys(piece))
        remember(piece)
        return True

    def without_repeats(segment: str) -> str:
        """The segment minus every run of _RUN words already said (in the text so far
        or earlier in the segment): keyword stuffing and pasted-twice sentences."""
        tokens = segment.split()
        norm = [_norm_phrase(t) for t in tokens]
        keep = [True] * len(tokens)
        local = set(grams)
        for i in range(len(tokens) - _RUN + 1):
            run = tuple(norm[i:i + _RUN])
            if not all(run):
                continue
            if run in local:
                keep[i:i + _RUN] = [False] * _RUN
            else:
                local.add(run)
        return " ".join(t for t, k in zip(tokens, keep) if k)

    heads: List[str] = []
    for field in HEAD_FIELDS:                        # "grocery grocery" says it once
        value = _clean(row.get(field))
        if value and value not in heads:
            heads.append(value)
            add(value, force=True)
    title = dedup_title(_clean(row.get("Product_title")))
    if count_tokens(title) > budget - used:          # a title alone over budget (rare)
        title = _clip(title, count_tokens, max(budget - used, 0))
    add(title, force=True)

    for field in TAG_FIELDS:
        value = _clean(row.get(field))
        if field == "color" and is_placeholder_colour(value):
            continue                                 # '14' or 'default' says nothing
        if not _adds_words(value, seen):
            continue
        subs = [s.strip() for s in value.split(",") if s.strip()]
        if len(subs) > 1:                            # 'vegan, keto' with keto in the title
            said = " ".join(pieces)
            subs = [s for s in subs if not _phrase_in(s, said)]
        add(", ".join(subs))

    for text, value in _spec_items(row.get("specs")):
        if _number_said(value, seen):                # 'unit count 18.00 count' vs '18 pack'
            continue
        if _adds_words(value, seen):
            add(text)

    failed = False
    mostly_said: List[str] = []
    description = _blank(row.get("prod_description")) or _blank(row.get("gen_description"))
    for segment in segments(description):
        segment = strip_boilerplate(segment)
        # 'No preservatives.' is one new word and a fact; a long segment that mostly
        # repeats the title is not
        if not segment:
            continue
        if not _adds_words(segment, seen, min_new=1, max_seen_share=0.8):
            mostly_said.append(segment)
            continue
        segment = without_repeats(segment)
        if not segment or not _adds_words(segment, seen, min_new=1, max_seen_share=0.8):
            continue
        if failed and not _is_sentence(segment):
            continue                                 # no header-only or cut-off scraps
        if add(segment):
            continue
        if not failed:
            failed = True
            room = budget - used
            if clip_last and room >= min_clip_tokens:
                clean = _clean(segment)
                add(_trim_clip(_clip(clean, count_tokens, room), clean))
    # Room left: a templated sentence that only looked said ('the same hot sauce ... add
    # zesty jalapeno flavor') may still name the one thing that sets the product apart
    for segment in mostly_said:
        if used >= budget:
            break
        rest = without_repeats(segment)
        # never a scrap: a lone number or letter ('1', 'c'), a bare header ('features:')
        if (rest and content_words(rest) and not rest.rstrip().endswith(":")
                and _adds_words(rest, seen, min_new=1, max_seen_share=0.8)):
            add(rest)
    return " ".join(pieces)


# ------------------------------------------------------------------ the sidecar
SIDECAR_COLUMNS = ("catalog_index", "parent_asin", "embed_text")


def sidecar_paths(csv_path: Path) -> Tuple[Path, Path]:
    """(rows, metadata) files next to the catalog, named by it and by COMPACT_VERSION: a
    catalog never reads another catalog's or another rule version's sidecar."""
    stem = Path(csv_path).with_suffix("")
    base = f"{stem.name}.compact_v{COMPACT_VERSION}"
    return stem.with_name(base + ".csv"), stem.with_name(base + ".meta.json")


def compact_cache_path(default_cache: Path) -> Path:
    """The compact vectors, next to the default cache under a SHORT name (a digest of its
    name): the default cache's own name is ~180 characters, and with Windows' 260-char
    path limit anything longer built on it failed to write."""
    p = Path(default_cache)
    digest = hashlib.md5(p.name.encode("utf-8")).hexdigest()[:12]
    return p.with_name(f"embeddings_{digest}_compact{COMPACT_VERSION}.npy")


def stamp_path(cache: Path) -> Path:
    """What the compact vectors were encoded from (the sidecar's md5, the embedder)."""
    return Path(cache).with_suffix(".json")


def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sidecar_matches(meta: Mapping, csv_hash: str, n_rows: int, model_name: str,
                    max_seq_length: Optional[int], fields: Sequence[str]) -> Optional[str]:
    """Why a sidecar's metadata does not fit the loaded catalog, embedder and field list,
    or None when it does. A mismatch must never be served."""
    checks = (
        ("version", COMPACT_VERSION), ("csv_hash", csv_hash), ("n_rows", int(n_rows)),
        ("model", model_name), ("max_seq_length", max_seq_length),
        ("fields", list(fields)),
    )
    for key, want in checks:
        if meta.get(key) != want:
            return f"{key} differs"
    return None


def check_rows(meta: Mapping, catalog_index, sidecar_asins, catalog_asins,
               n_catalog: int) -> Optional[str]:
    """Why the sidecar's rows do not fit the catalog, or None: the count the build wrote,
    every position in range, and the product at each position the one the build saw
    (parent_asin, missing read as '' on both sides). The load path and encode run the
    same check, so neither can serve or encode another product's text."""
    n = len(catalog_index)
    if n != meta.get("n_changed"):
        return "row count differs"
    if n and (min(catalog_index) < 0 or max(catalog_index) >= n_catalog):
        return "rows out of range"
    if any(catalog_asins[i] != a for i, a in zip(catalog_index, sidecar_asins)):
        return "parent_asin differs"
    return None


def changed_rows(default_counts, budget: int) -> List[int]:
    """Positions whose default text the embedder would cut."""
    return [i for i, n in enumerate(default_counts) if n > budget]
