"""Fine categories, roles and facet values for Electronics and Grocery rows, and what a
watch-related title sells, derived from the TITLE (plus `dietary`, `store` and `specs`)
at load time -- the Electronics / Grocery counterpart of app/product_type.py.

The catalog has no product-type field outside Fashion: an Electronics row is "a phone
case" or "a phone" only in its title, and about 60% of Electronics rows are accessories
that NAME a device ("Case for iPhone 12", "Charger for Apple Watch"). So a device ask
("gift for a vintage watch collector") had nothing to rank on, and the rail had nothing
to filter on. `annotate_catalog()` adds, on both backends (app/search.py), never
embedded and never written to the CSV (search_text and the embedding cache key stay
byte-identical, no re-embed):

- `item_category`: an ELECTRONICS_CATEGORIES / GROCERY_CATEGORIES key or "unknown";
  None on Fashion and on rows of no known vertical;
- `item_role`: Electronics device / accessory / part / consumable / service, Grocery
  food / non_food; None for "unknown" and elsewhere;
- `watch_kind`, on EVERY vertical: what a watch-related title sells (WATCH_KINDS), None
  when the title is not about a watch. An Electronics "watches" row is exactly a
  watch_kind "watch" row (one source);
- `diet_tags` (Grocery), `works_with` (Electronics accessories, parts, consumables):
  tuples of fixed vocabulary keys, () elsewhere;
- `item_condition` (Electronics): "renewed" / "used" as the title words it, else None;
- `brand`: `store`, except an Electronics row whose store is a CHIP vendor ("intel" on
  a Lenovo laptop: 3,463 rows) and whose specs name the brand.

How a title is read (the 2026-10-27 taxonomy study, scratch cas_taxonomy.py; numbers
in the module notes): the product is the HEAD noun, not the first keyword. The title is
cut into segments at separators and connectors; after a TARGET connector ("for",
"compatible with", "fits", "replacement for") a device noun is what the product is FOR
and is skipped there ("Silicone Case for iPhone 12" is a case). Inside a segment the
RIGHT-MOST category phrase wins (English compounds are right-headed: "iphone case",
"camera lens cap"), at the same end the longer phrase, then rule order. WEAK rules are
bundle words that lose to an earlier device noun ("Wireless Earbuds ... Charging Case"
stays earbuds). Held-out precision (one reader, frozen rules): Electronics category
85.0%, device-vs-accessory role 93.1%; Grocery category 84.4%, food/non-food 95.6%;
watch-vs-accessory 99%.

Pure code (re/pandas only): no LLM, no I/O, deterministic, English-only and
hand-maintained like product_type and the query guard. Speed: each rule is indexed by
the tokens its matches can END and START with, derived from the PATTERN itself
(_pattern_index), never from a catalog: a segment runs a rule only where one of its
head tokens occurs. tests/test_catalog_taxonomy.py checks, under CAS_SLOW_TESTS=1 on
the whole catalog, that no match ends or starts on a token the index lacks, that the
fast reading equals the unindexed one (classify_text_all: every rule from every token
start) where it departs from finditer's, and counts that residue (classify_text_ref:
finditer skips overlapping matches, 159 + 153 titles on 2026-10-27's catalog).
"""

import time

# When THESE rules were loaded, before anything else runs: app/search.py keys its local
# cache of the derived columns on this file's bytes only while the file is no newer than
# this (an edit after the import would file the old rules' columns under the new rules'
# name). Never read by a classifier.
IMPORTED_AT = time.time()

import math  # noqa: E402
import re  # noqa: E402
from typing import Dict, FrozenSet, List, Optional, Tuple  # noqa: E402

import pandas as pd  # noqa: E402

try:  # Python 3.11+ (the App and the laptop run 3.11 / 3.13)
    from re import _constants as _sre_c, _parser as _sre_parse
except ImportError:  # pragma: no cover -- older Pythons
    import sre_constants as _sre_c  # type: ignore[no-redef]
    import sre_parse as _sre_parse  # type: ignore[no-redef]

CATEGORY_COLUMN = "item_category"
ROLE_COLUMN = "item_role"
WATCH_COLUMN = "watch_kind"
DIET_COLUMN = "diet_tags"
WORKS_WITH_COLUMN = "works_with"
CONDITION_COLUMN = "item_condition"
BRAND_COLUMN = "brand"
# Every column annotate_catalog() adds, in the order it adds them; none is embedded
ADDED_COLUMNS = (CATEGORY_COLUMN, ROLE_COLUMN, WATCH_COLUMN, DIET_COLUMN, WORKS_WITH_COLUMN,
                 CONDITION_COLUMN, BRAND_COLUMN)
UNKNOWN = "unknown"
ELECTRONICS, GROCERY, FASHION = "Electronics", "Grocery", "Fashion"


def vertical_of(value) -> Optional[str]:
    """"Electronics" / "Grocery" / "Fashion" for a `bsns_vrtcl_name` value, else None.
    Case-insensitive like product_type.is_fashion (the old local frame was lowercase);
    NaN, None and pd.NA are no vertical."""
    if not isinstance(value, str):
        return None
    return _VERTICALS.get(value.strip().casefold())


_VERTICALS = {"electronics": ELECTRONICS, "grocery": GROCERY, "fashion": FASHION}


def _norm(title) -> str:
    """Casefolded title, curly apostrophe and en/em dashes straightened; non-str -> ''."""
    if not isinstance(title, str):
        return ""
    return (title.casefold().replace("\u2019", "'").replace("\u2013", "-")
            .replace("\u2014", "-"))


# ------------------------------------------------------------------ the rule index
# A title is read token by token: [a-z0-9]+ runs, everything else separates. A rule runs
# only at a token its matches can END on (its "heads": "case" for "iphone case"), and
# only from a token its matches can START on ("firsts"), at most `maxlen` tokens back.
# Those sets come from the regex itself (sre_parse), so a new catalog needs nothing
# re-derived (the 2026-10-27 prototype read them off this catalog's matches instead).
# Tokens with a digit are keyed by their letters before the first digit plus "0"
# ("s21ultra" -> "s0", "38mm" -> "0", "ddr4" -> "ddr0"): what a `\d` class can promise.
# A `\w` run makes a PREFIX key ("galaxy tab a\w*" ends on any token starting with "a");
# one with no letters before it ("gopro hero\s?\d*\w*" can end on "black") makes the rule
# ANCHORED: it runs from each token it can start on instead (the same matches, see
# Taxonomy._found_fast). `\w` is read as a token character; one matching '_' or a
# non-ASCII letter (both separators here) could in principle end a match elsewhere --
# the CAS_SLOW_TESTS check counts any token the index misses on the whole catalog.
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_LETTERS_RE = re.compile(r"[a-z]*")
_TOKEN_CHARS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789")
_DIGITS = frozenset("0123456789")
_WILD = "\x01"          # "any token characters from here" in a derived token
_CAP = 12               # tokens: no rule phrase is longer; an unbounded repeat reads as 12
_LIMIT = 200_000        # abstract strings per sub-pattern before a rule gives up its index
_MAX_REPEATS = 10       # copies of a repeated group read before its token shapes must settle


class _Unsupported(Exception):
    """A pattern the index derivation cannot summarise: the rule then runs everywhere."""


def _token_key(tok: str) -> str:
    """Index key of a title token (see above)."""
    if tok.isalpha():
        return tok
    return tok[:_LETTERS_RE.match(tok).end()] + "0"


def _canon(text: str) -> str:
    """Canonical derived token text: letters as is; at the first digit, the letters
    before it + '0'; at a wildcard before any digit, the letters before it + _WILD. A
    canonical text ending in '0' or _WILD has its key decided: what follows is ignored."""
    for i, ch in enumerate(text):
        if ch == _WILD:
            return text[:i + 1]
        if ch in _DIGITS:
            return text[:i] + "0"
    return text


# An abstract string: (n tokens, first token, last token, starts on a token char, ends
# on one). Two special values: the empty string (transparent) and separators only.
_EMPTY = (0, "", "", True, True)
_SEP = (0, "", "", False, False)


def _join(a: str, b: str) -> str:
    if a[-1:] in ("0", _WILD):
        return a
    return _canon(a + b)


def _cat(a, b):
    """The abstract string of the concatenation of two abstract strings."""
    if a is _EMPTY or a == _EMPTY:
        return b
    if b is _EMPTY or b == _EMPTY:
        return a
    na, fa, la, lead_a, trail_a = a
    nb, fb, lb, lead_b, trail_b = b
    if na == 0:
        return _SEP if nb == 0 else (nb, fb, lb, False, trail_b)
    if nb == 0:
        return (na, fa, la, lead_a, False)
    if trail_a and lead_b:  # the last token of a runs on into the first of b
        merged = _join(la, fb)
        return (min(na + nb - 1, _CAP), merged if na == 1 else fa, merged if nb == 1 else lb,
                lead_a, trail_b)
    return (min(na + nb, _CAP), fa, lb, lead_a, trail_b)


def _cat_sets(left, right):
    out = {_cat(a, b) for a in left for b in right}
    if len(out) > _LIMIT:
        raise _Unsupported("too many strings")
    return out


def _char(ch: str):
    if ch == _WILD:
        return (1, _WILD, _WILD, True, True)
    if ch in _TOKEN_CHARS:
        t = "0" if ch.isdigit() else ch
        return (1, t, t, True, True)
    return _SEP


def _charset(items):
    chars, wide = set(), False
    for op, av in items:
        if op is _sre_c.NEGATE:
            wide = True
        elif op is _sre_c.LITERAL:
            chars.add(chr(av))
        elif op is _sre_c.RANGE:
            lo, hi = av
            if hi - lo > 16:
                wide = True
            else:
                chars.update(chr(c) for c in range(lo, hi + 1))
        elif op is _sre_c.CATEGORY:
            if av is _sre_c.CATEGORY_DIGIT:
                chars.add("0")
            elif av is _sre_c.CATEGORY_SPACE:
                chars.add(" ")
            elif av is _sre_c.CATEGORY_WORD:
                chars.add(_WILD)
            else:
                wide = True
        else:
            raise _Unsupported(str(op))
    if wide:
        return {_SEP, _char(_WILD)}
    return {_char(c) for c in chars}


def _shape(s):
    """What a further concatenation can depend on: everything but the exact count."""
    return (min(s[0], 2),) + s[1:]


def _repeat(lang, lo: int, hi: int):
    """lo..hi copies of a language. Copies are added until one more copy brings no new
    shape (then no later copy can: _cat reads only the shape); if more copies are allowed
    and they add tokens, the count is read as the cap."""
    current, out, copies = {_EMPTY}, set(), 0
    while True:
        if copies >= lo:
            if copies > lo and {_shape(s) for s in current} <= {_shape(s) for s in out}:
                if copies < hi and max(s[0] for s in current) > max(s[0] for s in out):
                    current = {(_CAP,) + s[1:] if s[0] else s for s in current}
                out |= current
                return out
            out |= current
        if copies == hi:
            return out
        if copies >= lo + _MAX_REPEATS:
            raise _Unsupported("a repeat keeps adding token shapes")
        current = _cat_sets(current, lang)
        copies += 1


def _lang(items):
    out = {_EMPTY}
    for op, av in items:
        out = _cat_sets(out, _node(op, av))
    return out


def _node(op, av):
    if op is _sre_c.LITERAL:
        return {_char(chr(av))}
    if op is _sre_c.NOT_LITERAL or op is _sre_c.ANY:
        return {_SEP, _char(_WILD)}
    if op is _sre_c.IN:
        return _charset(av)
    if op is _sre_c.BRANCH:
        return set().union(*(_lang(alt) for alt in av[1]))
    if op is _sre_c.SUBPATTERN:
        return _lang(av[-1])
    if op in (_sre_c.MAX_REPEAT, _sre_c.MIN_REPEAT, getattr(_sre_c, "POSSESSIVE_REPEAT", None)):
        lo, hi, sub = av
        return _repeat(_lang(sub), lo, hi)
    if op in (_sre_c.AT, _sre_c.ASSERT, _sre_c.ASSERT_NOT):
        return {_EMPTY}   # zero-width: they only narrow what matches (a superset is safe)
    if op is getattr(_sre_c, "ATOMIC_GROUP", None):
        return _lang(av)
    raise _Unsupported(str(op))


class _Keys:
    """The tokens a rule's matches can end (or start) on: exact keys (_token_key form)
    plus prefixes ("a" = any token starting with "a"); `anything` = any token at all."""
    __slots__ = ("exact", "prefixes", "anything")

    def __init__(self, texts):
        self.exact = frozenset(t for t in texts if not t.endswith(_WILD))
        self.prefixes = tuple(sorted({t[:-1] for t in texts if t.endswith(_WILD)} - {""}))
        self.anything = any(t == _WILD for t in texts)

    def accepts(self, tok: str) -> bool:
        """Whether a title token can be one of these (the reading's filter)."""
        if self.anything:
            return True
        if (tok if tok.isalpha() else _token_key(tok)) in self.exact:
            return True
        return bool(self.prefixes) and tok.startswith(self.prefixes)


def _pattern_index(pattern: str):
    """(heads, firsts, maxlen) of a pattern: the _Keys its matches can end and start on,
    and the most tokens a match can span. None when the pattern is beyond the derivation
    (the rule then runs from every token)."""
    try:
        strings = _lang(_sre_parse.parse(pattern))
    except (_Unsupported, RecursionError):
        return None
    words = [s for s in strings if s[0] >= 1]
    if not words:
        return None
    return (_Keys({s[2] for s in words}), _Keys({s[1] for s in words}),
            max(s[0] for s in words))


def _longest_first(pattern: str) -> str:
    """Reorder the top-level alternatives of a r"\\b(?:a|b|...)\\b" pattern so phrases
    with more words come first (stable). Python's alternation returns the FIRST
    alternative that matches at a position, so "ac adapter|ac charger adapter" would stop
    at the shorter phrase and let another rule's head noun win the right-most test."""
    if not (pattern.startswith("\\b(?:") and pattern.endswith(")\\b")):
        return pattern
    body = pattern[5:-3]
    alts, depth, in_class, escaped, cur = [], 0, False, False, []
    for ch in body:
        if escaped:
            cur.append(ch)
            escaped = False
            continue
        if ch == "\\":
            cur.append(ch)
            escaped = True
            continue
        if in_class:
            cur.append(ch)
            if ch == "]":
                in_class = False
            continue
        if ch == "[":
            in_class = True
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "|" and depth == 0:
            alts.append("".join(cur))
            cur = []
            continue
        cur.append(ch)
    alts.append("".join(cur))
    if depth != 0:
        return pattern
    alts = sorted(alts, key=lambda a: -a.count(" "))
    return "\\b(?:" + "|".join(alts) + ")\\b"


_ANYTHING = _Keys({_WILD})


class _Rule:
    __slots__ = ("key", "pattern", "weak", "order", "device", "heads", "firsts", "maxlen")

    def __init__(self, key: str, pattern: str, weak=False):
        self.key = key
        self.pattern = re.compile(_longest_first(pattern))
        self.weak = weak        # False, True (loses to an earlier device) or 2 (to any noun)
        self.order = 0
        self.device = False
        index = _pattern_index(self.pattern.pattern)
        if index is None:
            self.heads, self.firsts, self.maxlen = _ANYTHING, _ANYTHING, _CAP
        else:
            self.heads, self.firsts, self.maxlen = index

    @property
    def anchored(self) -> bool:
        """Read from its first tokens, not its heads (a head could be any token)."""
        return self.heads.anything


def R(key: str, pattern: str, weak=False) -> _Rule:
    return _Rule(key, pattern, weak)


def _token_starts(seg: str, toks: List[str]) -> List[int]:
    """The offset of each token of `seg` (toks = _TOKEN_RE.findall(seg))."""
    starts, cur = [], 0
    for tk in toks:
        p = seg.find(tk, cur)
        starts.append(p)
        cur = p + len(tk)
    return starts


class Taxonomy:
    """Ordered rules over one vertical plus its category table {key: (label, role)}."""

    def __init__(self, name, categories, rules, cut, weak_vs_any=False,
                 model_tokens=frozenset()):
        self.name = name
        self.categories = categories
        self.rules = rules
        self.cut = cut                    # connector word -> {next word or "": is_target}
        self.weak_vs_any = weak_vs_any    # Grocery: a weak noun loses to ANY earlier strong one
        self.model_tokens = model_tokens  # words of a bare model name (iphone, galaxy, pro...)
        self.index: Dict[str, List[_Rule]] = {}          # head key -> rules
        self.prefix_index: Dict[str, List[_Rule]] = {}   # head prefix -> rules
        self.anchored: List[_Rule] = []                  # rules whose heads could be any token
        for i, r in enumerate(rules):
            r.order = i
            if r.key not in categories:
                raise KeyError(r.key)
            r.device = categories[r.key][1] == "device"
            if r.anchored:
                self.anchored.append(r)
                continue
            for head in r.heads.exact:
                self.index.setdefault(head, []).append(r)
            for prefix in r.heads.prefixes:
                self.prefix_index.setdefault(prefix, []).append(r)
        self.prefix_lengths = tuple(sorted({len(p) for p in self.prefix_index}))

    def segments(self, t: str) -> List[Tuple[str, Optional[bool]]]:
        """[(segment text, connector kind)] in title order (see iter_segments)."""
        return list(self.iter_segments(t))

    def iter_segments(self, t: str):
        """Cut a normalised title at separators and connectors with a word scan (no
        regex). The kind is True after a TARGET connector ("for"), None after a BUNDLE
        connector ("with"), else False. `self.cut` maps a connector word to {next word or
        "": kind}; "" means the word alone is a connector."""
        for c in _PAD_CHARS:            # str.translate(dict) cost ~10 us/title; this ~1 us
            if c in t:
                t = t.replace(c, _PAD_TO[c])
        words = t.split()
        conn = self.cut
        seg, target, i, n = [], False, 0, len(words)
        while i < n:
            w = words[i]
            i += 1
            while w and w[0] in _OPEN:
                yield " ".join(seg), target
                seg, target = [], False
                w = w[1:]
            if not w:
                continue
            if w in _SEPS:
                yield " ".join(seg), target
                seg, target = [], False
                continue
            c = conn.get(w)
            if c is not None:
                nxt = words[i] if i < n else None
                if nxt is not None and nxt in c:
                    yield " ".join(seg), target
                    seg, target = [], c[nxt]
                    i += 1
                    continue
                if "" in c:
                    yield " ".join(seg), target
                    seg, target = [], c[""]
                    continue
            end = len(w)
            while end and w[end - 1] in _CLOSE:
                end -= 1
            if end < len(w):
                if end:
                    seg.append(w[:end])
                yield " ".join(seg), target
                seg, target = [], False
                continue
            seg.append(w)
        yield " ".join(seg), target

    def _decide(self, found):
        """found: [(end, -start, -order, rule, start, text)] -> the winning item or None.
        Right-most end wins, then the longer phrase, then rule order; a WEAK rule (bundle
        word) loses to a device noun named earlier in the segment (weak=2: to any strong
        noun; Grocery: to any strong noun as well)."""
        first_device = first_strong = None
        any_ok = self.weak_vs_any
        for _end, _nstart, _norder, r, start, _text in found:
            if r.weak:
                continue
            if first_strong is None or start < first_strong:
                first_strong = start
            if (r.device or any_ok) and (first_device is None or start < first_device):
                first_device = start
        best = None
        for item in found:
            r, start = item[3], item[4]
            if r.weak == 2 and first_strong is not None and first_strong < start:
                continue
            if r.weak and first_device is not None and first_device < start:
                continue
            if best is None or item[:3] > best[:3]:
                best = item
        return best

    def _model_only(self, rule, text: str) -> bool:
        """A device phrase made only of model words and numbers ("iPhone 7 Plus")."""
        return (rule.device and bool(self.model_tokens)
                and all(tk in self.model_tokens or tk[0].isdigit()
                        for tk in _TOKEN_RE.findall(text)))

    def _found_fast(self, seg: str, target):
        """Every (end, -start, -order, rule, start, text) the indexed reading finds: for
        each token a rule can end on, the match from the EARLIEST token (at most maxlen
        back) that reaches it. An anchored rule instead tries every token it can start
        on; _decide picks the same winner from either list (the right-most end, then the
        earliest start), so both readings agree wherever the index is complete."""
        toks = _TOKEN_RE.findall(seg)
        if not toks:
            return None
        keys = [tok if tok.isalpha() else _token_key(tok) for tok in toks]
        idx, prefix_index, prefix_lengths = self.index, self.prefix_index, self.prefix_lengths
        found = starts = None
        for j, key in enumerate(keys):
            rules = idx.get(key)
            for size in prefix_lengths:
                more = prefix_index.get(toks[j][:size])
                if more:
                    rules = (rules or []) + more
            if not rules:
                continue
            if starts is None:
                starts = _token_starts(seg, toks)
            he = starts[j] + len(toks[j])
            for r in rules:
                if target and r.device:
                    continue
                firsts = r.firsts
                exact, prefixes, anything = firsts.exact, firsts.prefixes, firsts.anything
                lo = j - r.maxlen + 1
                for k in range(lo if lo > 0 else 0, j + 1):
                    if not (anything or keys[k] in exact
                            or (prefixes and toks[k].startswith(prefixes))):
                        continue
                    m = r.pattern.match(seg, starts[k])
                    if m is not None and m.end() >= he:
                        if found is None:
                            found = []
                        found.append((m.end(), -starts[k], -r.order, r, starts[k], m.group()))
                        break
        for r in self.anchored:
            if target and r.device:
                continue
            firsts = r.firsts
            for k, key in enumerate(keys):
                if not (firsts.anything or key in firsts.exact
                        or (firsts.prefixes and toks[k].startswith(firsts.prefixes))):
                    continue
                if starts is None:
                    starts = _token_starts(seg, toks)
                m = r.pattern.match(seg, starts[k])
                if m is not None:
                    if found is None:
                        found = []
                    found.append((m.end(), -starts[k], -r.order, r, starts[k], m.group()))
        return found

    def _found_ref(self, seg: str, target):
        """The exhaustive reading: every rule's finditer on the segment."""
        found = []
        for r in self.rules:
            if target and r.device:
                continue
            for m in r.pattern.finditer(seg):
                found.append((m.end(), -m.start(), -r.order, r, m.start(), m.group()))
        return found or None

    def _found_all(self, seg: str, target):
        """The unindexed reading: every rule matched from EVERY token start. finditer
        reports only non-overlapping matches, so it can lose a phrase that overlaps an
        earlier one ("lens mount adapter"); this is what the fast reading must equal."""
        toks = _TOKEN_RE.findall(seg)
        if not toks:
            return None
        starts = _token_starts(seg, toks)
        found = []
        for r in self.rules:
            if target and r.device:
                continue
            for s in starts:
                m = r.pattern.match(seg, s)
                if m is not None:
                    found.append((m.end(), -s, -r.order, r, s, m.group()))
        return found or None

    def _classify(self, t: str, finder) -> Tuple[Optional[str], str]:
        if not t:
            return None, ""
        si = -1
        fallback = model_only = None
        bundled = False
        for seg, target in self.iter_segments(t):
            si += 1
            if target is None:
                bundled = True
            if not seg:
                continue
            found = finder(seg, target)
            if found is None:
                continue
            item = found[0] if len(found) == 1 else self._decide(found)
            if item is None:
                continue
            rule = item[3]
            where = "h" if si == 0 else str(si)
            if model_only is not None:
                # the head named only a model ("iPhone 7 Plus, iPhone 8 Plus, ... Case"): a
                # later accessory or part noun is the product -- never past a bundle
                # connector ("Galaxy Tab A 10.1 with 64GB Memory Card" stays a tablet)
                if not rule.device and not rule.weak and not target and not bundled:
                    return rule.key, "m" + str(si)
                continue
            if not target and self._model_only(rule, item[5]):
                model_only = (rule.key, where)
                continue
            if not rule.weak or not self.weak_vs_any:
                return rule.key, where
            if fallback is None:
                fallback = (rule.key, "w" + ("" if si == 0 else str(si)))
        if model_only is not None:
            return model_only
        return fallback if fallback is not None else (None, "")

    def classify_text(self, t: str) -> Tuple[Optional[str], str]:
        """(category key or None, how) of a NORMALISED title (_norm); `how` names the
        segment that decided ('h' = the head)."""
        return self._classify(t, self._found_fast)

    def classify_text_ref(self, t: str) -> Optional[str]:
        """The slow reference reading (every rule's finditer on every segment), for the
        soundness check; the fast reading differs from it on a counted residue (159
        Electronics and 153 Grocery titles of 2026-10-27's catalog), all of it finditer's
        overlap rule: classify_text_all agrees with the fast reading on every title."""
        return self._classify(t, self._found_ref)[0]

    def classify_text_all(self, t: str) -> Optional[str]:
        """The unindexed reading (every rule from every token start): the fast reading's
        definition. Equal to classify_text on all 198,904 distinct Electronics / Grocery
        titles of 2026-10-27's catalog (CAS_SLOW_TESTS checks residue titles + a sample)."""
        return self._classify(t, self._found_all)[0]

    def classify(self, title) -> Optional[str]:
        return self.classify_text(_norm(title))[0]

    def role(self, key) -> Optional[str]:
        return self.categories[key][1] if key in self.categories else None


# Connectors. A TARGET connector introduces the device the product is for.
_OPEN = frozenset("([{|\u3010")
_CLOSE = frozenset(",;:|)]}\u3011")
_SEPS = frozenset({"-", "--", "---", "+", "|", "&&", ",", ";", "(", ")", "[", "]", "{", "}",
                   "\u3010", "\u3011"})
# punctuation that cuts even inside a word ("kit,portable"): padded with spaces first
_PAD_CHARS = ",;|()[]{}\u3010\u3011"
_PAD_TO = {c: " " + c + " " for c in _PAD_CHARS}
_T, _N, _B = True, False, None   # target / plain / bundle (falsy)


# ====================================================================== ELECTRONICS
# key: (shopper-facing label, role); role in device / accessory / part / consumable /
# service. Real devices are rare: 226 phone rows against 15,492 cases.
_ELECTRONICS_TABLE = {
    # devices
    "headphones": ("Headphones & earbuds", "device"),
    "audio": ("Speakers & home audio", "device"),
    "phones": ("Cell phones & home phones", "device"),
    "computers": ("Laptops & desktops", "device"),
    "tablets": ("Tablets & e-readers", "device"),
    "monitors": ("Monitors", "device"),
    "tv_video": ("TVs, projectors & streaming", "device"),
    "cameras": ("Cameras & camcorders", "device"),
    "watches": ("Watches, smartwatches & fitness trackers", "device"),
    "smart_home": ("Smart home & security", "device"),
    "networking": ("Networking & Wi-Fi", "device"),
    "gaming": ("Video game consoles & controllers", "device"),
    "car": ("Car electronics & GPS", "device"),
    "office": ("Printers, scanners & office electronics", "device"),
    "optics": ("Binoculars, telescopes & optics", "device"),
    "lighting": ("Lights & lamps", "device"),
    # accessories / parts / consumables / service
    "cases": ("Cases, covers & skins", "accessory"),
    "screen_protectors": ("Screen protectors", "accessory"),
    "power": ("Chargers, power banks & power supplies", "accessory"),
    "batteries": ("Batteries", "consumable"),
    "cables": ("Cables, adapters & hubs", "accessory"),
    "computer_accessories": ("Keyboards, mice & computer accessories", "accessory"),
    "components": ("Computer components & replacement parts", "part"),
    "storage": ("Storage & memory cards", "accessory"),
    "camera_accessories": ("Lenses, tripods & camera accessories", "accessory"),
    "backdrops": ("Photo backdrops & props", "accessory"),
    "bags": ("Bags, backpacks & sleeves", "accessory"),
    "mounts": ("Mounts, stands & holders", "accessory"),
    "av_accessories": ("Remotes & antennas", "accessory"),
    "watch_accessories": ("Watch bands & watch accessories", "accessory"),
    "supplies": ("Ink, toner, labels & media", "consumable"),
    "tools": ("Tools, testers & cleaning", "accessory"),
    "service": ("Protection plans", "service"),
}

_NOT_AFTER_NUM = r"(?<!\d)(?<!\d )"

ELEC_RULES = [
    R("headphones",
      r"\b(?:head ?phones?|ear ?buds?|ear ?phones?|head ?sets?|ear ?pieces?|in-ear monitors?|iems?|"
      r"airpods(?: pro| max)?|earpods|neck ?bands?(?= (?:headphones?|earbuds?|bluetooth))|"
      r"bone conduction(?: headphones?)?|hearing amplifiers?)\b"),
    R("audio",
      r"\b(?:speakers?|sound ?bars?|soundbars?|sub ?woofers?|subwoofers?|amplifiers?|"
      + _NOT_AFTER_NUM + r"amps?|"
      r"(?:av|a/v|stereo|audio|home theat(?:er|re)|surround(?: sound)?|network|"
      r"bluetooth) receivers?|"
      r"turntables?|record players?|radios?|boom ?box(?:es)?|(?:cd|mp3|mp4|music|"
      r"cassette) players?|"
      r"(?:personal|portable|pocket|mp3|mp4) media players?|"
      r"cassette (?:recorders?|decks?)|walkmans?|ipod(?: (?:nano|classic|shuffle|touch))?|"
      r"loud ?speakers?|(?:voice|digital|audio|sound|micro ?cassette|cassette|tape|"
      r"minidisc) recorders?|microphones?|mics?|lavalier|preamps?|pre-amps?|"
      r"audio interfaces?|(?:audio|dj|mixing|sound) (?:mixers?|consoles?|boards?)|dj controllers?|"
      r"karaoke(?: machines?| systems?)?|(?:stereo|hi-?fi|shelf|micro|mini|compact) (?:systems?|"
      r"stereos?)|"
      r"home theat(?:er|re)(?: systems?)?|equali[sz]ers?|jukebox(?:es)?|megaphones?|pa systems?|"
      r"walkie[- ]talkies?|two[- ]way radios?|studio monitors?|monitor speakers?|"
      r"(?:digital|electric|electronic|midi|musical|piano|portable) (?:pianos?|keyboards?)|"
      r"keyboard pianos?|synthesi[sz]ers?|"
      r"(?:guitar|bass) (?:amps?|amplifiers?|pedals?|effects?)|guitar tuners?)\b"),
    R("phones",
      r"\b(?:(?:cell|mobile|smart|cordless|corded|landline|desk|office|ip|voip|flip|satellite|"
      r"senior|"
      r"basic|home|conference|unlocked|android|caller id|business|wall|dect|rotary|retro|"
      r"big button|"
      r"trimline|5g|4g|lte|gsm|prepaid) ?phones?|cellphones?|smartphones?|telephones?|"
      r"satellite (?:communicators?|messengers?)|"
      r"iphone(?: (?:\d+|x[rs]?|se))?(?: (?:pro|plus|max|mini))*(?= *$|,| -| \(|\s+(?:\d+ ?gb|"
      r"unlocked|renewed|fully|factory|smartphone|cell))|"
      r"galaxy (?:s|note|a|z ?flip|z ?fold)\s?\d+\w*(?= *$|,| -| \(|\s+(?:\d+ ?gb|unlocked|renewed|"
      r"factory|smartphone|cell|5g|lte|android))|"
      r"pixel \d\w*(?= *$|,| -| \(|\s+(?:\d+ ?gb|unlocked|renewed|factory|smartphone|5g))|"
      r"phones?(?= *$))\b"),
    R("computers",
      r"\b(?:laptops?|notebooks?(?! (?:paper|journal))|chromebooks?|macbooks?(?: (?:air|pro))?|"
      r"ultrabooks?|"
      r"netbooks?|desktops?(?! (?:stand|organizer|mount))|(?:desktop|personal|gaming|mini|tower|"
      r"all-in-one|all in one|business|home|small form factor|sff|workstation) (?:computers?|pcs?)|"
      r"computers?|mini pcs?|imacs?|mac minis?|mac pros?|workstations?|thin clients?|servers?|"
      r"2-in-1s?|convertibles?|(?:windows|gaming|desktop|tower) pcs?|pc towers?)\b"),
    R("tablets",
      r"\b(?:tablets?(?! (?:pcs?|computers?))(?<!drawing tablet)(?<!graphics tablet)(?<!pen tablet)"
      r"(?<!writing tablet)|ipads?(?: (?:pro|air|mini))?(?: \d+(?:\.\d+)?(?: ?(?:inch|\"|"
      r"in))?)?(?: \(?\d+(?:th|nd|rd|st) gen(?:eration)?\)?)?(?= *$|,| -| \(|\s+(?:wi-?fi|\d+ ?gb|"
      r"renewed|cellular))|"
      r"e-?readers?|ebook readers?|kindle(?: paperwhite| oasis| scribe| kids)?(?= *$|,| -| \(|"
      r"\s+(?:e-?reader|\d+ ?gb|wi-?fi))|"
      r"galaxy tab(?: [as]\d*\w*)?(?= *$|,| -| \(|\s+(?:\d+(?:\.\d+)?\"?|wi-?fi|\d+ ?gb|tablet))|"
      r"fire (?:hd )?\d+(?: kids)?(?: tablet)|tablet (?:pcs?|computers?)|android tablets?)\b"),
    R("monitors",
      r"\b(?:(?:lcd|led|ips|gaming|computer|pc|portable|curved|ultrawide|ultra-wide|4k|hd|fhd|qhd|"
      r"widescreen|touch ?screen|desktop|usb-?c?|oled|\d+(?:\.\d+)?-?(?:inch|\"|in)) monitors?|"
      r"monitors?(?<!baby monitor)(?<!heart rate monitor)(?<!pressure monitor)(?<!studio monitor)"
      r"(?<!tire pressure monitor)|portable (?:displays?|screens?))\b"),
    R("tv_video",
      r"\b(?:tvs?(?! (?:mounts?|stands?|antennas?|remotes?|box(?:es)?|brackets?|wall|cables?|"
      r"covers?|cabinets?|trays?|shelf|shelves|consoles?|lifts?))|televisions?|hdtvs?|smart tvs?|"
      r"(?:led|lcd|"
      r"oled|qled|4k|uhd|hd) tvs?|projectors?(?! (?:screens?|mounts?|lamps?|bulbs?|brackets?|"
      r"stands?|ceiling))|"
      r"(?:streaming|media) (?:players?|sticks?|devices?|box(?:es)?)|roku(?: (?:express|ultra|"
      r"streaming stick|premiere))?(?= *$|,| -| \()|fire ?tv(?: stick)?(?: 4k)?(?= *$|,| -| \(|"
      r" with)|"
      r"chromecast(?: with google tv)?|apple tv(?: 4k)?(?= *$|,| -| \()|"
      r"(?:android|smart) tv box(?:es)?|tv box(?:es)?|"
      r"(?:dvd|blu-?ray|vcr|vhs|dvd/vcr|dvd-vcr) (?:players?|recorders?|combos?|combo players?|"
      r"systems?)|vcrs?|dvrs?(?<!security dvr)|"
      r"projection screens?|projector screens?)\b"),
    R("cameras",
      r"\b(?:cameras?|camcorders?|dslrs?|slrs?|mirrorless|action cams?|gopro(?: hero\s?\d*\w*)?|"
      r"instax(?: mini)?(?: \d+)?(?= *$|,| -| \(| instant| camera)|polaroid(?: now|"
      r" onestep)?(?= *$|,| -| \()|"
      r"(?:video|digital|film|instant|kids|point and shoot|trail|game|hunting|underwater|360|"
      r"vlogging|cinema|webcam) cams?|drones?|quadcopters?|camera drones?|"
      r"(?:digital|video|film|instant|point and shoot|trail|game|hunting|underwater|360|vlogging|"
      r"cinema|action|disposable|kids|toy) cameras?)\b"),
    R("watches",
      r"\b(?:smart ?watch(?:es)?|smartwatch(?:es)?|wrist ?watch(?:es)?|wristwatch(?:es)?|"
      r"(?:fitness|activity|health|sleep|step) trackers?|pedometers?|"
      r"(?:gps|golf|running|sports?|digital|analog|analogue|kids|military|tactical|diving|dive|"
      r"pocket|nurse|automatic|mechanical|quartz|dress|casual|luxury|men's|women's|mens|womens|"
      r"unisex|chronograph|solar|atomic|calling|phone) watch(?:es)?|"
      r"apple watch(?: series \d+| se| ultra)?(?: (?:\d\d ?mm|gps|cellular|\(gps\)))*(?= *$|,| -|"
      r" \()|"
      r"galaxy watch\s?\d*(?: (?:active|classic|pro))?\s?\d*(?: ?\d\d ?mm)?(?= *$|,| -| \(|"
      r" smartwatch| bluetooth| lte)|"
      r"(?:fitbit|garmin|amazfit|polar|suunto|coros|withings|fossil|ticwatch|huawei) (?:"
      r"(?:charge|versa|sense|inspire|alta|luxe|ionic|blaze|surge|flex|ace|forerunner|vivoactive|"
      r"vivomove|vivosmart|vivofit|fenix|venu|instinct|epix|enduro|bip|gtr|gts|band|t-rex|"
      r"ignite|vantage|grit|unite|pace|apex|vertix|scanwatch|gen)\b)(?: \d+\w*)?(?: (?:hr|plus|pro|"
      r"lite|se|active|classic|s|x|solar|sapphire))?"
      r"(?= *$|,| -| \(| smart| fitness| gps| activity| health| \d+ ?mm)|"
      r"heart rate monitors?|watch(?:es)?(?= *$))\b"),
    R("smart_home",
      r"\b(?:(?:security|surveillance|cctv|ip|wifi|wi-fi|wireless|outdoor|indoor|ptz|dome|bullet|"
      r"spy|hidden|nanny|doorbell|baby|pet|home|floodlight|spotlight|poe|4g|lte|solar) (?:cam|cams|"
      r"cameras?)|nanny cams?|doorbells?|video doorbells?|"
      r"smart (?:plugs?|bulbs?|switch(?:es)?|locks?|thermostats?|speakers?|displays?|lights?|"
      r"outlets?|sensors?|home hubs?|hubs?|smoke detectors?|blinds?)|"
      r"echo(?: dot| show| pop| studio| plus| flex)?(?= *$|,| -| \(| \d| with| smart)|"
      r"google (?:home|nest)(?: mini| hub| audio)?|nest (?:cam|thermostat|hub|protect|mini|wifi)|"
      r"(?:door|window|motion|contact|water leak|leak|smoke|temperature|humidity|glass break) "
      r"(?:sensors?|detectors?|alarms?)|alarm systems?|security systems?|"
      r"(?:nvr|dvr) (?:systems?|kits?)|security dvrs?|nvrs?|baby monitors?|video baby monitors?|"
      r"intercoms?|(?:pen|spy|button|clock|usb|charger|smoke detector) cams?|(?:smoke|"
      r"carbon monoxide|co) (?:detectors?|alarms?)|(?:smart )?thermostats?|"
      r"(?:range|wifi) extender plugs?|motion detectors?|smart home)\b"),
    R("networking",
      r"\b(?:routers?|modems?|(?:wifi|wi-fi|wireless|range|network|signal|internet|mesh|"
      r"cell(?:ular)?|"
      r"cell phone|repeater) (?:extenders?|repeaters?|boosters?|amplifiers?)|range extenders?|"
      r"mesh (?:wi-?fi|systems?|routers?|networks?)|access points?|"
      r"(?:network|ethernet|gigabit|poe|managed|unmanaged|lan) switch(?:es)?|"
      r"powerline(?: adapters?| kits?)?|"
      r"(?:wifi|wi-fi|wireless|usb|network|ethernet|lan|bluetooth|gigabit) (?:adapters?|cards?|"
      r"dongles?)|(?:nic|network interface) cards?|hotspots?|mifi|nas(?: storage| servers?)?|"
      r"print servers?|(?:cable|dsl|fiber) modems?)\b"),
    R("gaming",
      r"\b(?:(?:game|gaming|video game|handheld|retro|portable|arcade) (?:consoles?|systems?|"
      r"machines?)|consoles?(?<!mixing console)(?<!dj console)|"
      r"(?:playstation|ps)\s?[2345](?: (?:console|pro|slim|digital edition|disc edition))*(?= *$|,|"
      r" -| \()|"
      r"xbox(?: one| series [xs]| 360)?(?: [xs])?(?: console)?(?= *$|,| -| \()|"
      r"nintendo switch(?: (?:oled|lite))?(?: console)?(?= *$|,| -| \(| with)|nintendo (?:3ds|2ds|"
      r"ds|wii|wii u)(?= *$|,| -| \()|"
      r"(?:game|gaming|wireless|pro|xbox|ps[2345]|playstation|switch|pc|bluetooth) (?:controllers?|"
      r"gamepads?|joysticks?)|controllers?(?! (?:holder|stand|charger|charging|skin|grips?|"
      r"thumb|case|mount|cover))|gamepads?|joysticks?|joy-?cons?|(?:racing|steering) wheels?|"
      r"(?:vr|virtual reality) (?:headsets?|glasses|goggles|systems?)|oculus(?: quest\s?\d*|"
      r" rift\w*| go)?(?= *$|,| -| \()|"
      r"meta quest\s?\d*|steam deck|(?:sega )?genesis mini|snes classic|nes classic(?: edition)?|"
      r"playstation classic|arcade (?:machines?|cabinets?|sticks?)|fight sticks?)\b"),
    R("car",
      r"\b(?:car (?:stereos?|radios?|audio|amplifiers?|amps?|speakers?|subwoofers?|dvd players?|"
      r"video|play adapters?|play)|(?:double|single) din(?: car)?(?: stereos?| radios?|"
      r" head units?)?|"
      r"head units?|dash ?cams?|dashcams?|dashboard cameras?|(?:backup|back-up|reverse|reversing|"
      r"rear ?view|rearview|license plate) (?:cameras?|cams?|systems?|mirrors?)|"
      r"radar(?: and laser)? detectors?|radar/laser detectors?|laser detectors?|chart ?plotters?|"
      r"gps(?: navigators?| navigation(?: systems?)?| units?| trackers?| devices?)|"
      r"(?:sat ?nav|navigators?)|obd-?(?:ii|2) (?:scanners?|readers?|adapters?)|obd2? scanners?|"
      r"code readers?|car (?:alarms?|security systems?|starters?)|remote (?:car )?starters?|"
      r"(?:sub|subwoofer) (?:box(?:es)?|enclosures?)|amp(?:lifier)? wiring kits?|"
      r"fm transmitters?|cb radios?|marine (?:stereos?|radios?|speakers?)|fish ?finders?|"
      r"tire pressure monitors?|tpms|(?:car|vehicle|truck) (?:gps|trackers?|tv))\b"),
    R("office",
      r"\b(?:printers?(?<!3d printer)|3d printers?|(?:photo|label|receipt|thermal|laser|inkjet|"
      r"all-in-one|portable|sticker|barcode|instant) printers?|"
      r"scanners?(?<!obd2 scanner)(?<!obd scanner)|"
      r"(?:document|photo|film|slide|barcode|flatbed) scanners?|label (?:makers?|machines?)|"
      r"calculators?|shredders?|laminators?|copiers?|fax machines?|typewriters?|"
      r"(?:electronic|digital) (?:dictionar(?:y|ies)|translators?|organi[sz]ers?)|"
      r"(?:time|punch) clocks?|money counters?|bill counters?|cash registers?|pos systems?|"
      r"(?:digital|lcd|electronic) (?:notepads?|writing (?:tablets?|boards?))|"
      r"(?:lcd )?writing tablets?|smart (?:notebooks?|pens?)|e-?ink (?:notebooks?|tablets?)|"
      r"(?:digital )?(?:photo|picture) frames?|presentation clickers?|"
      r"(?:alarm|digital|wall|desk|projection|atomic) clocks?)\b"),
    R("optics",
      r"\b(?:binoculars?|monoculars?|telescopes?|microscopes?|spotting scopes?|rangefinders?|"
      r"range finders?|night vision(?: goggles| monoculars?| binoculars?| scopes?| devices?)?|"
      r"thermal (?:imagers?|imaging cameras?|monoculars?)|magnifiers?|magnifying glass(?:es)?|"
      r"loupes?|(?:rifle )?scopes?(?<!spotting scope)(?<!oscilloscope)(?<!finder scope))\b"),
    R("lighting",
      r"\b(?:(?:led )?(?:strip|rope|string|fairy|neon|rgb|tape) lights?|led strips?|light strips?|"
      r"flashlights?|torches?|headlamps?|lanterns?|night ?lights?|(?:desk|table|floor|reading|"
      r"bedside|clip|book|piano|architect|uv|sun|sad|therapy|lava|salt) lamps?|"
      r"light bulbs?|(?:led|smart|edison|halogen|incandescent|fluorescent|uv|black) (?:bulbs?|"
      r"lights?)|"
      r"work lights?|flood ?lights?|spot ?lights?|(?:camping|emergency|solar|outdoor|underwater|"
      r"dive|diving|"
      r"motion sensor|puck|cabinet|closet|grow|disco|party|stage|dj|laser) lights?|"
      r"light bars?|(?:light|lighting) kits?(?<!studio lighting kit))\b"),
    R("cases",
      r"\b(?:(?<!charging )(?<!carrying )(?<!charger )(?<!battery )(?<!travel )(?<!storage )"
      r"cases?(?<!pc case)(?<!computer case)(?<!atx case)(?<!tower case)|covers?(?<!dust cover)|"
      r"skins?|decals?|stickers?|bumpers?|shells?(?<!egg shells)|folios?|"
      r"(?:protective|silicone|leather|rugged|hard|soft|clear|tpu|wallet|flip|folio|keyboard|"
      r"armor|shockproof|kids|kid-proof|kidproof|book|smart) (?:cases?|covers?)|"
      r"(?:otterbox|lifeproof|spigen|supcase|i-blason|incipio|speck|urban armor gear|uag|"
      r"zagg|tech21|mous|pelican)(?: (?:defender|commuter|symmetry|statement|prefix|fre|nuud|"
      r"ultra hybrid|tough armor|rugged armor|liquid air|unicorn beetle|monarch|plasma|"
      r"pathfinder|civilian|metropolis|presidio|candyshell|duraflex)\b(?: (?:series|pro|xt|"
      r"plus))?)?"
      r"(?= *$|,| -| \(| for| case| series)|"
      r"(?:defender|commuter|symmetry) series|dust (?:covers?|plugs?)|keyboard (?:covers?|skins?|"
      r"protectors?)|"
      r"arm ?bands?(?! (?:for (?:apple )?watch|compatible with (?:apple )?watch)))\b"),
    R("cases",
      r"\b(?:charging|carrying|charger|battery|travel|storage) cases?\b", weak=True),
    R("screen_protectors",
      r"\b(?:screen protectors?|(?:tempered |hydrogel |privacy |anti-?glare |matte |hd |clear )?"
      r"(?:glass|film|screen|lens|camera lens) protectors?|protectors?(?<!surge protector)"
      r"(?<!cord protector)(?<!cable protector)(?<!voltage protector)(?<!keyboard protector)"
      r"(?<!grille protector)(?<!grill protector)|"
      r"screen (?:guards?|films?|shields?|protection)|tempered glass(?: films?| screens?)?|"
      r"privacy (?:screens?|filters?)|anti-?glare (?:screens?|filters?|films?)|"
      r"(?:protective|protection) films?|hydrogel films?|blue light (?:filters?|screens?))\b"),
    R("power",
      r"\b(?:chargers?|charging (?:stations?|docks?|stands?|pads?|hubs?|bases?|cradles?|mats?|"
      r"blocks?|bricks?|adapters?|kits?|towers?)|power (?:banks?|adapters?|adaptors?|supply|"
      r"supplies|cords?|strips?|stations?|inverters?|centers?|bricks?|blocks?|outlets?|"
      r"converters?|packs?|plugs?|taps?|modules?|transformers?)|powerbanks?|"
      r"(?:ac|dc|ac/dc|ac-dc|ac dc|laptop|wall|usb|travel|car|universal|switching|replacement) "
      r"(?:adapters?|adaptors?|chargers?)|ac adapter chargers?|surge protectors?|"
      r"surge suppressors?|"
      r"extension (?:cords?|leads?|sockets?)|ups(?: battery(?: backup)?)?(?= *$|,| -| \(| \d)|"
      r"battery (?:backups?|packs?|chargers?|banks?)|(?:portable )?power stations?|inverters?|"
      r"solar (?:panels?|chargers?|generators?|power banks?)|(?:voltage|step-?up|step-?down) "
      r"(?:converters?|transformers?|regulators?)|transformers?|outlet (?:extenders?|adapters?)|"
      r"wall (?:plugs?|taps?|outlets?)|(?:travel|international|universal) (?:plugs?|adapters?)|"
      r"power cables?|ac (?:power )?cords?|power$|psus?|(?:ac )?(?:charger|power) (?:adapters?|"
      r"adaptors?)|"
      r"pwr supply|surge guards?|magsafe(?: chargers?)?(?= *$|,| -| \())\b"),
    R("batteries",
      r"\b(?:batter(?:y|ies)|(?:aa|aaa|9v|cr\d{4}|lr\d+|18650|21700|lithium|alkaline|rechargeable|"
      r"button|coin|li-?ion|nimh|ni-mh|lipo) (?:cells?|batter(?:y|ies))|cr\d{4}|lr\d{2,4})\b"),
    R("batteries",
      r"\bbatter(?:y|ies)\b", weak=True),
    R("cables",
      r"\b(?:cables?|cords?|wires?(?<!speaker wire)|leads?(?<!\d leads)|adapters?|adaptors?|"
      r"dongles?|hubs?(?<!smart home hub)|splitters?|extenders?|converters?|switch(?:es|ers?)?"
      r"(?<!nintendo switch)(?<!smart switch)(?<!light switch)(?<!network switch)|"
      r"connectors?|couplers?|jacks?|plugs?|terminals?|patch (?:panels?|cords?)|keystones?|"
      r"(?:usb|hdmi|vga|dvi|displayport|thunderbolt|lightning|usb-c|type-c|ethernet|audio|"
      r"optical|toslink|rca|coaxial|coax|aux|sata|speaker|xlr|mmcx|banana|f-type) "
      r"(?:cables?|cords?|wires?|adapters?|hubs?|switch(?:es)?|splitters?|extenders?|"
      r"converters?|connectors?|plugs?)|kvm(?: switch(?:es)?)?|"
      r"cable (?:management|organi[sz]ers?|ties?|clips?|sleeves?|raceways?|covers?|boxes?|trays?)|"
      r"zip ties?|wire (?:looms?|ties?|connectors?|nuts?)|heat shrink(?: tubing)?|"
      r"(?:usb|hdmi|vga) (?:a|c|male|female)|y-cables?|otg|wire looms?|split looms?|conduits?|"
      r"(?:\d[- ]gang |speaker |hdmi |av |keystone |decora )?wall plates?|binding posts?|"
      r"sleeve tubes?|tubing)\b"),
    R("computer_accessories",
      r"\b(?:keyboards?(?<!piano keyboard)(?<!midi keyboard)(?<!digital keyboard)|"
      r"(?<!mickey )(?<!minnie )(?<!danger )mouse|mice|mouse ?pads?|mousepads?|desk ?pads?|"
      r"desk mats?|"
      r"webcams?|web cams?|web cameras?|docking stations?|docks?(?<!charging dock)|"
      r"(?:laptop|notebook) (?:cooling pads?|coolers?)|cooling pads?|trackpads?|touchpads?|"
      r"trackballs?|styluses?|stylus(?: pens?)?|(?:digital|touch|active|capacitive) pens?|"
      r"apple pencils?|(?:drawing|graphics|graphic|pen|digital art) tablets?|"
      r"pen displays?|capture cards?|video capture(?: cards?| devices?)?|presenters?|"
      r"laser pointers?|wrist rests?|keycaps?|key caps|numeric keypads?|number pads?|keypads?|"
      r"usb fans?|desk fans?|monitor (?:light|lamp)s?|privacy filters?|card readers?|"
      r"(?:usb|sd|micro ?sd|memory card) readers?|fingerprint readers?|security keys?|"
      r"cooling fans? for laptops?|port replicators?)\b"),
    R("components",
      r"\b(?:(?:graphics|video|sound|gpu) cards?|gpus?|motherboards?|mainboards?|logic boards?|"
      r"(?:desktop|laptop|mobile|server|intel|amd) (?:processors?|cpus?)|"
      r"(?:cpu|case|pc|computer|cooling|radiator|exhaust|laptop|replacement|"
      r"internal|rgb) fans?|cpu coolers?|(?:liquid|aio|water) coolers?|heat ?sinks?|heatsinks?|"
      r"thermal (?:paste|compound|pads?)|(?:pc|computer|atx|micro atx|matx|itx|mini itx|"
      r"mid[- ]tower|full[- ]tower|gaming|tower|server|htpc) (?:cases?|chassis)|chassis|"
      r"(?:ram|memory) (?:modules?|kits?|sticks?|upgrades?)|(?<!gb )(?<!\dgb)ddr\d(?:l)?(?: (?:ram|"
      r"memory|sdram|"
      r"so-?dimm|dimm))?|so-?dimms?|dimms?|sdram|"
      r"(?:replacement |laptop |lcd |led |touch )?(?:lcd|led|oled) (?:screens?|displays?|panels?)"
      r"(?: (?:replacement|assembly|digitizer))?|digitizers?|(?:screen|display|lcd) (?:replacement|"
      r"assembly|assemblies)|replacement (?:screens?|keyboards?|parts?|hinges?|fans?|speakers?|"
      r"lcds?|displays?|batter(?:y|ies)|housings?|bezels?|covers?|shells?|buttons?|ribbons?|"
      r"cables?|antennas?|lens(?:es)?|boards?|motors?)|hinges?|bezels?|palmrests?|palm rests?|"
      r"(?:dc|power) (?:jacks?|connectors?|sockets?)(?= *$|,| -| \(| for| with| cable| harness)|"
      r"ribbon cables?|flex cables?|ear ?(?:pads?|cushions?|tips?|hooks?|gels?|wings?|cups?)|"
      r"eartips|earpads|cushions?|caddy|caddies|(?:hdd|ssd|hard drive) (?:caddy|trays?|brackets?)|"
      r"raid controllers?|(?:pcie?|pci express|pci-e|m\.2|nvme|sata) (?:cards?|adapters?|"
      r"risers?|expansion cards?|controllers?)|riser cards?|expansion cards?|"
      r"(?:laser|optical) (?:lens|pickups?)|power buttons?|"
      r"(?:speaker|subwoofer|woofer|tweeter|midrange) (?:drivers?|cones?)|tweeters?|woofers?|"
      r"crossovers?|circuit boards?|pcbs?|modules?(?<!memory module)|relays?|"
      r"arduino|raspberry pi(?: \d)?(?: model \w+)?|microcontrollers?|development boards?|"
      r"breadboards?|resistors?|capacitors?|transistors?|diodes?|sensors? modules?|"
      r"replacement parts?|spare parts?|repair parts?|parts?(?= *$)|grill(?:e)?s?|main ?boards?|"
      r"(?:mid|full|mini|micro)[- ]towers?|(?:liquid|water) cooling(?: systems?| kits?| loops?)?|"
      r"(?:projector|projection|tv|television|rptv|dlp) (?:bare )?(?:lamps?|bulbs?)|"
      r"(?:lamp|bulb|projector|tv) assembl(?:y|ies)|bare (?:lamps?|bulbs?)|replacement lamps?|"
      r"dash kits?|(?:stereo|radio|dash) "
      r"install(?:ation)? kits?|"
      r"(?:wiring )?harness(?:es)?(?<!camera harness)(?<!camera harnesses))\b"),
    R("components",
      r"\b(?:\d+ ?(?:gb|tb|mb) (?:ddr\d )?(?:ram|memory|sdram))\b", weak=True),
    R("storage",
      r"\b(?:(?:external|portable|internal|desktop|laptop|usb|sata|ssd|solid state|nas|"
      r"surveillance|gaming)? ?hard (?:drives?|disks?)(?<!gb hard drive)(?<!tb hard drive)|"
      r"(?<!gb )(?<!tb )(?<!\dgb)(?<!\dtb)(?:hdds?|ssds?)|solid state drives?|"
      r"(?:usb |pen |thumb |jump |flash |memory |usb flash |usb c flash |otg flash )drives?|"
      r"flash (?:drives?|disks?|memory)|thumbdrives?|pendrives?|"
      r"(?:memory|micro ?sd|sd|sdhc|sdxc|micro ?sdhc|micro ?sdxc|cf|compact ?flash|cfast|"
      r"cfexpress|xqd|tf|sim|xd) cards?|microsd(?:hc|xc)?|sdhc|sdxc|memory sticks?|"
      r"(?:blank )?(?:dvds?|cds?|blu-?rays?|bd-?rs?|dvd-?rs?|dvd\+rs?|cd-?rs?|cd-?rws?|"
      r"m-discs?|discs?|disks?|media)(?<!dvd player)(?=(?: \w+){0,3}? (?:spindles?|packs?|discs?|"
      r"pack|blank|media))|blank (?:dvds?|cds?|discs?|media|blu-?rays?|bd-?r)|"
      r"(?:hard drive|hdd|ssd|drive|disk|m\.2|nvme) (?:enclosures?|docks?|docking stations?|"
      r"cases?|adapters?)|enclosures?(?<!sub enclosure)(?<!subwoofer enclosure)|"
      r"(?:usb|external|portable) (?:storage|drives?)|floppy (?:disks?|drives?)|"
      r"zip disks?|(?:dvd|cd|blu-?ray|optical) (?:drives?|burners?|writers?|rw drives?)|"
      r"(?:sd|sdhc|sdxc|microsd|micro sd|tf|cf) (?:plus |ultra |extreme |pro |high speed |"
      r"memory )cards?|"
      r"network attached storage|cassettes?(?: tapes?)?|(?:audio|video|vhs|dv|mini ?dv) tapes?)\b"),
    R("storage",
      r"\b(?:\d+(?:\.\d+)? ?(?:gb|tb) (?:ssd|hdd|emmc|storage|solid state drive|hard drive|"
      r"pcie ssd|nvme ssd|m\.2 ssd|flash storage|sd card|micro ?sd card|"
      r"memory card))\b", weak=True),
    R("camera_accessories",
      r"\b(?:lens(?:es)?(?<!contact lens)(?<!contact lenses)|(?:camera|telephoto|wide[- ]angle|"
      r"macro|fisheye|zoom|prime|portrait|kit|phone|anamorphic|cine) lens(?:es)?|"
      r"lens (?:caps?|hoods?|covers?|shades?|filters?|adapters?|mounts?|pouches?|cleaning(?: kits?|"
      r" pens?)?|"
      r"cloths?|tissues?|kits?)|lens(?:es)? kit|"
      r"tripods?|monopods?|gimbals?|(?:camera |video |phone |handheld )?stabili[sz]ers?|"
      r"selfie (?:sticks?|rings?(?: lights?)?)|(?:camera|speedlight|speedlite|external|"
      r"on-camera|macro|ring|slave|studio) flash(?:es)?|flash (?:units?|guns?|diffusers?|triggers?|"
      r"brackets?)|speedlights?|speedlites?|flash$|"
      r"(?:uv|nd|cpl|polari[sz]ing|polari[sz]er|variable nd|star|graduated|close-up|lens|camera|"
      r"red|ir|infrared|diffusion|mist|step-up|step up|step-down) filters?|filter (?:kits?|sets?|"
      r"holders?|systems?|rings?|adapters?)|"
      r"(?:camera|neck|shoulder|wrist|hand|sling|dslr) straps?(?<!watch strap)|"
      r"(?:camera|battery|vertical|hand) grips?|cages?(?<!rabbit cage)|follow focus|"
      r"matte box(?:es)?|"
      r"(?:ring|led|studio|video|photography|photo|continuous|vlogging|selfie|fill|panel|"
      r"key|on-camera|softbox|umbrella) lights?(?: kits?)?|(?:studio |photography |lighting )?"
      r"softbox(?:es)?(?: kits?)?|(?:light|lighting|c|background|backdrop) stands?|"
      r"(?:photo|photography|shoot-through|reflective|reflector) umbrellas?|reflectors?|"
      r"diffusers?|light tents?|photo (?:box(?:es)?|studios?|tents?|light box(?:es)?)|"
      r"light ?box(?:es)?|"
      r"(?:instant|instax|polaroid|35mm|120|camera|color|black and white|slide|negative|"
      r"photographic) films?|film (?:rolls?|packs?)|(?:remote )?shutter (?:releases?|remotes?|"
      r"cables?)|cable releases?|intervalometers?|viewfinders?|eye ?cups?|eyecups?|"
      r"(?:lens|mount|camera) adapters?|extension tubes?|teleconverters?|focal reducers?|"
      r"quick release (?:plates?|clamps?)|(?:ball|pan|fluid|tripod|gimbal) heads?|"
      r"(?:camera|cold shoe|hot shoe) mounts?|hot shoes?|cold shoes?|camera (?:rigs?|sliders?|"
      r"cranes?|dollies|dolly|clamps?|accessories|remotes?|cleaning kits?|rain covers?|"
      r"harness(?:es)?)|"
      r"(?:video |camera |boom )?sliders?|clapper ?boards?|light meters?|grey cards?|gray cards?|"
      r"color checkers?|lavalier mic(?:rophone)? kits?|teleprompters?|(?:print|reversal|"
      r"b&w) films?|"
      r"filters?(?<!air filter)(?<!water filter)(?<!bandpass filter)(?<!privacy filter)"
      r"(?<!pass filter)(?<!noise filter)(?<!line filter)(?<!emi filter)|polari[sz]ers?|"
      r"eye ?pieces?|view ?finders?|sun hoods?|lcd (?:hoods?|shades?)|pop-?up shades?|"
      r"(?:photo )?shoot(?:ing)? tents?|pop filters?|wind ?screens?|wind ?muffs?|dead ?cats?|"
      r"shock ?mounts?|wind ?busters?)\b"),
    R("backdrops",
      r"\b(?:backdrops?|backgrounds?(?<!white background)|(?:green|blue|chroma key|white|black) "
      r"screens?(?= *$|,| -| \(| backdrop| background| for| kit)|photo (?:booth|studio) props?|"
      r"photo ?booth(?: props?| frames?| backdrops?)?|(?:photography|photo|studio|party) props?|"
      r"(?:birthday|party|wedding|baby shower|photo) banners?|banners?|selfie frames?|"
      r"seamless (?:paper|backgrounds?)|muslin(?: backdrops?)?|"
      r"(?:party|birthday) decorations?)\b"),
    R("bags",
      r"\b(?:bags?(?<!air bag)(?<!bean bag)(?<!sand bag)(?<!dry bag)(?<!zip bag)|backpacks?|"
      r"sleeves?(?<!cable sleeve)(?<!short sleeve)(?<!long sleeve)|pouch(?:es)?|briefcases?|"
      r"messenger bags?|totes?|duffels?|duffles?|satchels?|organi[sz]er (?:bags?|pouch(?:es)?|"
      r"cases?)|(?:carrying|travel|storage|camera|eva|gig) (?:bags?|cases?)|gig ?bags?|"
      r"hard (?:carrying|travel|storage) cases?|rolling cases?|trolley cases?|"
      r"dry bags?|waterproof (?:pouch(?:es)?|phone pouch(?:es)?)|fanny packs?|waist packs?|"
      r"sling packs?|camera inserts?|lens pouch(?:es)?)\b"),
    R("mounts",
      r"\b(?:mounts?|mounting (?:brackets?|kits?|plates?|arms?|systems?)|brackets?|"
      r"arms?(?<!fire arms)|clamps?|cradles?|risers?(?<!pcie riser)|tv lifts?|lift mechanisms?|"
      r"(?:phone|cell phone|tablet|ipad|laptop|monitor|tv|headphone|speaker|car|bike|bicycle|"
      r"motorcycle|desk|wall|ceiling|vent|dash|dashboard|windshield|cup holder|magnetic|"
      r"suction|gooseneck|floor) (?:mounts?|stands?|holders?|arms?|brackets?|clips?|docks?|"
      r"cradles?|risers?)|pop ?sockets?|popsockets?|phone grips?|ring holders?|"
      r"(?:finger|phone) rings?|kickstands?|shelf|shelves|"
      r"(?:headphone|headset|controller) (?:hangers?|hooks?)|lap desks?|laptop (?:desks?|tables?|"
      r"trays?))\b"),
    R("av_accessories",
      r"\b(?:remotes?(?! (?:car|starters?|shutter|controlled|control (?:car|truck|boat|"
      r"helicopter|drone|toy|robot)))|remote controls?(?! (?:car|truck|boat|helicopter|drone|"
      r"toy|robot|plane|airplane))|remote controllers?|replacement remotes?|universal remotes?|"
      r"(?:tv|hdtv|digital|indoor|outdoor|amplified|fm|am|am/fm|car|wifi|wi-fi|gps|"
      r"radio|uhf|vhf|hd|ota|attic|rooftop|dipole|yagi|marine|cb|ham) antennas?|antennas?|"
      r"aerials?|ir (?:blasters?|extenders?|receivers?|repeaters?)|(?:hdmi|av|a/v) receivers?|"
      r"3d glasses)\b"),
    R("watch_accessories",
      r"\b(?:watch ?bands?|watchbands?|watch ?straps?|watchstraps?|watch (?:bracelets?|loops?|"
      r"links?|winders?|box(?:es)?|cases?|rolls?|pouch(?:es)?|stands?|holders?|displays?|"
      r"organi[sz]ers?|storage|trays?|pillows?|cushions?|tools?|repair kits?|kits?|batter(?:y|ies)|"
      r"crystals?|glass|bezels?|faces?|parts?|movements?|chargers?|charging (?:docks?|stands?|"
      r"cables?)|docks?|chains?|fobs?|clasps?|buckles?|pins?|spring bars?)|"
      r"watch winders?|winders?|(?:replacement|silicone|sport|leather|nylon|metal|stainless steel|"
      r"milanese|mesh|magnetic|elastic|stretch|braided|solo|sport loop|link|resin|ceramic|"
      r"titanium|canvas|woven|rubber|fabric|velcro|nato|zulu|perlon|bracelet) (?:bands?|straps?|"
      r"wristbands?|loops?)(?<!rubber band)(?<!elastic band)|wristbands?|"
      r"bands?(?<!rubber band)(?<!elastic band)(?<!resistance band)(?<!head band)(?<!hair band)"
      r"(?<!arm band)(?<!dual band)(?<!tri band)(?<!tri-band)(?<!dual-band)|straps?|sport loops?|"
      r"spring bars?|link (?:pins?|removers?)|(?:watch|band|strap) (?:link )?(?:removers?|tools?|"
      r"adjusters?)|case openers?|(?:watch|jewelry|jewellery) (?:cleaners?|cleaning (?:kits?|"
      r"cloths?|solutions?))|polishing cloths?)\b"),
    R("supplies",
      r"\b(?:ink(?: cartridges?| tanks?| refills?| bottles?)?|toner(?: cartridges?)?|"
      r"(?:printer|replacement|remanufactured|compatible) cartridges?|"
      r"cartridges?(?<!game cartridge)|"
      r"drum units?|printheads?|print heads?|ribbons?|(?:label|labeling|labelling|tze|dk) "
      r"(?:tapes?|rolls?|refills?|cartridges?)|labels?|(?:photo|printer|thermal|receipt|copy|"
      r"instant|zink|sticker|transfer|sublimation) paper|paper rolls?|filaments?|"
      r"(?:pla|abs|petg|tpu) filaments?|resin(?= *$|,| -| \())\b"),
    R("tools",
      r"\b(?:(?:screen|lens|keyboard|electronics?|computer|camera|sensor|record|vinyl|cd|dvd|"
      r"laptop|phone|tv) (?:cleaners?|cleaning (?:kits?|solutions?|sprays?|cloths?|wipes?|pens?|"
      r"brushes?|swabs?|gel|putty))|cleaning (?:kits?|cloths?|wipes?|swabs?|gel|putty|brushes?|"
      r"pens?|"
      r"cards?|discs?|tapes?|solutions?)|compressed air|air dusters?|(?:electric |"
      r"cordless )?dusters?|"
      r"microfiber cloths?|(?:tool|repair|precision|screwdriver|soldering|crimping|electronics|"
      r"computer|phone|watch|pry|opening|network|cable) (?:kits?|sets?|tools?)|toolkits?|"
      r"screwdrivers?|soldering (?:irons?|stations?|kits?)|solder|crimpers?|crimping tools?|"
      r"wire (?:strippers?|cutters?)|multimeters?|(?:cable|network|lan|battery|voltage|outlet|"
      r"usb|continuity) testers?|testers?|oscilloscopes?|tweezers|anti-?static (?:wrist )?"
      r"(?:straps?|mats?|bags?|wristbands?)|esd (?:mats?|straps?|bags?)|pry tools?|spudgers?|"
      r"heat guns?|magnifying lamps?|helping hands?|(?:label|cable) tags?|"
      r"(?:stud|metal|cable|wire|voltage) (?:finders?|detectors?)|metal detectors?)\b"),
    R("service",
      r"\b(?:protection plans?|(?:extended |accident(?:al)? )(?:protection|warranty)(?: plans?)?|"
      r"warranty(?: plans?)?(?= *$|,| -| \(| for| \d)|extended warrant(?:y|ies)|applecare\+?|"
      r"service plans?|"
      r"(?:\d|one|two|three|four|five)[- ](?:year|yr)s? (?:protection|warranty|plan))\b"),
    R("mounts",
      r"\b(?:stands?|holders?(?<!card holder))\b", weak=2),
    R("components",
      r"\b(?:cpus?|processors?)\b", weak=True),
    R("lighting",
      r"\b(?:lamps?|bulbs?)\b", weak=2),
]

# ====================================================================== GROCERY
# key: (shopper-facing label, role food / non_food). cake_decor is the Grocery
# vertical's real block of cake toppers and party decor (5.1% of rows).
_GROCERY_TABLE = {
    "coffee": ("Coffee", "food"),
    "tea": ("Tea", "food"),
    "beverages": ("Drinks & drink mixes", "food"),
    "snacks": ("Chips, crackers & snacks", "food"),
    "candy": ("Candy & chocolate", "food"),
    "bakery": ("Cookies, cakes & bakery", "food"),
    "baking": ("Baking supplies", "food"),
    "spices": ("Spices, herbs & seasonings", "food"),
    "sauces": ("Sauces, condiments & oils", "food"),
    "pantry": ("Pasta, rice, grains & beans", "food"),
    "breakfast": ("Breakfast, cereal & spreads", "food"),
    "canned": ("Soups, meals & canned goods", "food"),
    "nuts": ("Nuts, seeds & dried fruit", "food"),
    "protein": ("Protein & nutrition", "food"),
    "fresh": ("Meat, seafood, dairy & produce", "food"),
    "gifts": ("Gift baskets & assortments", "food"),
    "cake_decor": ("Cake toppers & party decor", "non_food"),
    "non_food": ("Kitchen, pet & other non-food", "non_food"),
}

GROC_RULES = [
    R("coffee",
      r"\b(?:coffees?|espresso(?! (?:machines?|makers?))|k-?cups?|kcups?|coffee pods?|"
      r"nespresso(?: pods?|"
      r" capsules?| compatible)?|keurig|cold brew|(?:whole|coffee) beans?|ground coffee|"
      r"instant coffee|decaf(?:feinated)?(?= *$|,| -| \(| coffee| blend| roast)|"
      r"(?:light|medium|dark|french|italian|breakfast|house) (?:roast|blend)|roast(?= *$|,| -| \()|"
      r"cappuccino|lattes?(?<!tea latte)(?<!chai latte)(?<!matcha latte)|mochas?(?! (?:sauce|"
      r"syrup))|"
      r"coffee (?:creamers?|syrups?)|creamers?|chicory|cafe(?= *$)|frappe(?: mix)?)\b"),
    R("tea",
      r"\b(?:teas?(?! (?:spoons?|towels?|cups?|pots?|kettles?|lights?|cakes?|sets?|infusers?|"
      r"strainers?|party|time|cookies?|biscuits?|sandwich(?:es)?))|tea ?bags?|loose ?leaf|"
      r"matcha|chai(?! (?:spice|seeds?))|rooibos|yerba ?mate|mate(?= *$|,| -| \()|"
      r"oolong|pu-?erh|earl grey|darjeeling|assam|sencha|genmaicha|hojicha|chamomile|"
      r"tisanes?|herbal (?:infusions?|blends?)|kombucha(?! (?:scoby|starter))|bubble tea|boba|"
      r"tea (?:lattes?|concentrates?|powders?)|iced tea)\b"),
    R("beverages",
      r"\b(?:juices?|sodas?|soft drinks?|drinks?|beverages?|(?:sparkling|mineral|spring|alkaline|"
      r"coconut|flavored|still|tonic|soda|seltzer|vitamin|smart|bottled|mineral|infused|cactus|"
      r"maple|birch|aloe|artesian|purified|distilled|electrolyte|structured|hydrogen|boxed|"
      r"canned) waters?|"
      r"seltzers?|tonics?|lemonades?|limeades?|punch|nectars?|smoothies?|shakes?(?<!protein shake)|"
      r"(?:energy|sports?|electrolyte|hydration|protein|meal) drinks?|energy shots?|"
      r"drink mix(?:es)?|drink powders?|hot cocoa|cocoa mix(?:es)?|drinking (?:cocoa|chocolate)|"
      r"almond breeze|milk straws?|flavou?r(?:ed)? straws?|"
      r"(?:red|white|ros[eé]|sparkling|dessert|table|fruit|cooking) wines?|cabernet(?: sauvignon)?|"
      r"merlot|pinot (?:noir|grigio|gris|blanc)|chardonnay|sauvignon blanc|syrah|shiraz|zinfandel|"
      r"malbec|tempranillo|riesling|prosecco|champagne|cava|moscato|sangria|port wine|sherry|"
      r"brut(?: cuvee)?|cuvee|rioja|chianti|bordeaux|beaujolais|grenache|sangiovese|"
      r"hot chocolate|(?:chocolate|almond|oat|soy|coconut|rice|cashew|evaporated|condensed|"
      r"powdered|dry|goat|lactose free|whole|skim|2%|organic) milks?|milks?(?<!milk chocolate)|"
      r"colas?|root beers?|ginger (?:ale|beer)|sarsaparilla|cider(?! vinegar)|"
      r"concentrates?|(?:coffee|drink|soda|snow ?cone|shaved ice|cocktail|beverage|flavou?ring|"
      r"latte|frappe|torani|monin) syrups?|"
      r"cordials?|horchata|agua fresca|malt(?: drinks?| beverages?)?|"
      r"wines?(?! vinegar)|beers?(?<!root beer)(?<!ginger beer)|sake|spirits?|liqueurs?|"
      r"mixers?|bitters|cocktail (?:mix(?:es)?|mixers?|syrups?)|margarita mix)\b"),
    R("snacks",
      r"\b(?:chips?(?<!chocolate chips)(?<!chocolate chip)(?<!baking chips)(?<!butterscotch chips)"
      r"(?<!peanut butter chips)(?<!white chips)(?<!carob chips)(?<!semi-sweet chips)|crisps?|"
      r"crackers?|pretzels?|popcorn|puffs?|cheese puffs|jerky|(?:meat|beef|pork|turkey|salami) "
      r"(?:sticks?|snacks?|bars?|bites?)|biltong|pork rinds?|chicharrones?|rice cakes?|"
      r"(?:snack|granola|cereal|fruit|nut|oat|energy|breakfast|chewy|crispy|keto|paleo|"
      r"date|seed|fig|trail mix|kind|oatmeal|nutrition|bakery|muesli|rice crispy|"
      r"cereal and milk) bars?|snacks?|trail mix(?:es)?|snack mix(?:es)?|party mix|"
      r"chex mix|nachos?|tortilla chips|veggie (?:straws?|chips?|sticks?)|seaweed snacks?|"
      r"fruit (?:snacks?|leathers?|rolls?|strips?|chips?|crisps?)|fruit ?snacks?|"
      r"corn nuts|cheese crisps|(?:corn|cheese|veggie|pita|bagel|kale|plantain|banana|apple|"
      r"coconut) (?:chips?|crisps?|sticks?)|sticks(?= *$)|bites(?= *$)|munchies|"
      r"crunch(?:ies)?(?= *$)|wasabi peas|snack packs?|lunchables)\b"),
    R("candy",
      r"\b(?:cand(?:y|ies)|chocolates?(?! (?:chips?|syrup|sauce|milk|powder|chip cookies?|"
      r"cookies?|cake|cakes|wafers?|biscuits?|cereal|protein|frosting|extract|drinks?|"
      r"covered (?:almonds?|nuts?|cashews?|pretzels?|raisins?|espresso|coffee beans?|"
      r"cherries|strawberries|fruit)))|"
      r"(?:chocolate|candy|toffee|caramel|nougat|wafer|peanut butter|crispy) bars?|"
      r"chocolate[- ]covered (?:almonds?|nuts?|cashews?|pretzels?|raisins?|espresso beans?|"
      r"coffee beans?|cherries|strawberries|fruit|peanuts?|blueberries|ginger|marshmallows?|"
      r"pecans?|macadamias?|hazelnuts?|oreos?)|truffles?(?! (?:oil|salt|sauce|pasta))|"
      r"bon ?bons?|pralines?|gumm(?:y|ies)|gummy bears?|gummi(?:es)?|jelly beans?|jellies(?= *$)|"
      r"(?:chewing |bubble )?gum(?! (?:arabic|base|paste|tragacanth))|mints?(?<!fresh mint)"
      r"(?<!dried mint)|breath mints?|lollipops?|lolli ?pops?|suckers?|(?<!cake )pops(?= *$)|"
      r"licorice|liquorice|toffees?|caramels?(?! (?:sauce|syrup|color|colour|extract|"
      r"flavou?r))|fudge(?! sauce)|nougats?|marshmallows?|taffy|brittle|"
      r"marzipan|halva|halwa|turkish delight|lokum|mochi|"
      r"(?:hard|soft|sour|chewy|cotton|rock|gummy|novelty|retro|bulk|christmas|easter|halloween|"
      r"valentine'?s?|peppermint|mexican|japanese|korean) cand(?:y|ies)|candy canes?|"
      r"peanut butter cups?|peppermint patt(?:y|ies)|kisses|m&m(?='?s\b)|m m(?='s)|skittles|"
      r"snickers|kit ?kat|reese(?='?s\b)|twix|hershey(?='?s?(?= *$|,| -| \())|cherry cordials?|"
      r"sweets|confections?|confectionery|dragees?|pastilles?|bonbons?|chews(?= *$|,| -| \())\b"),
    R("bakery",
      r"\b(?:cookies?(?! (?:cutters?|dough mix|mix(?:es)?|butter|stamps?|press(?:es)?|sheets?|"
      r"decorating))|biscuits?(?! mix)|biscotti|wafers?|shortbread|macarons?|macaroons?|"
      r"cakes?(?<!rice cake)(?<!rice cakes)(?! (?:mix(?:es)?|toppers?|decorations?|stands?|"
      r"pans?|boards?|picks?|sprinkles|flour|tins?))|cupcakes?(?! (?:toppers?|picks?|liners?|"
      r"wrappers?))|brownies?(?! mix)|muffins?(?! mix)|pastr(?:y|ies)|croissants?|danish|"
      r"pies?(?! filling| crust| crusts)|tarts?|donuts?|doughnuts?|"
      r"breads?(?! (?:mix|crumbs|flour|machine))|loaf|loaves|bagels?|buns?|"
      r"(?<!fruit )(?<!egg )(?<!spring )rolls?"
      r"|tortillas?|pita(?: breads?)?|naan|flatbreads?|crumpets?|"
      r"english muffins?|baguettes?|stollen|panettone|panforte|fruit ?cakes?|pound cakes?|"
      r"cheesecakes?|scones?|strudels?|baklava|churros?|stroopwafels?|madeleines?|"
      r"lady ?fingers?|pizzelles?|rugelach|babka|kolaches?|empanadas?|pretzel (?:rolls?|buns?)|"
      r"oreos?|graham crackers?|animal crackers?|ice cream (?:cones?|sandwich(?:es)?)|"
      r"cones?(?= *$))\b"),
    R("baking",
      r"\b(?:flours?|(?:cake|brownie|cookie|muffin|bread|baking|cornbread|biscuit|pizza|"
      r"cupcake|pie crust|frosting|icing|scone|donut|keto|gluten free) (?:mix(?:es)?|dough)|"
      r"baking (?:soda|powder|chips?|bars?|cocoa|chocolate|mix(?:es)?|supplies|essentials|"
      r"decorations?|spray)|yeast|(?:vanilla|almond|lemon|peppermint|orange|rum|maple|coconut|"
      r"banana|cherry|strawberry|coffee|anise|butter|rose|pandan) (?:extracts?|flavou?rings?|"
      r"essences?|emulsions?|(?<!coffee )beans?|paste)|extracts?(?<!coffee extract)|"
      r"vanilla(?= *$|,| -| \()|frostings?|icings?|fondant|sprinkles|nonpareils|jimmies|"
      r"sanding sugar|edible (?:glitter|dust|paint|ink|pearls?|decorations?|images?|"
      r"sheets?|flowers?|gold|silver|toppers?)|(?:food|gel|icing|candy) colou?r(?:ing|s)?|"
      r"luster dust|petal dust|(?:chocolate|carob|butterscotch|peanut butter|white|semi-?sweet|"
      r"bittersweet|mint|toffee) chips|chocolate chips?|cocoa (?:powder|nibs|butter)|cacao "
      r"(?:powder|nibs|butter|paste)|(?:brown|powdered|confectioners'?|cane|coconut|date|"
      r"granulated|raw|turbinado|demerara|muscovado|maple|caster|pearl|rock|palm|monk fruit|"
      r"organic) sugars?|sugars?(?<!no sugar)|sweeteners?|stevia|monk ?fruit|erythritol|xylitol|"
      r"allulose|agave(?: nectar| syrup)?|molasses|corn ?syrup|(?:pie|tart) (?:fillings?|"
      r"crusts?|shells?)|pie fillings?|gelatin|agar|pectin|cornstarch|corn starch|tapioca "
      r"(?:starch|flour)|arrowroot|xanthan gum|cream of tartar|meringue powder|almond (?:flour|"
      r"meal)|breadcrumbs?|bread crumbs|panko|graham cracker crumbs?|sourdough starter|"
      r"marzipan|almond paste|condensed milk|evaporated milk|shortening|"
      r"(?:baking|parchment) (?:paper|cups?)|cupcake liners?|baking mats?)\b"),
    R("spices",
      r"\b(?:spices?|seasonings?|(?:spice|seasoning|herb) (?:blends?|mix(?:es)?|rubs?|sets?|kits?)|"
      r"rubs?(?= *$|,| -| \()|(?:bbq|barbecue|dry|steak|meat|chicken|pork|rib|brisket|"
      r"coffee) rubs?|herbs?(?<!herbal)|(?:sea|kosher|himalayan|pink|smoked|celtic|flake|"
      r"black|garlic|onion|celery|truffle|table|rock|iodized|finishing|flaky|grey|gray|"
      r"pickling|seasoned|hawaiian|lava) salts?|salts?(?<!epsom salt)(?<!bath salt)|"
      r"peppercorns?|(?:black|white|cayenne|red|lemon|chili|chilli|ground|crushed red) peppers?"
      r"|pepper(?: flakes)?(?= *$|,| -| \()|cinnamon|nutmeg|cloves?(?! garlic)|"
      r"cumin|turmeric|paprika|oregano|basil(?! seeds?)|thyme|rosemary|sage|parsley|dill|"
      r"cilantro|coriander|cardamom|saffron|bay leaf|bay leaves|allspice|ginger powder|"
      r"garlic powder|onion powder|chili powder|chilli powder|curry(?! (?:sauce|paste|ketchup))|"
      r"curry powder|masala|garam masala|za'?atar|sumac|furikake|togarashi|shichimi|"
      r"five spice|old bay|adobo(?! sauce)|sazon|fajita seasoning|taco seasoning|everything bagel|"
      r"msg|bouillon(?: cubes?| powder)?|stock cubes?|(?:vanilla )?bean pods?|fenugreek|"
      r"asafoetida|hing|ajwain|mustard seeds?|fennel seeds?|celery seeds?|chili flakes|"
      r"red pepper flakes|herbes de provence|italian seasoning|poultry seasoning)\b"),
    R("sauces",
      r"\b(?:sauces?|salsas?|ketchup|catsup|mustards?(?! seeds?)|mayo(?:nnaise)?|aioli|"
      r"dressings?(?! (?:mix))|vinaigrettes?|marinades?|dips?(?<!skinny dip)|hummus|"
      r"guacamole|queso|pesto|chutneys?|relish(?:es)?|hot sauces?|sriracha|tabasco|"
      r"soy sauce|tamari|teriyaki|fish sauce|oyster sauce|hoisin|gochujang|sambal|harissa|"
      r"chimichurri|tahini|tzatziki|horseradish|wasabi(?! peas)|worcestershire|"
      r"vinegars?|(?:olive|avocado|coconut|sesame|canola|vegetable|sunflower|peanut|"
      r"grapeseed|walnut|truffle|chili|chilli|cooking|frying|rice bran|mct|flaxseed|"
      r"safflower|corn|almond|pumpkin seed|hemp seed|mustard|palm) oils?|oils?(?<!essential oil)|"
      r"ghee|cooking sprays?|(?:curry|chili|chilli|tomato|garlic|ginger|miso|tamarind|"
      r"red curry|green curry|harissa|shrimp|anchovy|bean|pepper) pastes?|miso|"
      r"(?:pizza|pasta|marinara|alfredo|bbq|barbecue|enchilada|buffalo|wing|steak|"
      r"cocktail|tartar|cheese|chocolate|caramel|fudge|dessert) sauces?|gravy(?! mix)|"
      r"glazes?|condiments?|syrups?(?<=chocolate syrup)|bbq(?= *$))\b"),
    R("pantry",
      r"\b(?:pastas?|spaghetti|penne|macaroni|fusilli|rigatoni|linguine|fettuccine|lasagna|"
      r"orzo|gnocchi|ravioli|tortellini|noodles?|ramen|udon|soba|rice noodles|vermicelli|"
      r"lo mein|pad thai|rice(?! (?:cakes?|crackers?|vinegar|paper|wine|krispies|bran|milk|"
      r"syrup|flour|noodles?|cooker|chips))|(?:basmati|jasmine|brown|white|wild|sushi|arborio|"
      r"black|red|sticky|glutinous|parboiled|instant|minute|forbidden|calrose) rice|"
      r"quinoa|couscous|bulgur|farro|barley|millet|amaranth|buckwheat|teff|freekeh|polenta|"
      r"grits|cornmeal|semolina|lentils?|dals?(?= *$|,| -| \()|chickpeas?|garbanzo(?: beans?)?|"
      r"(?:black|pinto|kidney|navy|lima|mung|adzuki|azuki|cannellini|great northern|fava|"
      r"split|dried|dry|refried|baked|white|red|green|cranberry|mayocoba|heirloom) (?:beans?|"
      r"peas|lentils)|beans(?= *$|,| -| \()|split peas|grains?(?= *$|,| -| \()|"
      r"mac(?:aroni)? (?:and|&|n) cheese|mac ?n ?cheese|instant noodles?|cup noodles?|"
      r"stuffing(?: mix)?|mashed potatoes|instant potatoes|potato flakes|tofu|tempeh|seitan|"
      r"textured vegetable protein|tvp)\b"),
    R("breakfast",
      r"\b(?:cereals?|oatmeal|oats|porridge|granola(?! bars?)|muesli|cream of wheat|"
      r"(?:pancake|waffle|crepe|french toast) (?:mix(?:es)?|syrups?)|pancakes?|waffles?|"
      r"(?:maple|pancake|waffle|breakfast|table|blueberry) syrups?|syrup(?= *$)|honey(?:comb)?|"
      r"(?:peanut|almond|cashew|sunflower|hazelnut|cookie|seed|pecan|walnut|pistachio|nut|"
      r"macadamia|coconut|apple|pumpkin) butters?|nut butters?|jams?|jell(?:y|ies)(?! beans?)|"
      r"preserves|marmalades?|fruit spreads?|spreads?|nutella|lemon curd|curds?|"
      r"breakfast (?:bars?|biscuits?|cookies?|sandwich(?:es)?|essentials)|pop[- ]?tarts?|"
      r"toaster pastr(?:y|ies)|instant oatmeal|overnight oats|hash ?browns?)\b"),
    R("canned",
      r"\b(?:soups?|broths?|(?<!bully )stocks?(?= *$|,| -| \()|bone broth|(?:chicken|beef|"
      r"vegetable|veggie|bone|fish|mushroom|seafood|dashi) (?:broth|stock|base|bouillon)|"
      r"chili(?: con carne| beans)?(?= *$|,| -| \()|stews?|chowders?|bisques?|"
      r"canned (?:(?:\w+ )?(?:fruits?|vegetables?|veggies|fish|meats?|beans?|tuna|chicken|salmon|"
      r"corn|"
      r"peas|tomatoes|soups?|chili|pumpkin|peaches|pineapples?|pears|mushrooms|carrots|potatoes|"
      r"beef|ham|crab|clams|oysters|sardines|goods?|milk|cheese|bread|butter|water))|(?:diced|"
      r"crushed|whole|stewed|sun-?dried|peeled|plum|cherry|san marzano) "
      r"tomatoes|tomato (?:paste|puree|sauce)(?<=puree)|(?:tuna|salmon|sardines?|anchovies|"
      r"mackerel|crab ?meat|clams|oysters|kippers|herring|spam|corned beef|vienna sausages?)"
      r"(?= *$|,| -| \(| in | fillets?| chunks?| steaks?| packets?| pouch)|"
      r"pickles?|pickled (?:vegetables?|onions?|peppers?|jalapenos?|beets?|ginger|radish(?:es)?|"
      r"eggs?|"
      r"herring|garlic|okra|cucumbers?|mango(?:es)?|lemons?|plums?|cabbage|carrots?|mustard greens|"
      r"green beans|asparagus|watermelon rind|tea leaves)|olives?(?! oil)|capers|artichoke hearts?|"
      r"roasted (?:red )?peppers|"
      r"sauerkraut|kimchi|mandarin oranges|fruit cocktail|"
      r"(?:canned|tinned|jarred) (?:fruits?|vegetables?|fish|meat|beans?)|ready meals?|"
      r"meals?(?= *$)|(?:mre|meals ready to eat)|curry(?= *$|,| -| \()(?<!curry powder)|"
      r"(?:instant|ready to eat|heat and eat|microwave) (?:meals?|curry|rice|dinners?)|"
      r"pouch meals|freeze[- ]dried meals?|dinner kits?|(?:hamburger|tuna) helper|"
      r"coconut (?:milk|cream)(?! powder)|pumpkin puree|baked beans|refried beans)\b"),
    R("nuts",
      r"\b(?:nuts?(?<!corn nuts)(?<!doughnuts)|almonds?|cashews?|pistachios?|peanuts?|walnuts?|"
      r"pecans?|macadamias?|hazelnuts?|filberts?|brazil nuts?|pine nuts?|chestnuts?|"
      r"(?:sunflower|pumpkin|chia|flax|hemp|sesame|watermelon|poppy|basil) seeds?|pepitas|"
      r"flaxseeds?|seeds(?= *$|,| -| \()|trail mix|mixed nuts|"
      r"dried (?:fruits?|mangos?|mangoes|apricots?|cranberr(?:y|ies)|cherr(?:y|ies)|"
      r"blueberr(?:y|ies)|figs?|dates|apples?|bananas?|pineapples?|papayas?|strawberr(?:y|ies)|"
      r"kiwis?|peaches|plums|prunes|raisins|goji berries|coconut|persimmons?|jackfruit|"
      r"durian|tomatoes|mushrooms)|(?:freeze[- ]dried) (?:fruits?|strawberr(?:y|ies)|blueberr(?:y|"
      r"ies)|raspberr(?:y|ies)|mangos?|"
      r"mangoes|apples?|bananas?|pineapples?|peaches|cherr(?:y|ies)|durian|jackfruit|kiwis?|figs?|"
      r"cranberr(?:y|ies)|corn|peas|vegetables?|dragon ?fruits?)|"
      r"raisins|sultanas|currants|prunes|dates(?= *$|,| -| \()|(?:medjool|deglet|ajwa|"
      r"mazafati) dates|figs(?= *$|,| -| \()|goji berries|mulberries|craisins|"
      r"coconut (?:flakes|chips|shreds)|shredded coconut|fruit and nut|nut mix|"
      r"(?:roasted|salted|raw|blanched|sliced|slivered|candied|honey roasted|smoked|"
      r"spiced|unsalted|shelled|in-shell|in shell) (?:almonds?|cashews?|peanuts?|pecans?|"
      r"walnuts?|pistachios?|nuts?|hazelnuts?|macadamias?|seeds))\b"),
    R("protein",
      r"\b(?:protein(?: (?:powders?|bars?|shakes?|drinks?|snacks?|cookies?|chips|puffs|bites|"
      r"crisps|balls|water|coffee|pancake mix|oatmeal|cereal|supplements?|blends?|isolate))?|"
      r"whey(?: protein)?|casein|collagen(?: peptides| powder)?|meal replacements?(?: shakes?|"
      r" bars?)?|"
      r"(?:pre|post)[- ]?workout|creatine|bcaas?|amino acids?|electrolytes?(?: powders?| drink mix|"
      r" "
      r"tablets| packets)?|supplements?|vitamins?|multivitamins?|probiotics?|prebiotics?|"
      r"(?:superfood|greens|fiber|fibre|mushroom|beet|moringa|spirulina|chlorella|maca|"
      r"ashwagandha|wheatgrass) (?:powders?|blends?|supplements?)|spirulina|chlorella|"
      r"(?<!coffee )(?<!nespresso )capsules?|tablets?|softgels?|gummies(?<=vitamin gummies)|"
      r"nutrition(?:al)? (?:shakes?|drinks?|bars?)|ensure|(?:baby|infant|toddler) formula|"
      r"formula(?= *$|,| -| \())\b"),
    R("fresh",
      r"\b(?:cheeses?(?! (?:puffs?|crackers?|crisps?|sauce|dip|powder|straws?|curls?|balls))|"
      r"cheddar|parmesan|mozzarella|brie|gouda|feta|camembert|manchego|pecorino|gruyere|"
      r"provolone|ricotta|mascarpone|halloumi|paneer|"
      r"(?<!easter )(?<!chocolate )(?<!candy )eggs?(?! noodles)"
      r"|butter(?= *$|,| -| \()|(?:salted|unsalted|cultured|"
      r"grass-?fed|irish|european) butter|yogh?urts?|kefir|cream cheese|sour cream|"
      r"heavy cream|whipped cream|ice creams?|gelato|sorbet|frozen (?:yogurt|desserts?|meals?|"
      r"pizzas?|vegetables?|veggies|fruits?|berries|dinners?|entrees?|waffles|treats?|novelt(?:y|"
      r"ies))|"
      r"(?:beef|pork|lamb|chicken|turkey|duck|veal|venison|bison|goat|wagyu|angus|ground beef|"
      r"steak|brisket|ribs?|bacon|ham|sausages?|salami|pepperoni|prosciutto|chorizo|hot dogs?|"
      r"franks?|bratwurst|kielbasa|pastrami|meatballs?|deli meats?|charcuterie|pate|foie gras)"
      r"(?= *$|,| -| \(| (?:\d|chub|links?|patties|fillets?|loins?|breasts?|thighs?|wings?|"
      r"legs?|roasts?|chops?|cuts?|tenderloins?|slices?|sliced|log|frozen|ground|whole|"
      r"halves|pack|box|portions?|and|ring|rings))|steaks?|scrapple|blood sausages?|"
      r"(?:\w+ )?salads?(?! (?:dressings?|kits?|toppers?|toppings?|spinners?|bowls?|tongs))|"
      r"(?:fish|salmon|cod|tilapia|snapper|halibut|haddock|pollock|swordfish) fillets?|"
      r"(?:salmon|shrimp|prawns?|lobsters?|crabs?|scallops?|cod|halibut|tilapia|trout|"
      r"oysters?|mussels?|clams?|caviar|roe|octopus|squid|calamari|catfish|mahi mahi|"
      r"swordfish|sea bass|seafood)(?= *$|,| -| \(| (?:\d|fillets?|portions?|tails?|meat|"
      r"frozen|wild|fresh|smoked|whole|legs?|claws?|steaks?|box|pack))|"
      r"(?:fresh|organic) (?:fruits?|vegetables?|produce|herbs)|fruit (?:baskets?|boxes?)|"
      r"meat(?= *$)|tofu(?= *$))\b"),
    R("gifts",
      r"\b(?:gift (?:baskets?|box(?:es)?|sets?|packs?|bags?|tins?|hampers?|crates?|bundles?|"
      r"assortments?)|(?:snack|candy|chocolate|cookie|coffee|tea|fruit|cheese|meat|nut|"
      r"gourmet|holiday|christmas|birthday|sympathy|care|valentine'?s?|easter|halloween|"
      r"thank you|get well|corporate) (?:gift )?(?:baskets?|box(?:es)?|hampers?|packages?|"
      r"crates?|towers?|trays?|samplers?)|care packages?|hampers?|gift ?baskets?|"
      r"advent calendars?|stocking stuffers?|party favors?|favors?(?= *$))\b"),
    R("cake_decor",
      r"\b(?:(?:cake|cupcake) (?:toppers?|picks?|decorations?|decorating (?:kits?|supplies)|"
      r"stands?|wrappers?|liners?|boards?|flags?|banners?)|toppers?(?<!salad topper)|"
      r"cupcake (?:liners?|wrappers?|cups?)|party (?:supplies|decorations?|favors?|packs?|"
      r"hats?|plates?|cups?|napkins?|balloons?)|balloons?|(?:birthday|party|wedding|baby "
      r"shower|bridal shower|graduation) (?:decorations?|supplies|banners?|signs?)|"
      r"(?<!edible )decorations?|banners?|candles?|confetti|"
      r"(?:paper|plastic) (?:plates?|cups?|straws?|napkins?)|napkins?|straws?|"
      r"piñatas?|pinatas?|picks(?= *$))\b"),
    R("non_food",
      r"\b(?:(?:dog|cat|pet|puppy|kitten|bird|horse|fish) (?:treats?|food|chews?|biscuits?|"
      r"snacks?)|bully sticks?|rawhides?|dental chews?|(?:for|to) (?:dogs|cats|pets)\b|"
      r"mugs?|tumblers?|(?:water|wine|travel) bottles?(?! of)|kettles?|teapots?|tea ?pots?|"
      r"(?:tea )?infusers?|strainers?|(?:coffee|espresso|pour[- ]over) (?:makers?|machines?|"
      r"grinders?|filters?|press(?:es)?|scales?)|french press(?:es)?|grinders?|filters?|"
      r"thermos|cutting boards?|knives?|(?:measuring|mixing) (?:cups?|spoons?|bowls?)|"
      r"spatulas?|whisks?|(?:baking|cake|muffin|loaf|bundt|sheet) (?:pans?|tins?|molds?)|"
      r"molds?|moulds?|cookie cutters?|cookie press(?:es)?|piping (?:bags?|tips?)|containers?|"
      r"jars?(?= *$|,| -| \()|"
      r"canisters?|storage (?:bags?|containers?)|lunch box(?:es)?|ice cube trays?|"
      r"soaps?|lotions?|shampoos?|candles?|essential oils?|incense|seeds for planting|"
      r"(?:garden|vegetable|herb|flower) seeds|plants?|seedlings?|(?<!garlic )bulbs?|"
      r"t-?shirts?|aprons?|socks|mugs?|ornaments?|keychains?|magnets?|stickers?|books?|"
      r"cookbooks?|recipe cards?|greeting cards?|gift cards?|toys?)\b"),
    R("fresh",
      r"\b(?:"
      r"(?:fresh|organic|conventional)? ?(?:apples?|bananas?|oranges?|lemons?|limes?|"
      r"grapefruits?|grapes?|strawberr(?:y|ies)|blueberr(?:y|ies)|raspberr(?:y|ies)|"
      r"blackberr(?:y|ies)|cherr(?:y|ies)|peaches|pears?|plums?|mangos?|mangoes|pineapples?|"
      r"melons?|watermelons?|cantaloupes?|kiwis?|avocados?|papayas?|pomegranates?|"
      r"coconuts?|figs|apricots?|nectarines?|persimmons?|lychees?|dragon ?fruits?|"
      r"passion ?fruits?|guavas?|tomatoes|potatoes|sweet potatoes|onions?|garlic|shallots?|"
      r"carrots?|celery|broccoli|cauliflower|cabbages?|lettuce|spinach|kale|cucumbers?|"
      r"zucchini|squash|pumpkins?|peppers?|jalapenos?|mushrooms?|asparagus|beets?|radish(?:es)?|"
      r"turnips?|corn|green beans|peas|herbs|microgreens|sprouts|yams?|ginger root|"
      r"ginger(?= *$)|cassava|taro|plantains?|okra|eggplants?|artichokes?|leeks?|"
      r"bok choy|chard|arugula|fennel|horseradish root|tomatillos?|chilies|chiles)"
      r"(?= *$|,| -| \(| (?:\d|lb|lbs|pound|each|bag|box|bunch|count|ct|organic|fresh|"
      r"conventional|whole|large|small|medium|pack))|produce|frozen [\w ]{0,12}?(?:peach|peaches|"
      r"berries|fruit|vegetables))\b", weak=True),
    R("snacks",
      r"\bbars?\b", weak=True),
    R("beverages",
      r"\b(?:waters?|syrups?)\b", weak=True),
    R("baking",
      r"\b(?:cocoa(?: powder)?|cocoa (?:liquor|mass|block|paste)|chocolate liquor|"
                r"cacao (?:liquor|mass)|corn starch|(?:potato|tapioca|cassava|kuzu|arrowroot|corn|"
                r"rice|wheat|pea) starch|starch(?:es)?|rose water|orange blossom water|"
                r"(?:corn|karo|simple|agave|rice|date|golden|cane) syrups?)\b"),
    R("gifts",
      r"\b(?:samplers?|sampler packs?|variety (?:packs?|box(?:es)?|bundles?|assortments?)|"
               r"assortments?|assorted)\b", weak=True),
    R("cake_decor",
      r"\bstraws?\b", weak=True),
    R("canned",
      r"\b(?:fruit cups?|tidbits|pur[eé]e(?:s)?|(?:taco|enchilada|pie) fillings?|"
                r"pepperoncinis?|peperoncini|banana blossoms?|jackfruit in brine)\b"),
    R("bakery",
      r"\b(?:croutons?|(?:tostada|taco) shells?|garlic knots?|stuffing)\b"),
    R("breakfast",
      r"\b(?:flap ?jacks?(?: mix)?|flapjack mix)\b"),
    R("pantry",
      r"\b(?:soy ?beans?|soybeans?)\b"),
    R("non_food",
      r"\b(?:nutcrackers?|nut crackers?|(?:dog|bark|shock|training|pet) collars?|"
                  r"leash(?:es)?|nesting dolls?|matryoshka)\b"),
    R("protein",
      r"\b(?:vitamin [a-z0-9]+ oil|fish oil|cod liver oil|krill oil|omega-?3s?|"
                 r"boost(?= (?:breeze|original|high protein|plus|nutritional|glucose))|"
                 r"ensure (?:plus|"
                 r"original|max|high protein))\b"),
]

# A TARGET connector introduces the device the product is for; "fit" only before a
# brand or model word ("Case Fit All-New Fire HD 10"; a bare "Secure Fit Sport
# Earphones" is no connector). BUNDLE connectors ("with") end the head but keep a device
# read there ("Galaxy Tab A ... with 64GB Memory Card" stays a tablet).
_E_CUT = {
    "for": {"": _T}, "compatible": {"": _T, "with": _T, "for": _T}, "compatibility": {"": _T},
    "fits": {"": _T},
    "fit": dict.fromkeys(
        "for all all-new new most the your model models apple samsung ipad iphone galaxy "
        "macbook kindle amazon airpods google pixel asus lenovo microsoft surface nintendo sony "
        "dell hp acer huawei lg motorola fitbit garmin gopro canon nikon".split(), _T),
    "designed": {"for": _T, "to": _T},
    "works": {"with": _T, "for": _T}, "work": {"with": _T}, "replacement": {"for": _T, "of": _T},
    "suitable": {"for": _T, "with": _T}, "made": {"for": _T}, "perfect": {"for": _T},
    "ideal": {"for": _T},
    "with": {"": _B}, "w/": {"": _B}, "includes": {"": _B}, "include": {"": _B},
    "including": {"": _B}, "and": {"": _N},
}
# Words of a bare model name: a head made only of these and numbers ("iPhone 7 Plus,
# iPhone 8 Plus, ... Cover") defers to a later accessory noun
_MODEL_TOKENS = frozenset((
    "apple iphone ipad ipod airpods macbook imac kindle fire hd hdx kids paperwhite oasis galaxy "
    "samsung note tab s a z flip fold pixel google xl moto motorola lg nokia oneplus huawei "
    "xiaomi redmi surface xbox one x playstation ps ps4 ps5 nintendo switch lite oled "
    "roku express ultra stick 4k chromecast gopro hero fitbit charge versa sense inspire "
    "garmin forerunner vivoactive fenix active classic gear pro plus max mini air se lite "
    "e edition gen generation th nd rd st inch"
).split())
ELECTRONICS_TAXONOMY = Taxonomy(ELECTRONICS, _ELECTRONICS_TABLE, ELEC_RULES, _E_CUT,
                                model_tokens=_MODEL_TOKENS)

# Grocery also cuts at "in", "by", "from", "plus", "made with" ...: what follows is a
# flavour or an ingredient ("Cold Brew Coffee ... in Oat Milk")
_G_CUT = {
    "for": {"": _T},
    "with": {"": _N}, "w/": {"": _N}, "in": {"": _N}, "includes": {"": _N},
    "including": {"": _N}, "plus": {"": _N}, "by": {"": _N}, "from": {"": _N},
    "made": {"with": _N, "from": _N}, "topped": {"with": _N}, "filled": {"with": _N},
    "covered": {"in": _N}, "dipped": {"in": _N},
}
GROCERY_TAXONOMY = Taxonomy(GROCERY, _GROCERY_TABLE, GROC_RULES, _G_CUT, weak_vs_any=True)

# The published vocabularies (the rail's `category` param and its labels; contract 3.1)
ELECTRONICS_CATEGORIES = {k: label for k, (label, _role) in _ELECTRONICS_TABLE.items()}
GROCERY_CATEGORIES = {k: label for k, (label, _role) in _GROCERY_TABLE.items()}
CATEGORY_LABELS = {**ELECTRONICS_CATEGORIES, **GROCERY_CATEGORIES}  # the keys are disjoint
ELECTRONICS_ROLES = ("device", "accessory", "part", "consumable", "service")
GROCERY_ROLES = ("food", "non_food")
ROLE_LABELS = {"device": "Devices", "accessory": "Accessories", "part": "Parts & components",
               "consumable": "Batteries & supplies", "service": "Protection plans",
               "food": "Food", "non_food": "Non-food"}
# The rail merges the fine keys into shop groups: group key -> (label, member keys).
# Service (protection plans) is in no group: a filter never offers it.
ELECTRONICS_RAIL_GROUPS = {
    "headphones": ("Headphones & earbuds", ("headphones",)),
    "audio": ("Speakers & audio", ("audio",)),
    "phones": ("Phones", ("phones",)),
    "computers_monitors": ("Computers & monitors", ("computers", "monitors")),
    "tablets": ("Tablets & e-readers", ("tablets",)),
    "tv_video": ("TV, projectors & streaming", ("tv_video",)),
    "cameras": ("Cameras", ("cameras",)),
    "watches": ("Watches & fitness trackers", ("watches",)),
    "smart_home_networking": ("Smart home & networking", ("smart_home", "networking")),
    "car": ("Car electronics & GPS", ("car",)),
    "gaming": ("Gaming", ("gaming",)),
    "other_devices": ("Other devices", ("optics", "lighting", "office")),
    "cases_protection": ("Cases & screen protection", ("cases", "screen_protectors")),
    "power_batteries": ("Chargers, power & batteries", ("power", "batteries")),
    "cables_remotes": ("Cables, adapters & remotes", ("cables", "av_accessories")),
    "keyboards_components": ("Keyboards, mice & components",
                             ("computer_accessories", "components")),
    "storage": ("Storage & memory", ("storage",)),
    "camera_accessories": ("Camera accessories", ("camera_accessories", "backdrops")),
    "mounts_bags": ("Mounts, stands & bags", ("mounts", "bags")),
    "watch_accessories": ("Watch bands & accessories", ("watch_accessories",)),
    "supplies_tools": ("Ink, supplies & tools", ("supplies", "tools")),
}
# Every Grocery key is its own group (18 entries; non_food included)
GROCERY_RAIL_GROUPS = {k: (label, (k,)) for k, label in GROCERY_CATEGORIES.items()}
# fine Electronics key -> its rail group key (service: none), for a filter on item_category
ELECTRONICS_GROUP_OF = {key: group for group, (_label, keys) in ELECTRONICS_RAIL_GROUPS.items()
                        for key in keys}


def electronics_category(title) -> Optional[str]:
    """item_category of one Electronics title (watch_kind decides "watches"), or None."""
    return _electronics_read(_norm(title))[0]


def grocery_category(title) -> Optional[str]:
    """item_category of one Grocery title, or None (annotate_catalog writes "unknown")."""
    return GROCERY_TAXONOMY.classify_text(_norm(title))[0]


# ====================================================================== DIETARY
# The enrichment's `dietary` column (scripts/enrich_catalog.py DIETARY_RULES, Grocery
# only, comma-separated) as a tuple of these keys. Its tags are positive regex matches
# over title + specs + description, with no negation handling ("processed in a
# non-Kosher facility"): title-sourced 90% precise (n=40), description-only 84% (n=50),
# hence the rail's hint "As the listing words it, not a certification". Its "sugar free"
# also covers "no added sugar" and "unsweetened", hence that label.
DIETARY_VALUES = ("gluten free", "organic", "kosher", "non-gmo", "vegan", "sugar free",
                  "dairy free", "keto", "vegetarian", "caffeine free", "nut free", "low carb",
                  "high protein", "low sodium", "halal")
DIETARY_LABELS = {
    "gluten free": "Gluten free", "organic": "Organic", "kosher": "Kosher",
    "non-gmo": "Non-GMO", "vegan": "Vegan", "sugar free": "Sugar free / no added sugar",
    "dairy free": "Dairy free", "keto": "Keto", "vegetarian": "Vegetarian",
    "caffeine free": "Caffeine free", "nut free": "Nut free", "low carb": "Low carb",
    "high protein": "High protein", "low sodium": "Low sodium", "halal": "Halal",
}
_DIET_SPELLINGS = {v.replace("-", " "): v for v in DIETARY_VALUES}


def diet_tags(value) -> Tuple[str, ...]:
    """The `dietary` column split into DIETARY_VALUES keys, in DIETARY_VALUES order, each
    once; () when missing. Hyphens and case are folded ("Gluten-Free" -> "gluten free");
    an unknown word is dropped (a fixed vocabulary, never seller text)."""
    if not isinstance(value, str) or not value.strip():
        return ()
    found = set()
    for part in value.split(","):
        key = " ".join(part.casefold().replace("-", " ").split())
        if key in _DIET_SPELLINGS:
            found.add(_DIET_SPELLINGS[key])
    return tuple(v for v in DIETARY_VALUES if v in found)


# ====================================================================== WATCH split
# On every vertical. A title is watch-RELATED when it names a watch, a smartwatch or a
# wearable tracker; `watch_kind` then says what the product is:
#   watch        -- a wristwatch / pocket watch / smartwatch / fitness tracker (the device)
#   band         -- band, strap, bracelet, loop, link (to wear it on)
#   storage      -- box, case, roll, stand, holder, display, organizer, winder, pillow
#   care_tool    -- repair kits, tools, spring bars, link removers, case openers,
#                   polishing cloths, cleaners
#   protection   -- screen protector, protective case / bumper / cover for a smartwatch
#   charger      -- charger, charging cable / dock / stand
#   part         -- crystal, bezel, battery, movement, crown, face, glass, clasp, buckle
#   other        -- watch-related but none of the above (watch chain, necklace charm, ...)
#   None         -- not about a watch ("bird watching", "watch cap", "watch movies")
# Same segment / right-most / target reading as the categories. Held-out 100 rows:
# watch-vs-accessory 99%, kind 96%. Known false positive: steampunk jewellery made of
# movements ("... Watch Movements Ring Mechanical Parts" reads watch).
WATCH_KINDS = ("watch", "band", "storage", "care_tool", "protection", "charger", "part", "other")
# A title is read only when it holds one of these (one C-level scan runs first)
_W_NEEDLES = ("watch", "fitbit", "tracker", "pedometer", "winder", "garmin", "amazfit",
              "timepiece", "chronograph")
_W_NEEDLE_RE = re.compile("|".join(_W_NEEDLES))
# Not about a watch: the verb ("watch live tv", "Outdoor Projector -- Watch Anywhere"), the
# tartan, the knit cap / hat ("Beanie Watch Caps", "Warm Watch Stocking Skull Knit Caps"),
# the games ("Yo-kai Watch", "Watch Dogs" masks), Fitbit's headphones and scale, and
# Grocery brands and promotions ("Sea Watch" clams, "Fire Watch" coffee, "Taste Watch Win"
# chips). The plurals and words added 2026-10-30 moved only those rows from "watch" to
# None (6 Fashion, 7 Grocery titles; nothing else in the catalog)
_W_NOT = re.compile(
    r"\b(?:bird ?watch(?:ing|er)|wildlife watching|watch(?:ing)? (?:tv|movies?|videos?|"
    r"shows?|netflix|youtube|football|the game|sports|live|television|anywhere)|to watch|"
    r"for watching|watch ?caps?|watch ?hats?|watch stocking|watch ?/ ?record|"
    r"watch, (?:forward|rewind|pause|record)|"
    r"black watch(?: tartan| plaid)?|watch ?dogs?|night ?watch|bay ?watch|watchmen|watch out|"
    r"neighborhood watch|watch tower|watchtower|watch party|weight watchers?|watch me|"
    r"watch (?:it|this|us|them|her|him)|kid watch tv|watch list|watch the|yo-?kai[- ]watch|"
    r"sea watch|fire watch|watch,? win|fitbit (?:flyer|aria))\b")
# The watch word is required: a "cord winder" or a "Side Winder" boot is not watch-related.
# Garmin's Approach G-series are handheld golf GPS units (only S- and X- are worn); an
# Amazfit word that is an accessory noun is no model ("Huami Amazfit Charger").
_W_DEVICE = re.compile(
    r"\b(?:smart ?watch(?:es)?|smartwatch(?:es)?|wrist ?watch(?:es)?|wristwatch(?:es)?|"
    r"pocket ?watch(?:es)?|timepieces?|chronographs?|fitness trackers?|activity trackers?|"
    r"pedometers?|watch(?:es)?|iwatch(?:es)?|fitbit(?: [a-z]+\s?\d*)?|"
    r"garmin (?:forerunner|vivoactive|vivomove|vivosmart|vivofit|fenix|venu|instinct|epix|"
    r"enduro|approach(?= [sx]\d)|lily)(?: \w+)?|"
    r"amazfit (?!(?:chargers?|charging|straps?|cases?|screen|protectors?|cables?|docks?|"
    r"films?)\b)\w+)\b")
# Feature words, not the product, once a bundle connector is crossed ("Sport MP3 Player
# with Pedometer/Stopwatch" is a player); before it they are the device ("Letsfit -
# Fitness Tracker with Heart Rate Monitor")
_W_FEATURE_DEVICES = ("pedometer", "fitness tracker", "activity tracker")
_W_ACC = tuple((kind, re.compile(pattern)) for kind, pattern in (
    ("band", r"\b(?:bands?|watch ?bands?|watchbands?|straps?|watch ?straps?|bracelets?(?= (?:strap|"
             r"band|replacement|compatible|for)|$)|loops?|sport loops?|solo loops?|wristbands?|"
             r"(?<=watch )links?|nato|zulu|deployment clasps?)\b"),
    # "box" is no storage next to "gift" ("... Watch with Gift Box"). "watch case(s)" with
    # the lookbehind first: after "cases" a "watch case" lookbehind never matched, so
    # "... Watch Cases ..." fell through to the watch itself (2026-10-30)
    ("storage", r"\b(?:(?<!gift )box(?:es)?(?! gift)|(?<=watch )cases?|(?:storage|display|"
                r"travel|carrying|organizer|jewelry|zipper|hard|leather|wooden|wood) cases?|rolls?|"
                r"stands?|holders?|displays?|organi[sz]ers?|trays?|pillows?|cushions?|winders?|"
                r"storage|pouch(?:es)?|cabinets?|drawers?|chests?)\b"),
    # "press" singular too ('presses?' read only "presse(s)"), and the wrench of a watch
    # tool kit ("Watch Tool Kit Plus Adjustable Watch Wrench" read as the watch itself).
    # The hole punch of a strap ("Leather Punch Watch Belt Hole Punch" read as a watch,
    # 2026-10-30 R2: the only catalog title it moves besides a "punch plier kit for ...
    # watch bands", which read as a band)
    ("care_tool", r"\b(?:tools?|tool kits?|repair kits?|kits?(?<=repair kit)|spring bars?|"
                  r"pin removers?|link removers?|removers?|case openers?|openers?|"
                  r"press(?:es)?|wrench(?:es)?|punch(?:es)?|"
                  r"screwdrivers?|tweezers|polishing cloths?|cloths?|cleaners?|cleaning "
                  r"(?:kits?|solutions?|cloths?)|loupes?|demagneti[sz]ers?|timegrapher|"
                  r"adjusters?|sizing (?:tools?|kits?))\b"),
    ("protection", r"\b(?:screen protectors?|protectors?|protective (?:cases?|covers?)|bumpers?|"
                   r"(?:tpu|pc|hard|soft|silicone|clear|glass|rugged|full) (?:cases?|covers?)|"
                   r"(?<!watch )cases?(?=.*\b(?:apple|iwatch|galaxy|fitbit|garmin|series|"
                   r"\d\dmm))|covers?|tempered glass|films?|shells?|skins?)\b"),
    ("charger", r"\b(?:chargers?|charging (?:cables?|cords?|docks?|stands?|stations?|cradles?|"
                r"pads?)|charging|docks?|cradles?|(?:usb|magnetic) cables?)\b"),
    # a "watch adapter" fits a watch to something else ("BestTechTool watch adapter
    # compatible with LEATHERMAN TREAD" read as a watch, 2026-10-30 R2); a charger adapter
    # is no watch adapter
    ("part", r"\b(?:crystals?|bezels?|batter(?:y|ies)|movements?(?! (?:sensor|tracker))|crowns?|"
             r"dials?(?= (?:replacement|parts?)|$)|faces?(?= replacement|$)|glass(?= replacement)|"
             r"clasps?|buckles?|hands?(?= replacement)|parts?|replacement parts?|"
             r"case backs?|gaskets?|stems?|lugs?|(?<=watch )adapt[eo]rs?)\b"),
    # a watch protection plan sells no watch; a conversion or quick release kit is no watch;
    # a slap bracelet, bangle or ring named after the watch is the bracelet ("Feather Slap
    # Bracelet Watch Patting Circle Slap Ring ... Slap Bangle" read as a watch, 2026-10-30
    # R2), while a "Slap Watch" names nothing after it and stays the watch
    ("other", r"\b(?:chains?|fobs?|necklaces?(?<!watch necklace)|charms?|"
              r"keychains?|lanyards?|brooch(?:es)?|cufflinks?|tie (?:clips?|bars?|pins?)|"
              r"stickers?|decals?|posters?|mugs?|t-?shirts?|shirts?|socks|costumes?|toys?|"
              r"cakes?|cookies?|candy|chocolates?|books?|calendars?|protection plans?|"
              r"conversion kits?|quick release kits?|slap (?:bracelets?|bangles?|rings?))\b"),
))
# Strong evidence the title sells the watch itself, so a band or box it names after the
# watch is the watch's own ("Analog Quartz Watch Leather Band")...
_W_SIGNAL = re.compile(
    r"\b(?:analog(?:ue)?|quartz|automatic|mechanical|self[- ]winding|hand[- ]wind(?:ing)?|"
    r"movement|chronograph|water[- ]?(?:resistant|proof)|\d+ ?(?:m|atm|bar)\b|dial|wristwatch|"
    r"men's watch|women's watch|mens watch|womens watch|dress watch|diver(?:'s)? watch|"
    r"sapphire|tourbillon|luminous|heart rate|blood (?:pressure|oxygen)|spo2|sleep (?:monitor|"
    r"tracker|tracking)|step counter|calls?|bluetooth|gps|ecg|touch ?screen|amoled|"
    r"notifications?|ip6[78])\b")
# ...unless the title also says the band or box is a replacement / fits another watch
_W_ACC_SIGNAL = re.compile(
    r"\b(?:replacement|compatible|compatibility|fits?|for (?:apple|samsung|fitbit|garmin|"
    r"galaxy|iwatch|amazfit|fossil|huawei)|quick release|\d{2} ?mm (?:band|strap|watch band)|"
    r"lug|spring bars?)\b")
_W_OWN_ACCESSORY = re.compile(
    r"\bwatch ?(?:bands?|straps?|box(?:es)?|winders?|cases?(?= for| holder| organizer| storage|$)|"
    r"rolls?|stands?|holders?|displays?|tools?|batter(?:y|ies)|crystals?|bezels?|parts?)\b")
# A "watch case" that the title says is for a smartwatch is its protective case, not a
# box to keep watches in: "HoneyGirl Watch Case for Fitbit Versa 3 ... Protective Bumper
# Shell", "Apple Watch Case ... Rugged Protective Case" and the TenCloud Garmin two-pack
# read storage (2026-10-30 R2: 3 Electronics and 2 Fashion titles; every storage "watch
# case" without one of these names -- "Watch Case for Men 6 Slots", a wooden watch case
# with a glass top -- stays storage). Not a bare size: "Watch Case 40mm" is the housing
_W_SMARTWATCH = re.compile(
    r"\b(?:apple watch|iwatch|galaxy watch|gear s\d|fitbit|garmin|amazfit|huawei watch|"
    r"ticwatch|smart ?watch(?:es)?)\b")


def _smartwatch_case(seg: str, start: int, t: str) -> bool:
    """Whether the storage word at seg[start:] is a "watch case" of a title naming a
    smartwatch (see _W_SMARTWATCH)."""
    return (seg.startswith("case", start) and seg[:start].endswith("watch ")
            and bool(_W_SMARTWATCH.search(t)))


# A band named right BEFORE a smartwatch maker's line name is the band for it, "for" left
# out: "Goton Leather Band Apple Watch Series 4 3 2 1, Retro Leather Watch Strap Compatible
# Apple Watch Band ..." read as the watch (the segment's right-most noun), so a Fashion band
# was a core watch on "apple watch band" (2026-10-30 live run). A maker lists its own watch
# under its name first ("Apple Watch Series 8 ... with Sport Band"), so a band word in front
# of the name is a third party's band; a generic name keeps the watch ("Leather Band
# Watch", "Mesh Band Smart Watch"), as do watch signals with no accessory signal (the
# _W_SIGNAL rule of a band after the watch). "Strap-for" glues the connector to the band
_W_BAND_BEFORE_NAMED = re.compile(
    r"(?:-for)? (?:samsung )?(?:apple watch|iwatch|galaxy watch|gear s\d|fitbit|garmin|amazfit|"
    r"huawei watch|ticwatch)\b")


def _band_before_named_watch(seg: str, accessories: list, t: str) -> bool:
    """Whether a band of `seg` (watch_kind_text's accessory tuples) directly precedes a
    named smartwatch (_W_BAND_BEFORE_NAMED) in a title without the watch's own signals."""
    if not any(kind == "band" and _W_BAND_BEFORE_NAMED.match(seg, end)
               for end, _neg_start, kind, _start in accessories):
        return False
    return not _W_SIGNAL.search(t) or bool(_W_ACC_SIGNAL.search(t))


def watch_kind_text(t: str) -> Optional[str]:
    """watch_kind of a NORMALISED title (_norm or product_type's _normalise)."""
    if not _W_NEEDLE_RE.search(t):
        return None
    t_clean = _W_NOT.sub(" ", t)
    if not _W_DEVICE.search(t_clean):
        return None
    bundled = after_target = named = False
    for seg, target in ELECTRONICS_TAXONOMY.iter_segments(t_clean):
        bundled = bundled or target is None
        after_target = after_target or bool(target)
        if not seg:
            continue
        devices = [m for m in _W_DEVICE.finditer(seg)
                   if not (bundled and m.group().startswith(_W_FEATURE_DEVICES))]
        named = named or bool(devices)
        accessories = [(m.end(), -m.start(), kind, m.start())
                       for kind, pattern in _W_ACC for m in pattern.finditer(seg)]
        if not devices and not accessories:
            continue
        if (target or after_target) and not accessories:
            # "band for apple watch": the watch after 'for' is the target, and so is one
            # in the list that runs on after it ("Cable ... for iPhone 13, iPad, Watch")
            continue
        last_device = max((m.end() for m in devices), default=-1)
        best = max(accessories) if accessories else None
        if best and (target or best[0] > last_device):
            kind = best[2]
            if (kind in ("band", "part", "storage", "protection") and devices
                    and _W_SIGNAL.search(t_clean) and not _W_ACC_SIGNAL.search(t_clean)
                    and not _W_OWN_ACCESSORY.search(seg)):
                return "watch"
            if kind == "storage" and _smartwatch_case(seg, best[3], t_clean):
                return "protection"
            return kind
        if best and _band_before_named_watch(seg, accessories, t_clean):
            return "band"
        if devices:
            return "watch"
    # watch-related, but no segment says what it sells; feature words alone are not
    return "other" if named else None


def watch_kind(title) -> Optional[str]:
    """What a watch-related title sells (WATCH_KINDS), None when it is not about a watch."""
    return watch_kind_text(_norm(title))


# What may follow a watch's name in a phrase that still IS the watch: a size, a series or
# model word ("apple watch series 7 45mm", "galaxy watch 4 classic", "fitbit charge 5").
# One reading per token: the first draft offered "45mm" and "45 mm" to two numeric
# alternatives, so a run of sizes before any other word backtracked exponentially (20
# sizes: ~1.5 s, 30: ~25 min). This one accepts exactly the same tails (4,882,812
# phrases compared, 2026-10-30) in linear time per start
_W_NAME_TAIL = re.compile(
    r"(?:\s+(?:series|se|ultra|pro|plus|classic|active\d*|lte|gps|\d+\.\d+(?: ?mm)?|"
    r"[\d/]+ ?mm|\d+)(?=\s|$))*\s*$")


def ends_on_a_watch(text: str) -> bool:
    """Whether a NORMALISED phrase names a watch LAST, so the phrase is the watch itself:
    "apple watch", "my fitbit charge 5", "galaxy watch 4 44mm" -- not "my watch-loving
    dad" or "watch lovers". product_type reads a jewellery ask FOR a watch with it
    ("bracelet for apple watch" asks for the band its titles type accessory), on the
    phrase after 'for' (at most five words); the search is quadratic in the phrase, so
    keep it to short phrases."""
    if watch_kind_text(text) != "watch":
        return False
    name = text[:_W_NAME_TAIL.search(text).start()]
    return any(m.end() == len(name) for m in _W_DEVICE.finditer(name))


# Electronics item_category and watch_kind are one source: a "watches" row is exactly a
# watch_kind "watch" row. The two readers disagreed on 104 of the 99,994 Electronics rows
# (97 read watch by watch_kind only); the 2026-10-27 port fixed the reader bugs that
# read showed (verbs, golf GPS units, a charger's model word, target lists, feature
# words) and lets watch_kind decide the rest -- except an Electronics device whose only
# watch words are feature words ("XIAOWU MP3 Player ... Video Play Pedometer Voice
# Recorder"): that row keeps its category and is no watch. A row the category rules
# called "watches" that the reader does not is filed by what the reader says it is (a
# heart-rate chest strap it does not read becomes "unknown").
_WATCH_KIND_CATEGORY = {"band": "watch_accessories", "storage": "watch_accessories",
                        "care_tool": "watch_accessories", "part": "watch_accessories",
                        "other": "watch_accessories", "charger": "power",
                        "protection": "cases"}


def _feature_words_only(t: str) -> bool:
    """Whether every watch word of a normalised title is a feature word (pedometer...)."""
    found = [m.group() for m in _W_DEVICE.finditer(_W_NOT.sub(" ", t))]
    return bool(found) and all(w.startswith(_W_FEATURE_DEVICES) for w in found)


# A box, roll or winder FOR watches is a watch accessory whatever container noun the
# listing adds after it: "Watch Box 12 Slot Display Case" read cases ("Cases, covers &
# skins", next to phone cases) and "Watch Storage Case for 10 Watches" cases too, the
# right-most noun winning (2026-10-30 R2; the reviewers' example, no Electronics catalog
# row today: the catalog's watch boxes are Fashion rows, which have no category). Read
# only on a watch_kind "storage" title whose category is a container, so an AirPods case
# holding a watch strap stays a case; a watch stand stays with the stands (mounts)
_W_STORAGE_NAMED = re.compile(
    r"\bwatch(?:es)? ?(?:box(?:es)?|winders?|rolls?|cases?|displays?|storage|organi[sz]ers?|"
    r"cabinets?|chests?|trays?|pillows?|cushions?)\b|\bfor (?:up to )?\d+ watch(?:es)?\b")
_CONTAINER_CATEGORIES = frozenset({"cases", "bags"})
# A cleaner is never the device it cleans. The categories read the right-most device noun
# of the deciding segment, so a cleaning product that names its device after the cleaning
# words read as that device (2026-10-30 R2: 17 catalog titles, every one read): "6 in 1
# Keyboard Cleaner Kit Earphone Cleaner Kit ..." and "Bluetooth Earbuds Cleaning Pen 3 in
# 1 Cleaner Kit Earphone Brush" headphones, "Charm &Magic Professional Camera Cleaning
# Set", "LituFoto K9 Camera Air Blowers" and "VSGO Camera Cleaning Gloves" cameras, "Vinyl
# Record Cleaner ... Brush Turntables" audio, two VCR head cleaners tv_video, and eight
# desk / keyboard vacuums and air dusters and a toy desk "dust cleaner" computers (via
# "desktop", "computer" or a "2-in-1"). When that segment (not a bundle's later one:
# "Canon ... Camera with ... Cleaning Kit" stays the camera) names one of these, the row
# is a cleaning tool. Only a DEVICE category moves (a "Lens Cleaning Pen" keeps its camera
# accessories). Never an ear cleaner (the otoscope cameras), a vacuum TUBE amplifier or a
# bare "brush" / "wipe" (a laptop case "& Dust Brush", a "wipe" in a camcorder's model
# line). Known residue: "... Cleaning Spray for Phones, Tablets and Laptops" (the device
# list after "for" runs past a comma, so "Tablets" decides in a later segment)
_CLEANER_PRODUCT = re.compile(
    r"\b(?:(?<!ear )cleaners?|(?<!ear )cleaning (?:kits?|sets?|pens?|brush(?:es)?|gloves?|"
    r"rollers?|cloths?|solutions?|sprays?|swabs?|wipes?|gel|putty|tools?)|"
    r"(?:computer|keyboard|desktop|desk|mini|usb|handheld|table|car) vacuums?(?! tubes?\b)|"
    r"(?:air|dust|lens) blowers?|(?:air )?dusters?|compressed air)\b")
_CLEANER_HINT = ("clean", "blower", "duster", "vacuum", "compressed air")


def _deciding_segment(t: str, how: str) -> str:
    """The segment of a normalised title that classify_text's `how` names ('h' the head,
    '3' the fourth, 'm2' / 'w3' a model or weak reading there)."""
    digits = how.lstrip("mw")
    wanted = int(digits) if digits.isdigit() else 0
    for i, (seg, _target) in enumerate(ELECTRONICS_TAXONOMY.iter_segments(t)):
        if i == wanted:
            return seg
    return ""


def _sells_a_cleaner(t: str, how: str) -> bool:
    """Whether the segment that decided a normalised title's category sells a cleaner."""
    return (any(h in t for h in _CLEANER_HINT)
            and bool(_CLEANER_PRODUCT.search(_deciding_segment(t, how))))


# A strap that names a camera and no watch is a camera strap. The watch accessories rule's
# bare "strap(s)" is the right-most noun of "Eorefo Camera Strap Leather Universal Shoulder
# Neck Belt Strap for ... DSLR Camera" and "NEEWER Faux Leather Hand Wrist Pad Strap for
# DSLR Camera", so camera straps sat on the watch rail, 3 of them on page 1 of "gift for a
# photographer" (2026-10-30 live run). Only where the watch reader finds no watch
# (watch_kind None) and the deciding segment holds the strap: a polishing cloth that lists
# cameras among the screens it wipes keeps its reading
_CAMERA_NAMED = re.compile(r"\b(?:cameras?|camcorders?|dslrs?|slrs?|mirrorless|gopros?)\b")
_STRAP_NOUN = re.compile(r"\b(?:straps?|bands?|wristbands?|loops?)\b")


def _camera_strap(t: str, how: str) -> bool:
    """Whether a normalised title filed with the watch accessories sells a camera strap."""
    return (bool(_CAMERA_NAMED.search(t))
            and bool(_STRAP_NOUN.search(_deciding_segment(t, how))))


# A case listed the way its tablet is: the head names the tablet and the product noun
# comes later, "Armorex Dragon Touch Alldaymall NeuTab Chromo Inc 7" Kids Tablet, Durable
# [Shockproof] [Impact Resistant] [Kids Safe] Rugged Silicone Cover (Hot Pink)" read as the
# tablet, a core one on "tablet for kids" (2026-10-30 live run: page 2, rerank 65). A
# tablets row whose later plain segments (after the deciding one, not after "for", never
# past a bundle "with") end on a case or cover is that case -- unless the title lists the
# device's own specs (storage, RAM, battery, OS, processor, a bundle, a renewed unit), as
# the real kids tablets that end on their "Kid-Proof Case" do ("HiGrace Kids Tablet 10
# inch, Android 12 ..., 32GB ROM, ..., Kid-Proof Case-Blue"), or names the Kids Edition's
# own "Kid-Proof Case". Tablets only: earbuds end on their charging case and camera
# bundles on "+ Case", with no such spec to tell them apart
_DEVICE_SPEC = re.compile(
    r"\b(?:\d+ ?(?:gb|tb)|ram|rom|\d+ ?mah|android|processor|cpu|(?:dual|quad|hexa|octa)[- ]?"
    r"core|\d+(?:\.\d+)? ?ghz|wi-?fi|bluetooth|bundle|renewed|refurbished|"
    r"kids?[- ]?proof case)\b")


def _ends_on_a_case(t: str, how: str) -> bool:
    """Whether the last reading of a normalised title's plain segments after the deciding
    one (`how`, as _deciding_segment reads it), before any bundle connector, is a case."""
    digits = how.lstrip("mw")
    decided = int(digits) if digits.isdigit() else 0
    tax, last = ELECTRONICS_TAXONOMY, None
    for i, (seg, target) in enumerate(tax.iter_segments(t)):
        if target is None:
            break
        if i <= decided or target or not seg:
            continue
        found = tax._found_fast(seg, target)
        if found:
            item = found[0] if len(found) == 1 else tax._decide(found)
            if item is not None:
                last = item[3].key
    return last == "cases"


def _electronics_read(t: str) -> Tuple[Optional[str], Optional[str]]:
    """(item_category or None, watch_kind) of a normalised Electronics title."""
    kind = watch_kind_text(t) if t else None
    key, how = ELECTRONICS_TAXONOMY.classify_text(t)
    if kind == "watch":
        if (key != "watches" and _ELECTRONICS_TABLE.get(key, ("", ""))[1] == "device"
                and _feature_words_only(t)):
            return key, None
        return "watches", kind
    if key == "watches":
        return _WATCH_KIND_CATEGORY.get(kind), kind
    if kind == "storage" and key in _CONTAINER_CATEGORIES and _W_STORAGE_NAMED.search(t):
        return "watch_accessories", kind
    if kind == "protection" and key == "watch_accessories":
        # a smartwatch's protective "watch case" (_smartwatch_case) is filed with the
        # protective cases, as _WATCH_KIND_CATEGORY files a "watches" row of that kind
        return _WATCH_KIND_CATEGORY[kind], kind
    if _ELECTRONICS_TABLE.get(key, ("", ""))[1] == "device" and _sells_a_cleaner(t, how):
        return "tools", kind
    if key == "watch_accessories" and kind is None and _camera_strap(t, how):
        return "camera_accessories", kind
    if key == "tablets" and not _DEVICE_SPEC.search(t) and _ends_on_a_case(t, how):
        return "cases", kind
    return key, kind


# ====================================================================== WORKS WITH
# The brand ecosystem an accessory, part or consumable NAMES ("Case for iPhone 12",
# "Charger for Galaxy Watch"): 39% of such Electronics rows. A device's own brand is its
# store / brand, so devices (and every other vertical) get (). Read from the title.
WORKS_WITH_LABELS = {
    "apple": "Apple", "samsung": "Samsung", "google": "Google", "amazon": "Amazon",
    "microsoft": "Microsoft", "playstation": "PlayStation", "nintendo": "Nintendo",
    "pc": "Dell, HP, Lenovo & PCs", "camera": "Canon, Nikon, Sony & GoPro",
    "fitness": "Fitbit & Garmin",
}
# The 2026-10-27 study's patterns, read token by token (a regex per ecosystem cost ~40 us
# a title): a whole token, a token prefix ("iphone12", "ipads"), or a phrase checked
# only when its first word is a token. A bare "Surface" counts only as a Microsoft
# Surface (a model word or a device list after it): half of the 466 Electronics titles
# saying it meant "surface mount", "any smooth surface" or "surface cleaner".
_WW_TOKENS = {
    "imac": "apple", "airpod": "apple", "airpods": "apple",
    "samsung": "samsung", "galaxy": "samsung",
    "pixel": "google", "chromecast": "google", "google": "google",
    "kindle": "amazon", "echo": "amazon", "alexa": "amazon",
    "xbox": "microsoft", "microsoft": "microsoft",
    "playstation": "playstation", "ps3": "playstation", "ps4": "playstation",
    "ps5": "playstation",
    "nintendo": "nintendo",
    "dell": "pc", "hp": "pc", "lenovo": "pc", "asus": "pc", "acer": "pc",
    "canon": "camera", "nikon": "camera", "gopro": "camera",
    "fitbit": "fitness", "garmin": "fitness",
}
_WW_PREFIXES = (("iphone", "apple"), ("ipad", "apple"), ("macbook", "apple"),
                ("iwatch", "apple"), ("ipod", "apple"), ("joycon", "nintendo"))
_WW_PHRASES = {   # first word -> (pattern, key)
    "apple": (re.compile(r"\bapple watch"), "apple"),
    "fire": (re.compile(r"\bfire (?:hd|tv|stick|\d)"), "amazon"),
    "switch": (re.compile(r"\bswitch (?:oled|lite)\b"), "nintendo"),
    "joy": (re.compile(r"\bjoy-?con"), "nintendo"),
    "sony": (re.compile(r"\bsony (?:a\d|alpha|nex|rx)"), "camera"),
    "surface": (re.compile(
        r"\bsurface (?:pro|go|book|laptop|duo|studio|pen|dock|headphones|earbuds|arc|precision|"
        r"type cover|keyboard|rt|\d)|\bsurface(?=,|\s+(?:and|or|lenovo|dell|hp|asus|acer|xps|"
        r"galaxy|pixel|notebook|laptop|tablet)\b)"), "microsoft"),
}
_WORKS_WITH_ROLES = frozenset({"accessory", "part", "consumable"})


def works_with_text(t: str) -> Tuple[str, ...]:
    """WORKS_WITH_LABELS keys a NORMALISED title names, in that order."""
    named = set()
    for tok in _TOKEN_RE.findall(t):
        key = _WW_TOKENS.get(tok)
        if key is not None:
            named.add(key)
            continue
        phrase = _WW_PHRASES.get(tok)
        if phrase is not None:
            if phrase[1] not in named and phrase[0].search(t):
                named.add(phrase[1])
            continue
        if tok[:1] in "ijm":
            for prefix, key in _WW_PREFIXES:
                if tok.startswith(prefix):
                    named.add(key)
                    break
    return tuple(key for key in WORKS_WITH_LABELS if key in named) if named else ()


# ====================================================================== CONDITION
# As the listing words it (the rail's hint). "used" is mostly a verb in titles ("can be
# used with", "used for bedside"): 94 Electronics titles say it, ~5 sell a used item, so
# only a condition phrase counts ("(Used)", "- Used", "Used - Good", a title opening with
# "Used"). A protection plan "for used/refurbished" items is no renewed product.
CONDITION_LABELS = {"renewed": "Renewed / refurbished", "used": "Used"}
_RENEWED_RE = re.compile(r"\b(?:renewed|refurbished)\b")
_USED_RE = re.compile(
    r"\bpre-?owned\b|[(\[]\s*used\s*[)\]]|^used\b(?! (?:for|with|in|to|on|by|as)\b)|"
    r"(?:^|\s)-\s*used\s*(?:$|[-,(]|\b(?:good|very good|like new|acceptable|excellent)\b)|"
    r"\bused\s*-\s*(?:good|very good|like new|acceptable|excellent)\b")


def condition_text(t: str) -> Optional[str]:
    """"renewed" / "used" from a NORMALISED Electronics title, else None."""
    if _RENEWED_RE.search(t):
        return "renewed"
    if "used" in t or "owned" in t:
        if _USED_RE.search(t):
            return "used"
    return None


# ====================================================================== BRAND
# 3,463 Electronics rows name a CHIP vendor as their store ("Lenovo 2023 Yoga 7i ...
# Laptop" sold under "intel"), so a Brands facet on "laptop for college" listed intel,
# amd and nvidia. Where the specs carry the maker ("...; Brand: ASUS; Series: ..."; 1,692
# of them) that is the brand; the rest keep the store. Never "Chipset Brand".
CHIP_VENDORS = frozenset({"intel", "amd", "nvidia", "mediatek", "qualcomm"})
_SPECS_BRAND_RE = re.compile(r"(?:^|;\s*)Brand:\s*([^;]+)")


def brand_of(vertical, store, specs) -> Optional[str]:
    """The `brand` of a row: its store, or the specs' Brand for a chip-vendor store."""
    if not isinstance(store, str):
        return None
    if vertical == ELECTRONICS and store.strip().casefold() in CHIP_VENDORS \
            and isinstance(specs, str):
        found = _SPECS_BRAND_RE.search(specs)
        if found and found.group(1).strip():
            return found.group(1).strip()
    return store


# ====================================================================== annotate
def _column(df: pd.DataFrame, name: str, n: int) -> list:
    return df[name].tolist() if name in df.columns else [None] * n


def annotate_catalog(df: pd.DataFrame) -> pd.DataFrame:
    """Add ADDED_COLUMNS IN PLACE and return the same frame (contract 3.1):

    - item_category / item_role: Electronics and Grocery only ("unknown" / None when no
      rule reads the title), None on Fashion and rows of no known vertical;
    - watch_kind: every row with a title (None when it is not about a watch);
    - diet_tags (Grocery), works_with (Electronics accessories, parts, consumables):
      tuples, () elsewhere;
    - item_condition: Electronics only;
    - brand: every row (the store; see brand_of).

    Reads bsns_vrtcl_name, Product_title, dietary, store and specs where present; never
    modifies them (search_text stays byte-identical). Never filters, concatenates,
    resets or reorders: hydration and GET /product use the ROW POSITION as
    catalog_index. Object dtype, so a row dict reads None or a tuple, never NaN.
    Idempotent. Each distinct (vertical, normalised title) and dietary string is read
    once and mapped back; `brand` reads the specs only for a chip-vendor store."""
    n, index = len(df), df.index
    verticals = [vertical_of(v) for v in _column(df, "bsns_vrtcl_name", n)]
    titles = _column(df, "Product_title", n)
    dietary = _column(df, "dietary", n)
    stores = _column(df, "store", n)
    specs = _column(df, "specs", n)
    categories, roles, kinds, conditions, brands = ([None] * n for _ in range(5))
    diets, works = [()] * n, [()] * n
    title_memo: Dict[Tuple[Optional[str], str], tuple] = {}
    diet_memo: Dict[str, tuple] = {}
    for i in range(n):
        vertical = verticals[i]
        t = _norm(titles[i])
        found = title_memo.get((vertical, t))
        if found is None:
            found = title_memo[(vertical, t)] = _read_title(vertical, t)
        categories[i], roles[i], kinds[i], works[i], conditions[i] = found
        if vertical == GROCERY:
            raw = dietary[i]
            if isinstance(raw, str):
                tags = diet_memo.get(raw)
                if tags is None:
                    tags = diet_memo[raw] = diet_tags(raw)
                diets[i] = tags
        brands[i] = brand_of(vertical, stores[i], specs[i])
    for name, values in ((CATEGORY_COLUMN, categories), (ROLE_COLUMN, roles),
                         (WATCH_COLUMN, kinds), (DIET_COLUMN, diets),
                         (WORKS_WITH_COLUMN, works), (CONDITION_COLUMN, conditions),
                         (BRAND_COLUMN, brands)):
        df[name] = pd.Series(values, index=index, dtype=object)
    if len(df) != n or not df.index.equals(index):
        raise RuntimeError("catalog_taxonomy.annotate_catalog changed the catalog's rows or index")
    return df


def _read_title(vertical: Optional[str], t: str) -> tuple:
    """(category, role, watch_kind, works_with, condition) of one normalised title."""
    if vertical == ELECTRONICS:
        key, kind = _electronics_read(t)
        key = key or UNKNOWN
        role = _ELECTRONICS_TABLE[key][1] if key != UNKNOWN else None
        works = works_with_text(t) if role in _WORKS_WITH_ROLES else ()
        condition = condition_text(t) if role != "service" else None
        return key, role, kind, works, condition
    kind = watch_kind_text(t) if t else None
    if vertical == GROCERY:
        key = GROCERY_TAXONOMY.classify_text(t)[0] or UNKNOWN
        role = _GROCERY_TABLE[key][1] if key != UNKNOWN else None
        return key, role, kind, (), None
    return None, None, kind, (), None


def taxonomy_counts(df: pd.DataFrame) -> dict:
    """Value counts of the added columns for the boot log and the slow test: categories
    and roles per vertical, watch kinds, tag values, conditions, and how many rows take
    their brand from the specs. Counts only, never titles."""
    out: dict = {}
    if CATEGORY_COLUMN not in df.columns:
        return out
    verticals = df["bsns_vrtcl_name"].map(vertical_of) if "bsns_vrtcl_name" in df.columns \
        else pd.Series([None] * len(df), index=df.index, dtype=object)
    for column in (CATEGORY_COLUMN, ROLE_COLUMN):
        per = {}
        for vertical in (ELECTRONICS, GROCERY):
            values = df.loc[verticals == vertical, column]
            if len(values):
                per[vertical] = {str(k): int(v) for k, v in
                                 values.fillna("none").value_counts().items()}
        out[column] = per
    out[WATCH_COLUMN] = {str(k): int(v)
                         for k, v in df[WATCH_COLUMN].dropna().value_counts().items()}
    for column in (DIET_COLUMN, WORKS_WITH_COLUMN):
        counts: Dict[str, int] = {}
        for tags in df[column]:
            for tag in tags or ():
                counts[tag] = counts.get(tag, 0) + 1
        out[column] = dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))
    out[CONDITION_COLUMN] = {str(k): int(v) for k, v in
                             df[CONDITION_COLUMN].dropna().value_counts().items()}
    if "store" in df.columns:
        stores = df["store"].where(df["store"].map(lambda s: isinstance(s, str)), None)
        out["brand_from_specs"] = int((df[BRAND_COLUMN].notna()
                                       & (df[BRAND_COLUMN] != stores)).sum())
    return out
