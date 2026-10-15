"""Product TYPE for Fashion rows, read from the title, and the types a shopper named.

The catalog has no product-type field: Fashion's `categ_lvl2_name` is only a gender
bucket ("Women's Clothing" also holds earrings, shoes and bags), so a clothing ask
had nothing to rank on. `annotate()` derives `product_type` (one of FASHION_TYPES or
"unknown") from Fashion titles at load time; Electronics and Grocery rows get None,
because their nouns ("tea bags", "ring light", "apple watch") collide with the
Fashion vocabulary and are not typed this way.

Pure code (re/pandas only): no LLM, no I/O. search_text and the embeddings are NOT
touched -- the type is a ranking signal for app/main.py, never embedded.

The title rules run in a fixed order, because Amazon titles are keyword-stuffed:
overrides (socks are apparel unless sold as shoes, shoe charms are accessories,
barefoot sandals are jewellery) -> belts a garment word describes ("dress belt") ->
strong non-apparel nouns -> apparel nouns -> weak footwear words -> weak accessory
words. Strong non-apparel nouns beat apparel nouns because accessory titles name
garments as MODIFIERS ("earrings for wedding dress"), while garment titles rarely
name earrings -- except a 'bow tie' neckline or 'handkerchief' hem, which are excepted.
"""

import math
import re
from collections.abc import Mapping
from typing import FrozenSet, Optional, Tuple

import pandas as pd

TYPE_COLUMN = "product_type"
SUBTYPE_COLUMN = "product_subtype"
FASHION_TYPES = ("apparel", "footwear", "jewelry", "bag", "headwear", "accessory")
UNKNOWN = "unknown"


def is_fashion(value) -> bool:
    """Whether a vertical value (or a row's `bsns_vrtcl_name`) is Fashion.

    Case-insensitive on purpose: the local backend's _build_search_text lowercases
    the vertical IN PLACE ('fashion'), Databricks rows keep 'Fashion'. A literal
    == "Fashion" test silently types nothing locally. NaN, None and pd.NA are not
    Fashion (`value or ""` would raise on pd.NA, so they are checked explicitly)."""
    if isinstance(value, (Mapping, pd.Series)):
        value = value.get("bsns_vrtcl_name")
    if value is None or value is pd.NA:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    return str(value).strip().casefold() == "fashion"


# ------------------------------------------------------------------ title rules
# Each rule is a pattern plus literal NEEDLES, one of which every match contains.
# The needle test is a C-level substring check, so the regex only runs on the few
# titles that could match: the prototype ran every pattern on every title (42 s for
# 100K titles). Titles are casefolded once, so no pattern needs re.IGNORECASE.
# tests/test_product_type.py checks that no needle ever hides a match.
def _rule(pattern: str, *needles: str):
    return re.compile(pattern), needles


def _find(rule, text: str):
    pattern, needles = rule
    for needle in needles:
        if needle in text:
            return pattern.search(text)
    return None


_HOSIERY = "intimates/sleep/hosiery"
_OVERRIDES = (  # (rule, type, subtype); first match wins
    (_rule(r"\b(socks?|hosiery|tights|pantyhose|stockings?|leg ?warmers?)\b",
           "sock", "hosiery", "tights", "pantyhose", "stocking", "warmer"),
     "apparel", _HOSIERY),
    (_rule(r"\b(shoe|croc|clog)s?\s*(charms?|laces?|horns?|trees?|racks?|organi[sz]ers?|"
           r"covers?|clips?|deodori[sz]ers?|stretchers?|cleaners?|inserts?)\b|\bshoelaces?\b|"
           r"\binsoles?\b",
           "shoe", "croc", "clog", "insole"),
     "accessory", "shoe care/charms"),
    (_rule(r"\b(jewel(le)?ry|earring|ring|watch)\s*(box(es)?|organi[sz]ers?|holders?|cases?|"
           r"trays?|stands?|displays?|cleaners?|making|storage|winders?)\b",
           "jewel", "ring", "watch"),
     "accessory", "storage/care"),
    (_rule(r"\bbarefoot sandals?\b|\bfoot jewel(le)?ry\b", "barefoot", "foot jewel"),
     "jewelry", "foot jewelry"),
    (_rule(r"\bhair\s*(clips?|pins?|ties?|bands?|bows?|accessor(y|ies)|combs?|sticks?|"
           r"jewel(le)?ry|chains?|vines?|pieces?|wreaths?|slides?|barrettes?|claws?|"
           r"scrunchies?)\b|\bheadbands?\b|\bhead ?pieces?\b|\bbarrettes?\b|\bscrunchies?\b|"
           r"\bfascinators?\b|\btiaras?\b|\bveils?\b|\bbobby pins?\b|\bclaw clips?\b|"
           r"\bflower crowns?\b|\bhead ?wraps?\b",
           "hair", "head", "barrette", "scrunchie", "fascinator", "tiara", "veil", "bobby pin",
           "claw clip", "flower crown"),
     "headwear", "hair accessory"),
)
# Water shoes are sold as "aqua socks" and knit sneakers as "sock sneakers", so the
# socks override typed 109 of 224 water-shoe titles apparel. A plural shoe noun named
# BEFORE the sock word is the product too ("rain boots with removable boot socks");
# after it, after 'for', or right before the sock word, it is what the socks are for
# ("socks for flats", "sneaker socks", "short boots socks", "shoe size 6-12").
_SHOE_NOT_SOCK = _rule(
    r"\b(water|aqua|swim|pool|beach|reef)[- ]?(shoes?|socks?)\b|"
    r"\bsock[- ](shoes|sneakers|trainers)\b",
    "water", "aqua", "swim", "pool", "beach", "reef", "sock")
_PLURAL_SHOE = _rule(
    r"(?<!\bfor )\b(shoes|sneakers|boots|trainers|sandals|loafers|clogs)\b"
    r"(?![- ]?(socks?|cuffs?|liners?|covers?)\b)",
    "shoes", "sneakers", "boots", "trainers", "sandals", "loafers", "clogs")


def _sold_as_shoes(t: str, sock_start: int) -> bool:
    """A title the socks override caught that is really a shoe (see _SHOE_NOT_SOCK)."""
    if _find(_SHOE_NOT_SOCK, t):
        return True
    shoe = _find(_PLURAL_SHOE, t)
    return shoe is not None and shoe.start() < sock_start

_JEWELRY = _rule(
    r"\b(earrings?|ear ?cuffs?|necklaces?|pendants?|lockets?|bracelets?|anklets?|bangles?|"
    r"brooch(es)?|lapel pins?|enamel pins?|cufflinks?|cuff ?links?|charms?|jewel(le)?ry|"
    r"body chains?|belly (button )?rings?|nose rings?|toe rings?|"
    # an O-ring / D-ring / key ring is hardware; '\b' keeps "halo ring", "head ring"
    r"(?<!\bo-)(?<!\bo )(?<!\bd-)(?<!\bd )(?<!\bkey )(?<!key)rings?)\b",
    # 'earring' and 'ear cuff' contain 'ring' and 'cuff'; 'lapel pin'/'enamel pin' end 'el pin'
    "ring", "cuff", "necklace", "pendant", "locket", "bracelet", "anklet", "bangle", "brooch",
    "el pin", "charm", "jewel", "body chain",
)
# Singular loafer/clog/brogue/bootie added to the prototype's plural-heavy list: shoes
# named only by model ("crocs ... graphic clog", "ugg ... loafer") were typed unknown.
# Likewise "ballet flat", and the "aqua socks" the socks override now lets through.
_FOOTWEAR = _rule(
    r"\b(shoes?|sneakers?|sandals?|boots?|booties?|heels|pumps|wedges|loafers?|oxfords|flats|"
    r"ballet flat|mules|clogs?|brogues?|slippers?|flip[- ]?flops?|slides|espadrilles?|"
    r"moccasins?|trainers|cleats|footwear|(water|aqua|swim|pool|beach|reef)[- ]?socks?)\b",
    "shoe", "sneaker", "sandal", "boot", "heels", "pumps", "wedges", "loafer", "oxfords",
    "flats", "ballet flat", "mules", "clog", "brogue", "slipper", "flip", "slides",
    "espadrille", "moccasin", "trainers", "cleats", "footwear", "sock",
)
_BAG = _rule(
    r"\b(bags?|handbags?|purses?|totes?|clutch(es)?|backpacks?|wallets?|cross[- ]?body|"
    r"satchels?|duffels?|duffles?|luggage|suitcases?|fanny packs?|pouch(es)?|wristlets?|"
    r"card (holders?|cases?)|briefcases?|messenger)\b",
    "bag", "purse", "tote", "clutch", "backpack", "wallet", "cross", "satchel", "duff",
    "luggage", "suitcase", "fanny pack", "pouch", "wristlet", "card ", "briefcase", "messenger",
)
_HEADWEAR = _rule(
    r"\b(hats?|caps?|beanies?|fedoras?|visors?|berets?|balaclavas?|turbans?|bonnets?|"
    r"bandanas?|headwear|sun ?hats?)\b",
    "hat", "cap", "beanie", "fedora", "visor", "beret", "balaclava", "turban", "bonnet",
    "bandana", "headwear",
)
_ACCESSORY = _rule(
    r"\b(sunglasses|eyeglasses|glasses|eyewear|goggles|watch(es)?|watch ?bands?|watch ?straps?|"
    r"smartwatch|umbrellas?|keychains?|key ?chains?|key ?rings?|lanyards?|masks?|"
    r"neck ?gaiters?|gaiters?|handkerchiefs?|pocket squares?|garters?|gloves?|mittens?|"
    r"neckties?|bow ?ties?|tie (clips?|bars?|tacks?)|suspenders|collar stays?|arm sleeves|"
    r"sweatbands?|wristbands?|earmuffs|hand ?fans?|folding fans?|patches|pins)\b",
    "glasses", "eyewear", "goggles", "watch", "umbrella", "key", "lanyard", "mask", "gaiter",
    "handkerchie", "pocket square", "garter", "glove", "mitten", "tie", "suspender",
    "collar stay", "arm sleeves", "band", "earmuff", "fan", "patches", "pins",
)
# 'bow tie' and 'handkerchief' also name a NECKLINE, WAIST or HEM ("bow tie neck
# blouse", "handkerchief hem tunic top"): when a garment follows before any 'for',
# 'and', 'with', '&', '+' or '|', the garment is the product (56 of 72 such blouses
# and dresses were typed accessory). "bow tie for tuxedo" and "bow tie and pocket
# square set" stay accessories.
_STYLE_ACCESSORY = _rule(
    r"\b(bow ?ties?|handkerchiefs?)(?=[- ]?(neck|collar|hem|waist)|"
    r"(?:(?!\b(for|and|with)\b|[&+|]).)*?\b(dress(es)?|sundress(es)?|gowns?|blouses?|tops?|"
    r"shirts?|t-?shirts?|tees?|tunics?|tanks?|tankinis?|sweaters?|cardigans?|pullovers?|"
    r"skirts?|pants|shorts|jumpsuits?|rompers?|bodysuits?)\b)",
    "tie", "handkerchie")
_STRONG = (  # (type, subtype or None = the type, rule); first match wins
    ("jewelry", None, _JEWELRY),
    ("footwear", None, _FOOTWEAR),
    ("bag", None, _BAG),
    ("headwear", "hat/cap", _HEADWEAR),
    ("accessory", None, _ACCESSORY),
)

_APPAREL = (  # (subtype, rule); first match wins
    ("swim/coverup", _rule(
        r"\b(swim ?suits?|swimwear|bikinis?|tankinis?|monokinis?|swim ?trunks|board ?shorts|"
        r"rash ?guards?|cover[- ]?ups?|bathing suits?|sarongs?|beachwear|swim)\b",
        "swim", "kini", "board", "rash", "cover", "bathing", "sarong", "beachwear")),
    ("intimates/sleep/hosiery", _rule(
        r"\b(bras?|bralettes?|panties|panty|underwear|lingerie|boxers?|briefs?|shapewear|"
        r"sleepwear|pajamas?|pyjamas?|pj|nightgowns?|nighties?|nightshirts?|robes?|bathrobes?|"
        r"thongs?|undershirts?|loungewear|corsets?|bustiers?|teddies|babydolls?|"
        # a slip, not 'slip-on' shoes or 'non-slip' soles (76 shoes were typed apparel)
        r"(?<!non-)(?<!non )(?<!anti-)(?<!anti )(?<!no-)(?<!no )"
        r"slips?(?![- ]?(on|ons|resistant|proof)\b))\b",
        "bra", "pant", "underwear", "lingerie", "boxer", "brief", "shapewear", "sleepwear",
        "jama", "pj", "night", "robe", "thong", "undershirt", "loungewear", "corset", "bustier",
        "slip", "teddies", "babydoll")),
    ("dress", _rule(
        r"\b(sundress(es)?|gowns?|kaftans?|caftans?|sarees?|saris?|lehengas?|abayas?|qipao|"
        r"cheongsam)\b|\bdress(es)?\b(?!\s*(shirts?|pants?|slacks|trousers?))",
        "dress", "gown", "kaftan", "caftan", "saree", "sari", "lehenga", "abaya", "qipao",
        "cheongsam")),
    ("jumpsuit/romper/set", _rule(
        r"\b(jumpsuits?|rompers?|playsuits?|overalls|onesies?|bodysuits?|outfits?|"
        r"(2|two|3|three)[- ]?pieces?|matching sets?|co-?ords?|tracksuits?|sweatsuits?|"
        r"clothing sets?)\b",
        "jumpsuit", "romper", "playsuit", "overalls", "onesie", "bodysuit", "outfit", "piece",
        "matching set", "co-ord", "coord", "tracksuit", "sweatsuit", "clothing set")),
    ("suit/tailoring", _rule(
        r"\b(suits?|tuxedos?|blazers?|sport ?coats?|waistcoats?|suit jackets?)\b",
        "suit", "tuxedo", "blazer", "sport", "waistcoat")),
    ("outerwear", _rule(
        r"\b(jackets?|coats?|parkas?|windbreakers?|ponchos?|cardigans?|kimonos?|shrugs?|capes?|"
        r"anoraks?|raincoats?|vests?|gilets?|fleeces?)\b",
        "jacket", "coat", "parka", "windbreaker", "poncho", "cardigan", "kimono", "shrug",
        "cape", "anorak", "vest", "gilet", "fleece")),
    ("top", _rule(
        r"\b(shirts?|t-?shirts?|tees?|blouses?|tops?|tanks?|camis?|camisoles?|tunics?|sweaters?|"
        r"hoodies?|sweatshirts?|pullovers?|polos?|henleys?|guayaberas?|jerseys?|turtlenecks?|"
        r"crewnecks?|tee-?shirts?)\b",
        "shirt", "tee", "blouse", "top", "tank", "cami", "tunic", "sweater", "hood", "pullover",
        "polo", "henley", "guayabera", "jersey", "turtleneck", "crewneck")),
    # Singular pant/legging/jogger/... added: ~400 titles ("... grid pant", "capri
    # jegging") were unknown. Not 'short' ('short sleeve'), 'slack', or 'jean'/'trouser'
    # (a "casual jean ... belt", a "trouser belt").
    ("bottom", _rule(
        r"\b(pants?|trousers|jeans|shorts|skirts?|skorts?|leggings?|chinos?|joggers?|capris?|"
        r"culottes|slacks|sweatpants?|khakis|jeggings?|bloomers)\b",
        "pant", "trousers", "jeans", "shorts", "skirt", "skort", "legging", "chino", "jogger",
        "capri", "culottes", "slacks", "khakis", "jegging", "bloomers")),
    ("other-apparel", _rule(
        r"\b(costumes?|uniforms?|scrubs|aprons?|clothes|clothing|apparel|garments?|kurtas?|"
        r"dashikis?|leotards?|unitards?|tutus?|bibs?|maxi|midi|kilts?|tuxes?)\b",
        "costume", "uniform", "scrubs", "apron", "cloth", "apparel", "garment", "kurta",
        "dashiki", "leotard", "unitard", "tutu", "bib", "maxi", "midi", "kilt", "tux")),
)

# Singular shoe words that are also other things ('oxford shirt', 'kentucky derby hat',
# 'mule deer tee', 'day hiker' socks): they decide footwear only when no stronger noun
# and no garment noun matched, i.e. on titles that name a shoe by model alone
# ("clarks ... pitney limit oxford", "rugged outback ... waterproof hiker"). A
# "slip-on" with no garment noun is a shoe too ("vans stormy bird slip-on").
_WEAK_FOOTWEAR = _rule(
    r"\b(oxford|(?<!kentucky )derby|hikers?|mule|pump|heel|slip[- ]?ons?)\b",
    "oxford", "derby", "hiker", "mule", "pump", "heel", "slip")
# ...unless the title also names a garment part or a non-shoe item ("long-sleeve
# oxford" is a shirt, "heel bar buckle ... belt", "pump replacement sleeve").
_WEAK_FOOTWEAR_VETO = _rule(
    r"\b(sleeves?|collar(ed)?|button[- ]?(down|up)|slicker|lifts?|savers?|replacement|belts?)\b",
    "sleeve", "collar", "button", "slicker", "lift", "saver", "replacement", "belt")
# Singular 'tie' is weak, not strong: 'tie front' and 'tie waist' garments name a
# garment noun, which wins; a bare "silk tie" or "tie for a suit" is the tie. Not a
# 'tie dye' print, a 'tie knot' strap or a 'string tie' shoe.
_WEAK = _rule(
    r"\b(belts?|scarf|scarves|shawls?|pashminas?|wraps?|chains?|chokers?|fans?|buttons|"
    r"(?<!string )(?<!string-)ties?(?![- ]?(dyed?|knot|behind)\b)|bow|lanyard)\b",
    "belt", "scar", "shawl", "pashmina", "wrap", "chain", "choker", "fan", "button", "tie",
    "bow", "lanyard",
)
# A belt is the PRODUCT when a garment word says what it is for: "dress belt" (a
# formal belt), "jeans belt", "waist belt", "belts for dresses". The garment rules
# read that word as the product: all 138 'dress belt' titles were typed apparel, and
# an outfit's belt slot then sorted before its shoes. Still a garment: one named
# first and joined by a set word ("wrap dress with waist belt"), one named after the
# belt ("1950s waist belt cocktail swing dress"), and body chains sold as a "waist belt".
_NOT_A_BELT = r"(?![- ]?(bags?|loops?|packs?|pouch))"
_BELT_MODIFIED = _rule(
    r"\b(dress|jeans?|denim|pants|trousers?|slacks|skirt|suit|tuxedo|uniform|waist)[- ]belts?\b"
    + _NOT_A_BELT, "belt")
_BELT_FOR = _rule(
    r"\bbelts?\b" + _NOT_A_BELT + r"[^,;|()]{0,40}?\bfor\b[^,;|()]{0,30}?"
    r"\b(jeans|pants|dress(es)?|trousers|skirts?|shorts|slacks|coats?)\b", "belt")
_BODY_CHAIN = _rule(r"\b(body|belly|waist) chains?\b", "chain")
_SET_JOIN = re.compile(r"\b(with|and|set)\b|\bw/|[&+]")

# Exceptions: a sleeve or toe style is not a hat, 'boot cut' jeans are not boots,
# 'paper bag waist' trousers are not bags, the 'pouch' of men's underwear is not a
# bag, a ring bearer outfit is not a ring.
_STYLE_CAP = _rule(r"\bcaps?[- ](sleeves?|toe)\b", "cap")
_REAL_HAT = _rule(r"\b(hats?|beanies?|baseball caps?|visors?)\b",
                  "hat", "beanie", "baseball cap", "visor")
_BOOTCUT = re.compile(r"\bboot[- ]?cut\b")
_PAPER_BAG = _rule(r"paper ?bag", "paper")
_REAL_BAG = _rule(r"\b(handbags?|purses?|totes?|backpacks?|clutch(es)?|crossbody)\b",
                  "handbag", "purse", "tote", "backpack", "clutch", "crossbody")
_POUCH = re.compile(r"\bpouch(es)?\b")
_UNDERWEAR = _rule(r"\b(underwear|boxers?|briefs?|thongs?|jockstraps?|trunks|bikinis?)\b",
                   "underwear", "boxer", "brief", "thong", "jockstrap", "trunks", "kini")
_RING_BEARER = re.compile(r"\bring bearer\b")

# Garment SETS: "sundress with headband", "shirt + bow tie set" -- the accessory is a
# component and the garment is the product.
_SET_WORD = _rule(r"\b(outfits?|sets?|suits?|tuxedos?|\d ?pcs|pieces?|matching|with)\b",
                  "outfit", "set", "suit", "tuxedo", "pcs", "piece", "matching", "with")
_SET_COMPONENT = _rule(
    r"\b(bow ?ties?|headbands?|suspenders|belts?|necklaces?|hair (bows?|clips?)|hats?|caps?|"
    r"gloves?|scarf|scarves)\b",
    "bow", "headband", "suspender", "belt", "necklace", "hair ", "hat", "cap", "glove", "scar")
_GARMENT = _rule(
    r"\b(dress(es)?|sundress(es)?|gowns?|jumpsuits?|rompers?|shirts?|t-?shirts?|tops?|"
    r"blouses?|pants|shorts|skirts?|leggings|jeans|tuxedos?|suits?|swimsuits?|bikinis?|"
    r"cover[- ]?ups?|onesies?|bodysuits?|pajamas?)\b",
    "dress", "gown", "jumpsuit", "romper", "shirt", "top", "blouse", "pants", "shorts",
    "skirt", "leggings", "jeans", "tuxedo", "suit", "bikini", "cover", "onesie", "bodysuit",
    "pajama")
_SET_ANCHOR = _rule(r"\b(outfits?|tuxedos?|suits?)\b", "outfit", "tuxedo", "suit")
# A set that also names one of these is sold AS that item ("dress with matching
# earrings", "bikini harness ... belly body chain").
_SET_NOT_GARMENT = _rule(
    r"\b(earrings?|bracelets?|anklets?|shoes?|sandals?|sneakers?|boots?|handbags?|purses?|"
    r"bags?|(body|belly|waist) chains?)\b",
    "ring", "bracelet", "anklet", "shoe", "sandal", "sneaker", "boot", "bag", "purse", "chain")


def _garment_set(t: str) -> bool:
    """'X dress with headband' / 'shirt + bow tie set': the garment is the product.
    Only when the garment is named BEFORE the accessory (or the title is an outfit,
    tuxedo or suit). The rarest test runs first and each pattern runs once."""
    component = _find(_SET_COMPONENT, t)
    if component is None or _find(_SET_WORD, t) is None:
        return False
    garment = _find(_GARMENT, t)
    if garment is None:
        return False
    return garment.start() < component.start() or _find(_SET_ANCHOR, t) is not None


def _belt_is_product(t: str) -> bool:
    """'men's leather dress belt', 'belts for jeans': see _BELT_MODIFIED."""
    if _find(_BODY_CHAIN, t):
        return False
    belt = _find(_BELT_MODIFIED, t)
    if belt is not None:
        # what follows 'for' is what the belt is for, not the product
        if _find(_GARMENT, t[belt.end():].split(" for ")[0]):
            return False
        before = t[:belt.start()]
        garments = _GARMENT[0].finditer(before) if _find(_GARMENT, before) else ()
        return not any(_SET_JOIN.search(before[g.end():]) for g in garments)
    belt = _find(_BELT_FOR, t)
    return belt is not None and _find(_GARMENT, t[:belt.start()]) is None


def _apparel_subtype(t: str) -> Optional[str]:
    for subtype, rule in _APPAREL:
        if _find(rule, t):
            return subtype
    return None


def _strong_type(t: str) -> Optional[Tuple[str, str]]:
    for type_, subtype, rule in _STRONG:
        if not _find(rule, t):
            continue
        if type_ == "headwear" and _find(_STYLE_CAP, t) and not _find(_REAL_HAT, t):
            continue
        if type_ == "footwear" and _BOOTCUT.search(t) and not _find(rule, _BOOTCUT.sub("", t)):
            continue
        if type_ == "bag" and _find(_PAPER_BAG, t) and not _find(_REAL_BAG, t):
            continue
        if (type_ == "bag" and _find(_UNDERWEAR, t) and _POUCH.search(t)
                and not _find(rule, _POUCH.sub("", t))):
            continue
        if (type_ == "jewelry" and _RING_BEARER.search(t)
                and not _find(rule, _RING_BEARER.sub("", t))):
            continue
        if (type_ == "accessory" and _find(_STYLE_ACCESSORY, t)
                and not _find(rule, _STYLE_ACCESSORY[0].sub(" ", t))):
            continue
        return type_, subtype or type_
    return None


def _classify_text(t: str) -> Tuple[str, str]:
    """classify_title on already-normalised text (see _normalise)."""
    garment_set = None  # computed at most once, and only when a rule needs it
    for rule, type_, subtype in _OVERRIDES:
        found = _find(rule, t)
        if found:
            if subtype == _HOSIERY and _sold_as_shoes(t, found.start()):
                continue
            if type_ == "headwear":
                garment_set = _garment_set(t)
                if garment_set:
                    break
            return type_, subtype
    # Before the garment-set test, which reads "dress belt with automatic buckle" as a
    # dress sold with a belt
    if _belt_is_product(t):
        return "accessory", "belt"
    if garment_set is None:
        garment_set = _garment_set(t)
    if garment_set and not _find(_SET_NOT_GARMENT, t):
        subtype = _apparel_subtype(t)
        if subtype:
            return "apparel", subtype
    strong = _strong_type(t)
    if strong:
        return strong
    subtype = _apparel_subtype(t)
    if subtype:
        return "apparel", subtype
    if _find(_WEAK_FOOTWEAR, t) and not _find(_WEAK_FOOTWEAR_VETO, t):
        return "footwear", "weak-footwear-word"
    if _find(_WEAK, t):
        return "accessory", "weak-accessory-word"
    return UNKNOWN, ""


def _normalise(title) -> str:
    """Casefolded title with the curly apostrophe straightened; NaN/None -> ''."""
    if title is None or title is pd.NA or (isinstance(title, float) and math.isnan(title)):
        return ""
    return str(title).casefold().replace("\u2019", "'")


def classify_title(title) -> Tuple[str, str]:
    """(type, subtype) of a Fashion title: type in FASHION_TYPES or "unknown".
    Case-insensitive: a lowercased title (the local backend's shape) classifies the same."""
    return _classify_text(_normalise(title))


# "accessories" names the whole non-garment, non-shoe range: "accessories for a black
# dress" wants earrings, clutches, belts and hats. Read only in the shopper's ask and
# in slot phrases, never in titles, where the rules keep their own reading ("hair
# accessories" is headwear, "shoe accessories" are shoe care). A narrower type the
# phrase also names wins ("hair accessories", "watch accessories"); a garment or shoe
# word does not, since it says what the accessories are FOR ("dress accessories").
ACCESSORY_FAMILY = frozenset({"jewelry", "bag", "headwear", "accessory"})
_FAMILY_WORD = _rule(r"\baccessor(y|ies)\b", "accessor")


def _names_family(t: str, type_: str) -> bool:
    return type_ not in ACCESSORY_FAMILY and _find(_FAMILY_WORD, t) is not None


def slot_type(phrase) -> str:
    """The type of a bundle slot phrase ("beach wedding sandals" -> footwear; a bare
    "delicate gold accessories" -> accessory, so it sorts after the shoes)."""
    t = _normalise(phrase)
    type_ = _classify_text(t)[0]
    return "accessory" if _names_family(t, type_) else type_


def slot_accepts(phrase) -> FrozenSet[str]:
    """The row types a bundle slot phrase asks for: its slot_type, the whole
    ACCESSORY_FAMILY for an 'accessories' slot (gold necklaces fit "delicate gold
    accessories"), or an empty set for an untyped slot ("warm desk lamp")."""
    t = _normalise(phrase)
    type_ = _classify_text(t)[0]
    if _names_family(t, type_):
        return ACCESSORY_FAMILY
    return frozenset({type_}) if type_ in FASHION_TYPES else frozenset()


# ------------------------------------------------------------------ annotate
def annotate(df: pd.DataFrame) -> pd.DataFrame:
    """Add `product_type` / `product_subtype` IN PLACE and return the same frame.

    Fashion rows get classify_title(Product_title); every other row (and every row
    of a frame without the vertical or title column) gets None. Both columns are
    object dtype holding str or None, so a row dict reads None, never NaN.

    Never filters, concatenates, resets or reorders: hydration and GET /product use
    the ROW POSITION as catalog_index, so a reordered frame would serve the wrong
    products. Idempotent (the columns are recomputed from the title each call).
    Distinct normalised titles are classified once and mapped back."""
    n_rows, index = len(df), df.index
    types, subtypes = [None] * n_rows, [None] * n_rows
    if "bsns_vrtcl_name" in df.columns and "Product_title" in df.columns:
        titles = df["Product_title"].tolist()
        memo = {}
        for i, vertical in enumerate(df["bsns_vrtcl_name"].tolist()):
            if not is_fashion(vertical):
                continue
            key = _normalise(titles[i])
            found = memo.get(key)
            if found is None:
                found = memo[key] = _classify_text(key)
            types[i], subtypes[i] = found
    df[TYPE_COLUMN] = pd.Series(types, index=index, dtype=object)
    df[SUBTYPE_COLUMN] = pd.Series(subtypes, index=index, dtype=object)
    if len(df) != n_rows or not df.index.equals(index):
        raise RuntimeError("product_type.annotate changed the catalog's rows or index")
    return df


def type_counts(df: pd.DataFrame) -> dict:
    """{type: rows} over the annotated frame, non-Fashion rows as 'none' (for logs)."""
    if TYPE_COLUMN not in df.columns:
        return {}
    counts = df[TYPE_COLUMN].fillna("none").value_counts()
    return {str(k): int(v) for k, v in counts.items()}


# ------------------------------------------------------------------ the shopper's ask
# The HEAD of a query is what the shopper wants; what follows a relational connector
# is context ("something to wear WITH my black boots", "earrings FOR a wedding dress").
# Longest first so ' to go with ' is not cut at its ' with '. ' to wear with ' is cut
# at its ' with ' on purpose: the head keeps 'to wear' ("something to wear"), which is
# the clothing ask; cutting before it would leave "something" and fall back to the
# whole query, where "boots" would win.
_CONNECTORS = (" that goes with ", " to go with ", " to match ", " with ", " for ")
_CONNECTOR_RE = re.compile(
    "|".join(re.escape(c) for c in sorted(_CONNECTORS, key=len, reverse=True)))
# ' plus ' joins two asks ("a dress plus sandals"), but not in "plus size dresses"
_CONJUNCTION_RE = re.compile(r"\s+(?:and|plus(?!\s+sizes?\b))\s+|\s*[,&+]\s*")
_LEADING_AND_RE = re.compile(r"\s*(?:and|plus(?!\s+sizes?\b))\s+")  # the 'and' of ", and"
# "outfit"/"look" name a bundle, not a type: "beach wedding outfit" asks for clothes,
# shoes and accessories together (an outfit bundle never reads requested_types).
_OUTFIT_WORDS_RE = re.compile(r"\b(outfits?|looks?)\b")
_CLOTHING_ASK_RE = re.compile(
    r"\b(clothes|clothing|garments?|apparel|attire|wardrobe|outerwear|activewear|sportswear|"
    r"workwear|athleisure|to wear|wear (to|for)|what (should|shall|can|could|do|would) "
    r"(i|we|you|he|she|they) wear)\b"
)
# The clothing ask as a VERB: only what precedes it can name the type ("what shoes to
# wear" -> footwear, "earrings to wear to a wedding" -> jewelry); after it comes the
# occasion, whose nouns are not the ask ("what to wear to a ring ceremony" -> clothing).
_WEAR_ASK_RE = re.compile(
    r"\b(?:(?:what|something) )?to wear\b|"
    r"\bwhat (?:should|shall|can|could|do|would) (?:i|we|you|he|she|they) wear\b|"
    r"\bwear (?:to|for)\b")
# ...and as a NOUN: an apparel word wherever it stands ("watch party clothes")
_CLOTHING_NOUNS = frozenset({
    "clothes", "clothing", "garment", "garments", "apparel", "attire", "wardrobe",
    "outerwear", "activewear", "sportswear", "workwear", "athleisure"})
# 'cloths' is the common typo for clothes ("cloths for a beach wedding in summer"),
# unless the query is about cleaning cloths.
_CLOTHS_RE = re.compile(r"\bcloths\b")
_CLEANING_RE = re.compile(
    r"\b(clean\w*|microfiber|microfibre|lens(es)?|dish(es)?|polish\w*|wipes?|glasses|"
    r"screens?|kitchen|cars?|table|burp|wash\w*|cheese|dust\w*|mop)\b"
)
_FAMILY_WORDS = frozenset({"accessory", "accessories"})
_FAMILY = "accessory family"  # a _head / _part_type result: the whole ACCESSORY_FAMILY
# A pouch for a device is an Electronics accessory in this catalog ("waterproof phone
# pouch"), not a Fashion bag: untyped, so it never asks for bags.
_DEVICE_POUCH_RE = re.compile(
    r"\b(?:phones?|cell ?phones?|cellphones?|smartphones?|iphones?|tablets?|ipads?|"
    r"kindles?|cameras?) pouch(?:es)?\b")
# Non-Fashion products whose noun is a Fashion type word. Alone ("apple watch band",
# "ring light", "camera bag") such a query keeps its lexical type and main.py's pool
# gate decides (the documented COLLISIONS). Named NEXT TO a Fashion item ("gym shorts
# and an apple watch band") the pool is Fashion-majority, the gate keeps the request,
# and the device rows (product_type None) rank below every Fashion row, out of the
# rerank window: in a conjunction such a part drops the request like an untyped one.
_DEVICES = (r"(?:phones?|cell ?phones?|cellphones?|smart ?phones?|iphones?|tablets?|ipads?|"
            r"kindles?|e-?readers?|cameras?|gopros?|laptops?|computers?|notebooks?|"
            r"macbooks?|chromebooks?|airpods|ear ?buds|headphones?|consoles?|drones?)")
_CROSS_VERTICAL_RE = re.compile(
    r"\b(?:" + _DEVICES + r" (?:bags?|backpacks?|cases?|pouch(?:es)?|sleeves?|straps?|"
    r"wallets?|holders?|rings?|charms?|lanyards?|accessories|stands?|covers?)|"
    r"ring ?lights?|smart ?rings?|smart ?watch(?:es)?|apple ?watch(?:es)?|"
    r"galaxy ?watch(?:es)?|fitbits?|garmins?|"
    r"(?:tea|coffee|trash|garbage|sandwich|freezer|storage|ziploc|sleeping|bean|vacuum) ?"
    r"bags?|lens caps?|(?:cable|zip) ties?|"
    r"(?:desk|ceiling|table|tower|box|usb|portable|floor|window|exhaust) fans?)\b")
# 'black tie' / 'white tie' before an occasion word is a dress code, not a necktie
# ("black tie event", "white tie gala"); a bare "black tie" is still the tie.
_DRESS_CODE_TIES = frozenset({"black-tie", "white-tie"})
_AFTER_DRESS_CODE = frozenset(
    "event events gala galas party parties wedding weddings affair dinner ball optional "
    "occasion function reception celebration night evening".split())
# Pairs whose LEFT noun is the product although a type noun follows it: an earring
# jacket is jewellery, a shoe wardrobe is storage for shoes, not clothing
_LEFT_HEADED_RE = re.compile(r"earrings? jackets?|\S+ wardrobes?")
# After a part's type noun, these say the next conjunction part continues the same
# phrase ("dresses in neutral and earthy tones", "shirts that are breathable and
# wrinkle free"), so an untyped part there is no second product.
_CONTINUERS = frozenset(
    "in on at of from by under over without like near during about into inside around "
    "across through that which who that's thats".split())
# A part that opens with one of these names a thing ("a headlamp", "some sunscreen")
_DETERMINERS = frozenset(
    "a an the some any my your his her our their this that these those another one two "
    "three four five six seven eight nine ten".split())
# A bare 'wear to' preceded by one of these is a question ("what do men wear to black
# tie"), not the compound "wear to work shoes"
_QUESTION_WORDS = frozenset(
    "what which how where when why should shall can could would will do does did".split())
# Words that never name a product on their own: a conjunction part made only of these
# ("black and white dress", "women's plus size dresses", "dress, size 8") is a
# modifier or filler, not a second product. Other untyped parts are judged by their
# shape (see _names_untyped_product). Words that are also products elsewhere stay out
# ('light' is a lamp, 'cream'/'orange'/'mint' are groceries, 'set', 'gear' and 'lace'
# are sold).
_NON_PRODUCT_WORDS = frozenset((
    # function words and fillers
    "a an the some any this that these those my your his her hers their our its i me we "
    "you he she they him them us it and or plus with for to of in on at by from as "
    "what which who how should shall can could do does would will is are be need needs "
    "want wants get buy find looking something anything everything stuff things thing "
    "items item more other others another else etc also too very really just only "
    "matching match new nice cute good best great cheap cheaper affordable budget "
    "under over below above than less "
    # colours that are not also groceries or products
    "black white red blue navy green yellow pink purple brown beige grey gray gold golden "
    "silver rose tan ivory maroon burgundy teal turquoise lavender lilac nude khaki "
    "charcoal multicolor colorful metallic floral striped plaid "
    # gender and age
    "men men's mens man women women's womens woman ladies lady ladies' girl girls girls' "
    "boy boys boys' kid kids kids' child children children's toddler toddlers baby "
    "babies infant teen teens junior juniors adult adults unisex male female "
    # seasons and sizes
    "spring summer fall autumn winter size sizes small medium large xs xl xxl xxxl petite "
    "tall big wide narrow regular "
    # materials and common descriptors
    "cotton linen wool silk satin velvet cashmere suede leather polyester chiffon "
    "comfortable comfy cozy cosy warm cool soft lightweight dark breathable waterproof "
    "stylish casual formal elegant classy chic trendy vintage modest sexy pretty "
    "beautiful simple plain basic classic modern long short loose fitted oversized slim "
    # garment features ("red dress, long sleeve"); a 'laptop sleeve' names its laptop
    "sleeve sleeves sleeveless neck neckline collar waist waisted rise fit hem length "
    # occasions and settings
    "wedding weddings party parties work office school church beach pool travel vacation "
    "holiday hiking camping running gym workout yoga outdoor indoor everyday daily date "
    "night day evening prom interview business event events occasion occasions reception "
    "ceremony gala brunch dinner funeral graduation festival concert cruise trip trips "
    "holidays vacations christmas xmas easter halloween thanksgiving valentines diwali eid "
    # counts, colour and print words ("socks, 6 pairs", "dresses, earthy tones")
    "one two three four five six seven eight nine ten pair pairs pack packs pcs count "
    "tone tones color colors colour colours shade shades print prints pattern patterns "
    "stripes dots always"
).split())
_EDGE_PUNCT_RE = re.compile(r"^\W+|\W+$")
_NUMERIC_RE = re.compile(r"[\d$%.,/x-]+")
# A plural noun ("hand warmers", "trekking poles"), not 'dress', 'gorgeous', 'plus'
_PLURAL_RE = re.compile(r"^[a-z][a-z-]+[^su'\W]s$")
# An adjective's ending ("airy", "breathable", "relaxed", "wrinkle-free"); the stem
# length keeps 'cable', 'bed' and 'tray' out
_ADJECTIVE_RE = re.compile(
    r"^[a-z'-]{2,}[b-df-hj-np-tv-z]y$|"
    r"^[a-z'-]{3,}(?:ful|ous|ive|able|ible|al|ic|ish|less|ed|ing|ant|free|proof|like)$")


def _normalise_query(query) -> str:
    return " " + " ".join(_normalise(query).split()) + " "


def _is_clothing_ask(segment: str, whole: str) -> bool:
    if _CLOTHING_ASK_RE.search(segment):
        return True
    return bool(_CLOTHS_RE.search(segment)) and not _CLEANING_RE.search(whole)


def _bare(token: str) -> str:
    return _EDGE_PUNCT_RE.sub("", token)


def _is_clothing_noun(token: str, whole: str) -> bool:
    bare = _bare(token)
    return bare in _CLOTHING_NOUNS or (bare == "cloths" and not _CLEANING_RE.search(whole))


def _dress_code_tie(tokens, i: int) -> bool:
    """Whether token i is the tie of 'black tie' / 'white tie' before an occasion word."""
    if i + 1 >= len(tokens) or _bare(tokens[i + 1]) not in _AFTER_DRESS_CODE:
        return False
    word = _bare(tokens[i])
    if word in _DRESS_CODE_TIES:
        return True
    return word in ("tie", "ties") and i > 0 and _bare(tokens[i - 1]) in ("black", "white")


def _head(tokens, whole: str) -> Tuple[Optional[str], int]:
    """(type, position) of the RIGHTMOST type noun in `tokens`; type is a FASHION_TYPES
    name or _FAMILY, (None, -1) when no token names a type.

    A query names the product last and its setting before it ("watch party clothes",
    "sandal weather dresses", "engagement ring photoshoot dress"), unlike a title,
    where a strong non-garment noun beats a garment one. A noun is read with its left
    neighbour by the title rules, so compounds resolve like a title ("dress shoes" ->
    footwear, "shoe charms", "apple watch band", "dress belt", "bow tie" -> accessory,
    "hair clips" -> headwear) -- except where the pair's type is only its left noun's
    own type winning by title precedence: then the noun named last is the product
    ("sneaker tee", "lord of the rings t-shirt", "black tie attire", "derby attire" ->
    apparel). A noun a neighbour cancels under the title rules is no noun here either
    ("boot cut jeans", "cap sleeve top", "ring bearer", "tie dye", "d ring belt",
    "kentucky derby hat"), nor is the tie of a dress code ("black tie event")."""
    n = len(tokens)
    if not n:
        return None, -1

    def kind(text, clothing):
        # a clothing-ask noun is an apparel token; the title rules still read it with
        # its neighbour ("party attire" -> apparel, "dress clothes" -> apparel)
        type_ = _classify_text(text)[0]
        return "apparel" if type_ == UNKNOWN and clothing else type_

    own = [kind(t, _is_clothing_noun(t, whole)) for t in tokens]
    pair = [UNKNOWN] + [
        kind(f"{tokens[i - 1]} {tokens[i]}",
             _is_clothing_noun(tokens[i - 1], whole) or _is_clothing_noun(tokens[i], whole))
        for i in range(1, n)]

    def cancelled(i):
        return own[i] != UNKNOWN and (
            (i > 0 and pair[i] == UNKNOWN) or (i + 1 < n and pair[i + 1] == UNKNOWN)
            or _dress_code_tie(tokens, i))

    head, at = None, -1
    for i, token in enumerate(tokens):
        # a type at i read with a cancelled left neighbour would be that neighbour's
        # type out of context ("d ring belt": 'ring belt' is jewelry)
        left = own[i]
        if i and not cancelled(i - 1):
            left = pair[i]
            compound = f"{_bare(tokens[i - 1])} {_bare(token)}"
            if (own[i] != UNKNOWN and own[i - 1] != UNKNOWN and pair[i] == own[i - 1]
                    and not _LEFT_HEADED_RE.fullmatch(compound)):
                # title precedence let the modifier's noun win ("sneaker tee" ->
                # footwear, "rings t-shirt" -> jewelry, "tie attire" -> accessory)
                left = own[i]
        if i and _DEVICE_POUCH_RE.search(f"{tokens[i - 1]} {token}"):
            head, at = None, -1              # an untyped product is the head
        elif _bare(token) in _FAMILY_WORDS:
            # a narrower family type named with it wins ("hair accessories" -> headwear);
            # a garment or shoe word says what they are for ("dress accessories")
            head, at = (left if left in ACCESSORY_FAMILY else _FAMILY), i
        elif own[i] != UNKNOWN:
            if not cancelled(i):
                head, at = left, i
        elif i and pair[i] != UNKNOWN and pair[i] != own[i - 1]:
            head, at = pair[i], i            # a compound only the pair names: 'hair clips'
    return head, at


def _tokens(text: str):
    return _OUTFIT_WORDS_RE.sub(" ", text).split()


# What _part_type says about the conjunction parts after it
_READ_NEXT = 0      # read them as usual
_SKIP_UNTYPED = 1   # an untyped one continues this part's phrase ("dresses in red and blue")
_OCCASION = 2       # they continue a wear-verb ask's occasion ("to a party and a wedding",
                    # "to a wedding and a ring ceremony"): only a part that ENDS on its
                    # type noun is a product ("... to a wedding and a clutch")


def _wear_compound(part: str, ask) -> bool:
    """'wear to work shoes': a bare 'wear to/for' with no question before it and no
    article after it, so the type noun the part ENDS on is the product. Not "what to
    wear to the derby", "what should I wear to a black tie" or "what do men wear to
    black tie", where the occasion follows."""
    if not ask.group().startswith("wear"):
        return False
    before = [_bare(t) for t in part[:ask.start()].split()]
    if any(w in _QUESTION_WORDS or w not in _NON_PRODUCT_WORDS for w in before):
        return False
    return not any(_bare(t) in _DETERMINERS for t in part[ask.end():].split())


def _part_type(part: str, whole: str) -> Tuple[Optional[str], int]:
    """(type, what the next parts are) for one conjunction part: the type is a
    FASHION_TYPES name, _FAMILY or None; the second item is _READ_NEXT, _SKIP_UNTYPED
    or _OCCASION."""
    ask = _WEAR_ASK_RE.search(part)
    if ask:
        # A wear-verb ask is clothing unless what precedes it names a type. What
        # follows is the occasion ("to a ring ceremony", "to the derby", "to a black
        # tie"), also across a conjunction ("to church and brunch") -- except in the
        # compound "wear to work shoes", whose last noun is the product
        type_ = _head(_tokens(part[:ask.start()]), whole)[0]
        if type_:
            return type_, _OCCASION
        if _wear_compound(part, ask):
            after = _tokens(part[ask.end():])
            type_, at = _head(after, whole)
            if type_ and at == len(after) - 1:
                return type_, _READ_NEXT
        return "apparel", _OCCASION
    tokens = _tokens(part)
    type_, at = _head(tokens, whole)
    if type_ is None:
        return None, _READ_NEXT
    text = " ".join(tokens)
    # A garment sold with a component is the garment ("shirt bow tie set"), as in titles
    if _classify_text(text)[0] == "apparel" and _garment_set(text):
        type_ = "apparel"
    if any(_bare(t) in _CONTINUERS for t in tokens[at + 1:]):
        return type_, _SKIP_UNTYPED
    return type_, _READ_NEXT


def _names_untyped_product(part: str, joiner: str, before_typed: bool,
                           after_modifier: bool) -> bool:
    """Whether an untyped conjunction part names a product (a headlamp, hand warmers),
    rather than modifying a typed one ("light, breezy clothes", "dress, size 8").

    By its shape, since no list of adjectives or products is complete: a part opening
    with an article or a count, or ending on a plural noun, is a product ("hiking boots
    and a headlamp", "winter coat and hand warmers"). Otherwise a part before a typed
    part, after a comma, or after another modifier is a modifier ("boho and chic
    dresses", "jeans, 32x30", "clothes, light and airy"), and a part joined by 'and'
    right after a typed one is a product unless it ends like an adjective ("hiking
    boots and headlamp" vs "a dress that's light and flowy"). Residual: a bare singular
    noun BEFORE the typed part ("headlamp and hiking boots") reads as a modifier."""
    tokens = [_bare(t) for t in part.split()]
    words = [w for w in tokens if len(w) > 1 and not _NUMERIC_RE.fullmatch(w)]
    if all(w in _NON_PRODUCT_WORDS for w in words):
        return False                          # 'black', 'size 8', 'something warm'
    if tokens[0] in _DETERMINERS:
        return True                           # 'a headlamp', 'two towels'
    last = words[-1]
    if last not in _NON_PRODUCT_WORDS and _PLURAL_RE.match(last):
        return True                           # 'hand warmers', 'trekking poles'
    if before_typed or after_modifier or joiner == ",":
        return False
    return not _ADJECTIVE_RE.match(last)


def _split_parts(segment: str):
    """[(joiner, part)]: the joiner is '' for the first part, ',' after a comma and
    'and' after and / plus / & / + (", and" counts as 'and')."""
    parts, start, joiner = [], 0, ""
    for m in _CONJUNCTION_RE.finditer(segment):
        parts.append((joiner, segment[start:m.start()]))
        joiner, start = ("," if "," in m.group() else "and"), m.end()
    parts.append((joiner, segment[start:]))
    out = []
    for joiner, part in parts:
        lead = _LEADING_AND_RE.match(part)
        out.append(("and", part[lead.end():]) if lead else (joiner, part))
    return out


def _segment_types(segment: str, whole: str) -> Optional[set]:
    """Types named in one segment; None when a conjunction part names a product no type
    covers. Parts joined by a conjunction ("dress and sandals") each contribute. A part
    like "a headlamp" or "hand warmers" next to "hiking boots" or "winter coat" is a
    product the shopper wants too: a footwear or clothing request would rank it below
    every Fashion row, out of the reranker's window, so no request is made at all. The
    same holds for a non-Fashion product named with a Fashion noun ("gym shorts and an
    apple watch band", see _CROSS_VERTICAL_RE)."""
    parts = _split_parts(segment)
    many = len(parts) > 1
    read, skip_untyped, occasion = [], False, False
    for joiner, part in parts:
        if many and _CROSS_VERTICAL_RE.search(part):
            return None
        if occasion:
            # "what to wear to church and brunch" / "... and a ring ceremony": the
            # occasion goes on; "... to a wedding and a clutch" names a product
            tokens = _tokens(part)
            type_, at = _head(tokens, whole)
            if type_ and at == len(tokens) - 1:
                read.append((joiner, part, type_))
            continue
        type_, rest = _part_type(part, whole)
        if type_ is None and skip_untyped:
            continue                          # "dresses in neutral and earthy tones"
        read.append((joiner, part, type_))
        occasion = rest == _OCCASION
        if type_:
            skip_untyped = rest == _SKIP_UNTYPED
    last_typed = max((i for i, (_, _, t) in enumerate(read) if t), default=-1)
    found, after_modifier = set(), False
    for i, (joiner, part, type_) in enumerate(read):
        if type_:
            found |= ACCESSORY_FAMILY if type_ == _FAMILY else {type_}
            after_modifier = False
        elif many:
            if _names_untyped_product(part, joiner, i < last_typed, after_modifier):
                return None
            after_modifier = True
    return found


def requested_types(query) -> Optional[FrozenSet[str]]:
    """The product types the SHOPPER NAMED, from the query text alone, or None.

    The head (text before the first relational connector) decides, read by its
    rightmost type noun (see _head) per conjunction part; a wear-verb ask reads only
    what precedes it (see _part_type). If it names nothing, only a clothing ask
    anywhere in the query counts ("something for her to wear to a gala" -> apparel): a
    type noun after the connector is the CONTEXT, so "accessories to go with my black
    dress" or "laces for sneakers" must not come back as apparel or footwear. A
    conjunction part that names an untyped or non-Fashion product ("hiking boots and a
    headlamp", "gym shorts and an apple watch band") makes it None. Purely lexical, so
    a lone Electronics/Grocery noun collides ("tea bags" -> bag, "ring light" ->
    jewelry, "apple watch" -> accessory). app/main.py must only act on the result
    inside a Fashion-majority candidate pool, which neutralises those."""
    whole = _normalise_query(query)
    # From 1, so a query that opens with a connector ("for men: dress shoes") keeps it
    # in the head instead of leaving the head empty
    match = _CONNECTOR_RE.search(whole, 1)
    head = whole[:match.start()] if match else whole
    found = _segment_types(head, whole) if head.strip() else set()
    if found is None:
        return None
    if not found and match and _is_clothing_ask(whole, whole):
        found = {"apparel"}
    return frozenset(found) if found else None


def type_rank(product_type, requested) -> int:
    """0 = a requested type (or nothing was requested), 1 = an untyped Fashion row
    ("unknown": mostly shoes named only by model), 2 = any other type.

    A non-Fashion row (product_type None) also ranks 2: it only meets a non-None
    `requested` inside a Fashion-majority pool (main.py's pool gate drops the request
    otherwise), where such rows are noise like lens cloths among dresses."""
    if not requested:
        return 0
    if product_type in requested:
        return 0
    if product_type == UNKNOWN:
        return 1
    return 2
