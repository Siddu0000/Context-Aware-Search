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
barefoot sandals are jewellery, face masks and neck gaiters are accessories) -> belts
a garment word describes ("dress belt") -> strong non-apparel nouns -> apparel nouns
-> weak footwear words -> shoe sizes and shoe models -> weak accessory words. Strong
non-apparel nouns beat apparel nouns because accessory titles name garments as
MODIFIERS ("earrings for wedding dress"), while garment titles rarely name earrings --
except a 'bow tie' neckline or 'handkerchief' hem, which are excepted.

Since 2026-10-19 annotate() also ADDS, all derived at load and never embedded (no
re-embed, search_text byte-identical): `shopper_gender` (the enriched gender tag with
its "unisex" fallback re-read from the title), `audience` (adult / kids / baby),
`fine_subtype` (a shop-style category: dresses, jeans, sneakers...) and `color_family`
(18 swatch families, every vertical). The shopper-side readers requested_fit /
requested_subtypes / requested_audience / implies_women / is_wedding_guest_ask read
the query like requested_types: lexical, English-only and hand-maintained, so
app/main.py acts on them only where it acts on a type request.

Since 2026-10-27 a watch or watch band (app/catalog_taxonomy.py's watch reader) is never
jewelry, even when its title says "bracelet": it types accessory. Since 2026-10-30
requested_types reads the ask the same way ("apple watch bracelet", "bracelet for apple
watch" -> accessory), else those bands ranked below every plain bracelet.
"""

import math
import re
from collections.abc import Mapping
from typing import FrozenSet, Optional, Tuple

import pandas as pd

from app import catalog_taxonomy

TYPE_COLUMN = "product_type"
SUBTYPE_COLUMN = "product_subtype"
# Added 2026-10-19 (annotate); none of them is an embedded field (app/search.py)
GENDER_COLUMN = "shopper_gender"
AUDIENCE_COLUMN = "audience"
FINE_SUBTYPE_COLUMN = "fine_subtype"
COLOR_FAMILY_COLUMN = "color_family"
FASHION_TYPES = ("apparel", "footwear", "jewelry", "bag", "headwear", "accessory")
UNKNOWN = "unknown"
SHOPPER_GENDERS = ("women", "men", "kids", "unisex")
AUDIENCES = ("adult", "kids", "baby")
FITS = ("petite", "maternity", "plus", "tall")


def is_fashion(value) -> bool:
    """Whether a vertical value (or a row's `bsns_vrtcl_name`) is Fashion.

    Case-insensitive on purpose: until 2026-10-15 the local backend's _build_search_text
    lowercased the vertical IN PLACE ('fashion'); both backends now keep 'Fashion', but a
    literal == "Fashion" test would silently type nothing on a lowercase frame. NaN, None and pd.NA are not
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
# Face masks, neck gaiters and balaclavas are accessories, not headwear: 810 of 1,432
# were typed headwear (17% of it) because 'bandana', 'headwrap' or 'balaclava' fired
# first, so a hat query or a Hats tick showed masks. A hat or garment named BEFORE the
# cover is still the product ("trapper hat with face cover", "long sleeve shirt with
# attached face cover"); a "dust cover" is packaging ("stud earrings with dust cover").
_FACE_COVER = _rule(
    r"\b(?:face|mouth|nose|ski|dust|cloth|fabric|cotton|reusable|washable)[- ]?masks?\b|"
    r"\b(?:face|mouth|nose)[- ]?(?:covers?|coverings?)\b|\bgaiters?\b|\bbalaclavas?\b",
    "mask", "cover", "gaiter", "balaclava")
_NAMED_BEFORE_COVER = _rule(
    r"\b(?:hats?|beanies?|caps?|visors?|berets?|fedoras?|bonnets?|ushankas?|helmets?|"
    r"shirts?|t-?shirts?|tees?|tops?|hoodies?|sweatshirts?|jackets?|dress(?:es)?|suits?|"
    r"pajamas?|rompers?|jumpsuits?)\b"
    r"(?![- ]?(?:patterns?|prints?|designs?|sleeves?|toe|grain)\b)",
    "hat", "beanie", "cap", "visor", "beret", "fedora", "bonnet", "ushanka", "helmet",
    "shirt", "tee", "top", "hood", "sweatshirt", "jacket", "dress", "suit", "pajama",
    "romper", "jumpsuit")
_OVERRIDES = (  # (rule, type, subtype); first match wins
    # not a "stocking stuffer" (a fidget keychain) or a "stocking cap" (a beanie)
    (_rule(r"\b(socks?|hosiery|tights|pantyhose|stockings?(?![- ]?(?:stuffers?|caps?)\b)|"
           r"leg ?warmers?)\b",
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
    (_FACE_COVER, "accessory", "accessory"),
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


def _named_before_cover(t: str, cover_start: int) -> bool:
    """A hat or garment named before the face cover is the product (see _FACE_COVER)."""
    first = _find(_NAMED_BEFORE_COVER, t)
    return first is not None and first.start() < cover_start

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
    r"bandanas?|headwear|sun ?hats?|snapbacks?)\b",
    "hat", "cap", "beanie", "fedora", "visor", "beret", "balaclava", "turban", "bonnet",
    "bandana", "headwear", "snapback",
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

# The singular 'jean' and 'short' name the garment in brand titles ("... straight leg
# stretch jean", "amanda skimmer short", "denim short"): 59 + ~150 such rows were
# unknown, 3 of the raw top 12 for "men's jeans" among them. Not a possessive brand
# ("jean's friend"), a name ("jean grey", "jean paul gaultier"), a short sleeve, hair
# or circuit, a lone query word ('short' needs a word before it), and see
# _singular_vetoed (belts, short boots).
_SINGULAR_JEAN = _rule(
    r"\bjean\b(?!'s)(?![- ](?:grey|gray|paul|louis|michel|claude|baptiste|design|pattern|"
    r"print)\b)", "jean")
_SINGULAR_SHORT = _rule(
    # "Short/Long Sleeve ... T-Shirt Tops" is a sleeve length too (2026-10-19 review: such
    # tees were filed under Shorts)
    r"(?<=[\w)] )(?<!life is )short\b(?!\s*(?:/|&|or\b|and\b)\s*long\b)"
    r"(?!(?:[- ][\w/]+)?[- ]sleeves?d?\b)"    # "short puff sleeve", "short cap sleeves"
    r"(?![- ]?(?:long[- ]sleeves?d?|sleeves?d?|circuit|hair|wigs?|fluffy|curly|"
    r"straight|bob|blonde|story|stories|films?|length|boots?|booties|socks?|crew|cuts?|term|"
    r"stack|list|neck|"
    # 'short' before a garment noun is its length ("short kurta")
    r"kurtas?|kurtis?|tunics?|kimonos?|capes?|robes?|gowns?|dress(?:es)?|skirts?|jackets?|"
    r"coats?|tops?|shirts?|tees?|overalls|rompers?|jumpsuits?|tutus?|"
    r"aprons?|uniforms?|costumes?|kilts?|veils?)\b)",
    "short")
# A jean the belt, tie or chain is for (either order); a short boot ("sorel farah short
# ... 7.5 b (m)", "ugg classic short"), a short wig or head scarf: not a garment
_JEAN_ACCESSORY = _rule(
    r"\bjean\b(?:\s+[\w'&-]+){0,4}?\s+(?:belts?|ties?|chains?|buckles?)\b|"
    r"\b(?:belts?|ties?|chains?|buckles?)\b(?:\s+[\w'&-]+){0,3}?\s+jean\b", "jean")
_NOT_A_SHORT = _rule(
    r"\b(?:ugg|sorel|frye|bearpaw|emu|hunter|kamik|bogs|timberland|scarf|scarves|wigs?|"
    r"head ?(?:wraps?|covers?|scarf|scarves)|tichel)\b",
    "ugg", "sorel", "frye", "bearpaw", "emu", "hunter", "kamik", "bogs", "timberland",
    "scar", "wig", "head", "tichel")


def _singular_vetoed(rule, t: str) -> bool:
    """Whether a _SINGULAR_JEAN / _SINGULAR_SHORT match is not the garment."""
    if rule is _SINGULAR_JEAN:
        return _find(_JEAN_ACCESSORY, t) is not None
    if rule is _SINGULAR_SHORT:
        return bool(_find(_SHOE_SIZE, t) or _find(_NOT_A_SHORT, t))
    return False


# What 'dress' modifies when it is not the garment (see the dress rule below). The first
# four kept their old prefix form; the rest are whole words ("dress belted", "dress
# flattering" are dresses). A hyphen joins them too: "Winter Dress-Coats ... Woolen Coat"
# was a dress, out of the window of "women's winter coat" (2026-10-19 review; 1 title).
# The shopper's ask reads the same list (requested_subtypes: "dress watch for men" asked
# for dresses, 2026-10-30 R2): an ask reads as its titles do
_DRESS_MODIFIED = (
    r"(?:shirts?|pants?|slacks|trousers?|(?:suits?|tuxedos?|blazers?|waistcoats?|vests?|"
    r"coats?|jackets?|socks|shoes?|boots?|booties|pumps?|heels?|sandals?|flats?|loafers?|"
    r"oxfords?|sneakers?|belts?|watch(?:es)?|gloves?|uniforms?)\b)")
_DRESS_MODIFIES = r"(?![\s-]*" + _DRESS_MODIFIED + r")"
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
    # 'dress' MODIFYING another noun is not the dress: "formal dress suit", "dress
    # waistcoat", "stiletto heel dress pump" (typed apparel/dress, so a Dresses filter
    # showed them); nor is a "fancy dress" costume (a costume, other-apparel below)
    ("dress", _rule(
        r"\b(sundress(es)?|gowns?|kaftans?|caftans?|sarees?|saris?|lehengas?|abayas?|qipao|"
        r"cheongsam)\b|(?<!fancy )\bdress(es)?\b" + _DRESS_MODIFIES,
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
    # one-word coats ("overcoat", "shacket") were unknown: '\bcoats?' needs the word alone
    # (not "peacoat": it is a colour name in shoe and hoodie titles)
    ("outerwear", _rule(
        r"\b(jackets?|coats?|parkas?|windbreakers?|ponchos?|cardigans?|kimonos?|shrugs?|capes?|"
        r"anoraks?|raincoats?|vests?|gilets?|fleeces?|shackets?|overcoats?|trenchcoats?|"
        r"topcoats?|puffers?)\b",
        "jacket", "coat", "parka", "windbreaker", "poncho", "cardigan", "kimono", "shrug",
        "cape", "anorak", "vest", "gilet", "fleece", "shacket", "puffer")),
    # not "top grain" leather (belts) or a "top-sider" (a boat shoe)
    ("top", _rule(
        r"\b(shirts?|t-?shirts?|tees?|blouses?|tops?(?![- ](?:grain|sider)\b)|tanks?|camis?|"
        r"camisoles?|tunics?|sweaters?|"
        r"hoodies?|sweatshirts?|pullovers?|polos?|henleys?|guayaberas?|jerseys?|turtlenecks?|"
        r"crewnecks?|tee-?shirts?)\b",
        "shirt", "tee", "blouse", "top", "tank", "cami", "tunic", "sweater", "hood", "pullover",
        "polo", "henley", "guayabera", "jersey", "turtleneck", "crewneck")),
    # Singular pant/legging/jogger/... added: ~400 titles ("... grid pant", "capri
    # jegging") were unknown. Not 'slack' or 'trouser' (a "trouser belt"); the singular
    # 'jean' and 'short' follow, with their own guards (_SINGULAR_JEAN, _SINGULAR_SHORT).
    ("bottom", _rule(
        r"\b(pants?|trousers|jeans|shorts|skirts?|skorts?|leggings?|chinos?|joggers?|capris?|"
        r"culottes|slacks|sweatpants?|khakis|jeggings?|bloomers)\b",
        "pant", "trousers", "jeans", "shorts", "skirt", "skort", "legging", "chino", "jogger",
        "capri", "culottes", "slacks", "khakis", "jegging", "bloomers")),
    ("bottom", _SINGULAR_JEAN),
    ("bottom", _SINGULAR_SHORT),
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
# "slip-on" with no garment noun is a shoe too ("vans stormy bird slip-on"), and so are
# a "mary jane" and a "sling-back" ("girls' mary jane ... low wedge", "bandolino ...
# sling-back").
_WEAK_FOOTWEAR = _rule(
    r"\b(oxford|(?<!kentucky )derby|hikers?|mule|pump|heel|slip[- ]?ons?|mary janes?|"
    r"sling[- ]?backs?)\b",
    "oxford", "derby", "hiker", "mule", "pump", "heel", "slip", "mary jane", "sling")
# About 15% of real shoes were `unknown`: their titles name only a model and a size
# ("brooks pureflow 6 black/anthracite 9.5 b (m)", "ugg ... suede 7.5 b (m)", "geox
# boy's newsavage 19 (toddler/little kid/big kid)"), so a footwear ask never seated
# them. A shoe size with a width letter, a kids' shoe-size range, an EU shoe size, or
# a brand's shoe model types them footwear -- only after every noun rule above found
# nothing ("US 6" alone is a dress size, so a bare US number is not enough).
_SHOE_SIZE = _rule(
    r"\b\d{1,2}(?:\.5)?\s?(?:[bcdemnw]|2e|3e|4e|5e|ee|eee|ww|xw)\s?\(\s?[bcdmw]\s?\)|"
    r"\b\d{1,2}(?:\.5)?\s?(?:[bdmnw]|2e|4e|ee|eee|wide|medium|narrow)\s?(?:\(\s?m\s?\)\s?)?us\b|"
    r"\b(?:infant|toddler|little kid|big kid)\s?/\s?(?:infant|toddler|little kid|big kid)\b|"
    r"\b\d{1,2}(?:\.5)?\s(?:toddler|little kid|big kid)\b|"
    r"\b\d{1,2}(?:\.5)?\s(?:m|w|ww|n|d|b|2e|4e)(?=\s?(?:$|[,)/]|-\s|wide\b|medium\b))|"
    r"\beu\s?(?:3[5-9]|4[0-7])(?:\.5)?\b|\b(?:3[5-9]|4[0-7])(?:\.5)?\s?eu\b",
    "(", "us", "/", "toddler", "little kid", "big kid", " m", " w", " n", " d", " b", " 2e",
    " 4e", "eu")
# Shoe models by brand: running and court shoes are sold by model name ("brooks
# glycerin 15", "asics gel-kayano", "nike air max plus"). The brand must come first.
_RUNNING_MODELS = (
    r"\bbrooks\b.*?\b(?:adrenaline|ghost|glycerin|dyad|beast|ariel|addiction|launch|levitate|"
    r"hyperion|ravenna|transcend|cascadia|caldera|revel|pureflow|purecadence|puregrit|bedlam|"
    r"catamount|divide|anthem)\b|"
    r"\basics\b.*?\b(?:gel[- ]?[a-z]+|gt[- ]?\d{3,4}|kayano|nimbus|cumulus|novablast|contend|"
    r"venture|excite|kinsei)\b|"
    r"\bsaucony\b.*?\b(?:guide|ride|triumph|kinvara|echelon|omni|hurricane|peregrine|endorphin|"
    r"cohesion|grid|vortex|xodus|excursion)\b|"
    r"\bhoka(?: one one)?\b.*?\b(?:arahi|bondi|clifton|gaviota|speedgoat|challenger|rincon|mach|"
    r"kawana|torrent|elevon)\b|"
    r"\bmizuno\b.*?\bwave\b|"
    r"\baltra\b.*?\b(?:torin|escalante|lone peak|paradigm|provision|rivera|superior|timp|"
    r"olympus)\b|"
    r"\bnew balance\b.*?\b(?:\d{3,4}v\d+|fresh foam|fuelcell)\b|"
    r"\bnike\b.*?\b(?:air max|air zoom|pegasus|vomero|structure|react|revolution|downshifter|"
    r"free rn|flex runner|tanjun|mercurial|vapormax|metcon|zoomx)\b|"
    r"\badidas\b.*?\b(?:ultra ?boost|pureboost|solar ?boost|supernova|adizero|duramo|"
    r"lite racer|questar|cloudfoam|alphabounce|predator|copa|terrex|predito)\b|"
    r"\bsalomon\b.*?\b(?:speedcross|xa pro|x ultra|sense ride|ultra glide|x mission|amphib)\b|"
    r"\bmerrell\b.*?\b(?:moab|vapor glove|trail glove|agility|bare access|antora|chameleon)\b")
_SNEAKER_MODELS = (
    r"\bconverse\b.*?\b(?:chuck|all star|one star|jack purcell|run star)\b|"
    r"\bvans\b.*?\b(?:old skool|sk8|authentic|atwood|ward|filmore|asher|ultrarange)\b|"
    r"\badidas\b.*?\b(?:superstar|stan smith|gazelle|samba|nmd|forum|campus|seeley)\b|"
    r"\bnike\b.*?\b(?:air force 1|air jordan|jordan \d+|cortez|huarache|presto|dunk|"
    r"court vision)\b|"
    r"\breebok\b.*?\b(?:classic leather|club c|question|zig)\b|"
    r"\bskechers\b.*?\b(?:go ?walk|go ?run|d'lites|uno|arch fit|flex appeal|bobs)\b")
_SHOE_MODEL = _rule(
    _RUNNING_MODELS + "|" + _SNEAKER_MODELS,
    "brooks", "asics", "saucony", "hoka", "mizuno", "altra", "new balance", "nike",
    "adidas", "salomon", "merrell", "converse", "vans", "reebok", "skechers")
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
# Up to three belt words may stand between ("casual dress leather belt", "ratchet dress
# slide belt", "formal dress work belt": ~40 men's belts were typed apparel/dress).
_NOT_A_BELT = r"(?![- ]?(bags?|loops?|packs?|pouch))"
_BELT_GAP = (
    r"(?:[- ](?:genuine|real|top|grain|full|leather|suede|work|slide|ratchet|casual|formal|"
    r"classic|reversible|stitched|braided|woven|elastic|stretch|skinny|thin|wide|western|"
    r"automatic|buckle|track|fashion|golf|plain|smooth|soft|pu|faux|vegan|canvas|nylon|"
    r"black|brown|tan|white|navy|grey|gray)){0,3}")
_BELT_MODIFIED = _rule(
    r"\b(dress|jeans?|denim|pants|trousers?|slacks|skirt|suit|tuxedo|uniform|waist)"
    + _BELT_GAP + r"[- ]belts?\b" + _NOT_A_BELT, "belt")
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
    r"\b(dress(es)?|sundress(es)?|gowns?|jumpsuits?|rompers?|shirts?|t-?shirts?|"
    r"tops?(?![- ](?:grain|sider)\b)|blouses?|pants|shorts|skirts?|leggings|jeans|tuxedos?|"
    r"suits?|swimsuits?|bikinis?|cover[- ]?ups?|onesies?|bodysuits?|pajamas?)\b",
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
        # What follows 'for' is what the belt is for, not the product; nor is a garment
        # word that modifies a second belt ("... waist belt cute dress belt for women")
        after = _BELT_MODIFIED[0].sub(" ", t[belt.end():].split(" for ")[0])
        if not _find(_GARMENT, after):
            before = t[:belt.start()]
            garments = _GARMENT[0].finditer(before) if _find(_GARMENT, before) else ()
            if not any(_SET_JOIN.search(before[g.end():]) for g in garments):
                return True
        # ...else a garment sold with its belt, unless the title also sells a belt FOR
        # garments ("leather belt for jeans and slacks ... fashion dress belt")
    belt = _find(_BELT_FOR, t)
    return belt is not None and _find(_GARMENT, t[:belt.start()]) is None


def _apparel_subtype(t: str) -> Optional[str]:
    for subtype, rule in _APPAREL:
        if _find(rule, t) and not _singular_vetoed(rule, t):
            return subtype
    return None


# A watch or its band is an accessory even when its title says "bracelet" ("Kenneth
# Cole ... Stainless Steel Bracelet Silver Casual Watch", "Leather Replacement Bracelet
# Strap" for an Apple Watch): jewellery is checked first, so 162 Fashion watches and 338
# watch bands were typed jewelry (2026-10-27), and a watch ask's window tiered them
# behind every accessory. catalog_taxonomy's watch reader decides (500 rows moved, 30 of
# 30 sampled were watches or bands); its winders and boxes keep the storage override
# above, its other kinds (charms, chains, parts) keep their reading.
def _watch_or_band(t: str) -> bool:
    return catalog_taxonomy.watch_kind_text(t) in ("watch", "band")


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
        if type_ == "jewelry" and _watch_or_band(t):
            return "accessory", "accessory"
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
            if rule is _FACE_COVER and _named_before_cover(t, found.start()):
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
    if _find(_SHOE_SIZE, t) or _find(_SHOE_MODEL, t):
        return "footwear", "footwear"
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
    Case-insensitive: a lowercased title (the local shape until 2026-10-15) classifies the same."""
    return _classify_text(_normalise(title))


# ------------------------------------------------------------------ fine subtype
# A shop-style category for apparel and footwear rows: the stored product_subtype is
# too coarse for a Style facet or a "jeans, not leggings" ask ("bottom" holds jeans,
# leggings and skirts; "top" holds tees, blouses, sweaters and hoodies). Read from the
# title, first match wins, so the order settles keyword-stuffed titles: ethnic wear
# before dresses ("lehenga choli"), sleep and lingerie before the garment they name
# ("pajama pants", "bikini panties"), dresses before what they modify ("sweater dress",
# "shirt dress"), suits before sets ("3 piece suit"), hoodies and coats before sweaters,
# shorts and skirts before jeans ("jean shorts", "denim skirt"). Other types: None.
FINE_SUBTYPE_LABELS = {
    "dresses": "Dresses",
    "tops_tees": "Tops & tees",
    "blouses_shirts": "Blouses & shirts",
    "sweaters": "Sweaters & cardigans",
    "hoodies": "Hoodies & sweatshirts",
    "coats_jackets": "Coats, jackets & vests",
    "suits_blazers": "Suits & blazers",
    "jeans": "Jeans",
    "pants": "Pants & trousers",
    "shorts": "Shorts",
    "leggings": "Leggings",
    "skirts": "Skirts",
    "jumpsuits_sets": "Jumpsuits, rompers & sets",
    "swim": "Swimwear & cover-ups",
    "sleep_lounge": "Sleep & lounge",
    "lingerie": "Bras, lingerie & underwear",
    "socks_hosiery": "Socks & hosiery",
    "ethnic_wear": "Ethnic & traditional wear",
    "sneakers": "Sneakers",
    "athletic": "Athletic shoes",
    "boots": "Boots",
    "sandals": "Sandals",
    "heels": "Heels & pumps",
    "flats": "Flats, loafers & oxfords",
    "slippers": "Slippers",
}
FOOTWEAR_SUBTYPES = frozenset(
    {"sneakers", "athletic", "boots", "sandals", "heels", "flats", "slippers"})
APPAREL_SUBTYPES = frozenset(FINE_SUBTYPE_LABELS) - FOOTWEAR_SUBTYPES

# A 'suit' that is no tailoring. Athletic and lounge sets (jogging, sport, workout,
# sauna, "hoodie and sweatpants suit" ...) were ~90 of the 782 "Suits & blazers" rows
# ("Mens Hooded Jogging Suits ... Sports Suits"): they are sets (_ATHLETIC_SUITS), a
# "romper suit" is a romper, a diving suit no tailoring (2026-10-19 review).
_SUIT_NOT = (r"(?<!track )(?<!sweat )(?<!jump )(?<!snow )(?<!ski )(?<!wet )(?<!body )"
             r"(?<!play )(?<!swim )(?<!bathing )(?<!rain )(?<!space )(?<!flight )(?<!boiler )"
             r"(?<!birthday )(?<!blow up )(?<!jogging )(?<!jogger )(?<!sport )(?<!sports )"
             r"(?<!running )(?<!workout )(?<!training )(?<!athletic )(?<!sauna )(?<!gym )"
             r"(?<!fitness )(?<!yoga )(?<!tracksuit )(?<!sweatpants )(?<!sweatshirt )"
             r"(?<!hoodie )(?<!romper )(?<!diving )(?<!dive )")
_ATHLETIC_SUIT_WORDS = (r"jogging|jogger|sports?|running|workout|training|athletic|sauna|gym|"
                        r"fitness|yoga|tracksuit|sweatpants|sweatshirt|hoodie|ski|play")
_ATHLETIC_SUITS = "(?:" + _ATHLETIC_SUIT_WORDS + ")"
_FINE_APPAREL = (
    ("ethnic_wear", _rule(
        r"\b(sarees?|saris?|lehengas?|lehnga|anarkalis?|kurtas?|kurtis?|salwars?|shalwars?|"
        r"kameez|churidars?|dupattas?|sherwanis?|cholis?|ghagras?|dhotis?|lungis?|abayas?|"
        r"jilbabs?|qipaos?|cheongsams?|hanboks?|ao dai|dashikis?|dirndls?|lederhosen|thobes?|"
        r"kebayas?)\b",
        "sare", "sari", "lehe", "lehn", "anarkali", "kurt", "salwar", "shalwar", "kameez",
        "churidar", "dupatta", "sherwani", "choli", "ghagra", "dhoti", "lungi", "abaya",
        "jilbab", "qipao", "cheongsam", "hanbok", "ao dai", "dashiki", "dirndl", "lederhosen",
        "thobe", "kebaya")),
    ("swim", _rule(
        r"\b(swim\w*|bathing suits?|tankinis?|monokinis?|board ?shorts|rash ?guards?|"
        r"cover[- ]?ups?|sarongs?|beachwear|burkinis?)\b",
        "swim", "bathing", "tankini", "monokini", "board", "rash", "cover", "sarong",
        "beachwear", "burkini")),
    ("sleep_lounge", _rule(
        r"\b(pajamas?|pyjamas?|pjs?|sleepwear|nightgowns?|nighties?|nightshirts?|nightdress(es)?|"
        r"night ?gowns?|sleep ?shirts?|robes?|bathrobes?|dressing gowns?|housecoats?|"
        r"house ?dress(es)?|loungewear|lounge (?:sets?|pants|shorts|wear)|"
        r"sleep (?:sets?|pants|shorts|dress(?:es)?|tops?)|sleepers?|sleep (?:&|and|n) play)\b",
        "jama", "pj", "sleep", "night", "robe", "dressing gown", "housecoat", "house", "lounge")),
    ("lingerie", _rule(
        r"\b(bras?|bralettes?|panties|panty|underwear|lingerie|boxers?|briefs?|thongs?|"
        r"g-?strings?|shapewear|corsets?|bustiers?|teddies|babydolls?(?! dress)|"
        r"bodystockings?|undershirts?|long johns|jockstraps?|"
        r"(?<!non-)(?<!non )(?<!anti-)(?<!anti )(?<!no-)(?<!no )"
        r"slips?(?![- ]?(on|ons|resistant|proof|dress(es)?)\b))\b",
        "bra", "pant", "underwear", "lingerie", "boxer", "brief", "thong", "string",
        "shapewear", "corset", "bustier", "teddies", "babydoll", "bodystocking", "undershirt",
        "long johns", "jockstrap", "slip")),
    ("swim", _rule(r"\bbikinis?\b", "bikini")),   # after lingerie: "bikini panties"
    ("socks_hosiery", _rule(
        r"\b(socks?|tights|pantyhose|stockings?(?![- ]?(?:stuffers?|caps?)\b)|leg ?warmers?|"
        r"hosiery)\b",
        "sock", "tights", "pantyhose", "stocking", "warmer", "hosiery")),
    ("dresses", _rule(
        r"\b(sundress(es)?|gowns?|kaftans?|caftans?|sweaterdress(es)?|shirtdress(es)?)\b|"
        r"(?<!fancy )\bdress(es)?\b(?![- ]up\b)" + _DRESS_MODIFIES,
        "dress", "gown", "kaftan", "caftan")),
    ("suits_blazers", _rule(
        # the veto guards the bare 'suit' only: an "athletic sport coat" is a sport coat
        r"\b(" + _SUIT_NOT + r"suits?|tuxedos?|tux|blazers?|sport ?coats?|waistcoats?)\b",
        "suit", "tux", "blazer", "sport", "waistcoat")),
    ("jumpsuits_sets", _rule(
        r"\b(jumpsuits?|jump suits?|rompers?|playsuits?|overalls?|dungarees|onesies?|"
        r"bodysuits?|body suits?|unitards?|leotards?|tracksuits?|track suits?|sweatsuits?|"
        r"sweat suits?|snowsuits?|snow suits?|" + _ATHLETIC_SUITS + r" suits?|coveralls?|"
        r"outfits?|co-?ords?|sets?(?! of\b))\b",
        "jump", "romper", "playsuit", "overall", "dungarees", "onesie", "bod", "unitard",
        "leotard", "track", "sweat", "snow", "suit", "coverall", "outfit", "co-ord", "coord",
        "set")),
    ("hoodies", _rule(r"\b(hoodies?|hoody|sweatshirts?)\b", "hood", "sweatshirt")),
    ("coats_jackets", _rule(
        r"\b(jackets?|coats?|parkas?|windbreakers?|ponchos?|anoraks?|raincoats?|"
        r"(?<!sweater )vests?|gilets?|puffers?|trench|trenchcoats?|overcoats?|topcoats?|"
        r"shackets?|capes?|bombers?)\b",
        "jacket", "coat", "parka", "windbreaker", "poncho", "anorak", "vest", "gilet",
        "puffer", "trench", "shacket", "cape", "bomber")),
    ("sweaters", _rule(
        r"\b(sweaters?|cardigans?|pullovers?|jumpers?|turtlenecks?|shrugs?|boleros?|"
        r"kimonos?)\b",
        "sweater", "cardigan", "pullover", "jumper", "turtleneck", "shrug", "bolero",
        "kimono")),
    ("skirts", _rule(r"\b(skirts?|skorts?|tutus?)\b", "skirt", "skort", "tutu")),
    ("shorts", _rule(r"\b(shorts|bermudas?)\b", "shorts", "bermuda")),
    ("shorts", _SINGULAR_SHORT),              # vetoed like the type rule
    ("jeans", _rule(
        r"\b(jeans|jeggings?)\b|\bjean\b(?!'s)(?![- ](?:grey|gray|paul|louis|michel|claude|"
        r"jackets?|shirts?|vests?|dress(?:es)?|skirts?|shorts?|overalls|rompers?)\b)",
        "jean", "jegging")),
    # not what a top is worn with ("Tunic Tops for Leggings" was filed under Leggings)
    ("leggings", _rule(r"(?<!for )(?<!with )(?<!over )\b(leggings?|yoga pants|tights)\b",
                       "legging", "yoga pants", "tights")),
    ("pants", _rule(
        r"\b(pants?|trousers?|chinos?|joggers?|capris?|culottes|slacks|sweatpants?|khakis|"
        r"palazzos?|cargos)\b",
        "pant", "trouser", "chino", "jogger", "capri", "culottes", "slacks", "khakis",
        "palazzo", "cargos")),
    ("blouses_shirts", _rule(
        r"\b(blouses?|tunics?|button[- ]?(?:down|up)s?|dress shirts?|oxford shirts?|oxfords?|"
        r"flannels?|hawaiian shirts?|aloha shirts?|guayaberas?|western shirts?|camp shirts?|"
        r"chambray|linen shirts?|tuxedo shirts?)\b",
        "blouse", "tunic", "button", "dress shirt", "oxford", "flannel", "hawaiian", "aloha",
        "guayabera", "western shirt", "camp shirt", "chambray", "linen shirt", "tuxedo shirt")),
    ("tops_tees", _rule(
        r"\b(t-?shirts?|tee-?shirts?|tees?|tanks?|camis?|camisoles?|crop tops?|"
        r"tops?(?![- ](?:grain|sider)\b)|polos?|henleys?|jerseys?|shirts?|halters?|"
        r"singlets?|crewnecks?)\b",
        "shirt", "tee", "tank", "cami", "top", "polo", "henley", "jersey", "halter", "singlet",
        "crewneck")),
    # A fleece named alone is the zip-up layer ("Gamer Fleece", "zip fleece"). Last,
    # since "fleece" mostly describes another garment ("fleece leggings", "fleece
    # sweatpant": 241 rows the sweaters rule used to take)
    ("coats_jackets", _rule(r"\bfleeces?\b(?![- ]?(?:lined|lining)\b)", "fleece")),
)
_FINE_FOOTWEAR = (
    ("slippers", _rule(r"\b(slippers?|house ?shoes?|scuffs?)\b", "slipper", "house", "scuff")),
    ("boots", _rule(
        r"\b(boots?|booties|bootie|wellies|wellingtons?|chukkas?|hikers?|"
        r"ugg\b.*?\bclassic (?:short|mini|tall|ultra))\b",
        "boot", "wellies", "wellington", "chukka", "hiker", "ugg")),
    ("sandals", _rule(
        r"\b(sandals?|flip[- ]?flops?|slides?|gladiators?)\b",
        "sandal", "flip", "slide", "gladiator")),
    ("athletic", _rule(
        r"\b(running|runners?|trail|training|trainers?|cross[- ]?trainers?|basketball|tennis|"
        r"volleyball|soccer|football|baseball|softball|golf|bowling|cycling|cleats?|spikes|"
        r"track|walking|hiking|athletic|gym|workout|jogging|marathon)\b|" + _RUNNING_MODELS,
        "run", "trail", "train", "basketball", "tennis", "volleyball", "soccer", "football",
        "baseball", "softball", "golf", "bowling", "cycling", "cleat", "spikes", "track",
        "walking", "hiking", "athletic", "gym", "workout", "jogging", "marathon", "brooks",
        "asics", "saucony", "hoka", "mizuno", "altra", "new balance", "nike", "adidas",
        "salomon", "merrell")),
    ("sneakers", _rule(
        r"\b(sneakers?|high[- ]?tops?|low[- ]?tops?|canvas shoes?|skate shoes?|"
        r"slip[- ]?ons?)\b|" + _SNEAKER_MODELS,
        "sneaker", "high", "low", "canvas", "skate", "slip", "converse", "vans", "adidas",
        "nike", "reebok", "skechers")),
    ("heels", _rule(
        r"\b(heels|high heels?|heeled|heel|pumps?|stilettos?|wedges?|platforms?|"
        r"sling[- ]?backs?|d'orsay)\b",
        "heel", "pump", "stiletto", "wedge", "platform", "sling", "d'orsay")),
    ("flats", _rule(
        r"\b(flats|flat|ballet flats?|loafers?|moccasins?|oxfords?|brogues?|derby|derbies|"
        r"boat shoes?|deck shoes?|mary janes?|driving shoes?|espadrilles?|dress shoes?|"
        r"ballets?|monk straps?)\b",
        "flat", "loafer", "moccasin", "oxford", "brogue", "derb", "boat", "deck", "mary jane",
        "driving", "espadrille", "dress shoe", "ballet", "monk")),
)


# Words that name a garment alone but only describe the tee, tank or top they sit next to
# (2026-10-19 review). First match wins in a fixed order, so "Basic Plain Tunic Tee" was
# a blouse, "T-Shirt V-Neck Pullover" a sweater and "Tank Tops Vest" a coat, and a
# t-shirt ask partitioned them behind every matching row (a rerank-92 tee fell to #87).
# Such a word is skipped when the title also names one of its head nouns and nothing
# else of that rule ("Puffer Vest", "Pullover Sweater", "Tunic Blouse" keep theirs).
# Catalog: 873 apparel rows moved, none changed type; of 6,709 apparel titles saying
# t-shirt or tee, tops_tees went 4,302 -> 4,584 (most of the rest are t-shirt dresses,
# tee-and-shorts sets and "T-Shirt Blouse" keyword stuffing).
_TEE_NOUN = (r"t-?shirts?|tee-?shirts?|tees?|tank ?tops?|tanks?|camis?|camisoles?")
# A bare "top" is no head noun: "Open Front Plaid Vest Cowl Neck Tops" is a vest and
# "Fleece Pullover Top" a sweater (the catalog diff, 2026-10-19)
_TOP_MODIFIERS = {   # key: (modifier pattern, head nouns that make it a modifier)
    "coats_jackets": (re.compile(r"\bvests?\b"),
                      _rule(r"\b(?:" + _TEE_NOUN + r"|(?:vest|crop|halter) tops?)\b", "shirt",
                            "tee", "tank", "cami", "top")),
    "sweaters": (re.compile(r"\bpullovers?\b"),
                 _rule(r"\b(?:" + _TEE_NOUN + r"|shirts?|blouses?|tunics?|henleys?|"
                       r"polos?)\b", "shirt", "tee", "tank", "cami", "blouse", "tunic",
                       "henley", "polo")),
    "blouses_shirts": (re.compile(r"\btunics?\b"),
                       _rule(r"\b(?:" + _TEE_NOUN + r")\b", "shirt", "tee", "tank", "cami")),
}


def _only_a_top_modifier(key: str, rule, t: str) -> bool:
    """Whether the rule for `key` matched `t` only on a word describing a named top."""
    modifier = _TOP_MODIFIERS.get(key)
    if modifier is None:
        return False
    word, heads = modifier
    if not word.search(t) or _find(heads, t) is None:
        return False
    return rule[0].search(word.sub(" ", t)) is None


def _fine_subtype(t: str, type_) -> Optional[str]:
    """The FINE_SUBTYPE_LABELS key of a normalised title of this type, or None."""
    if type_ == "apparel":
        rules = _FINE_APPAREL
    elif type_ == "footwear":
        rules = _FINE_FOOTWEAR
    else:
        return None
    for key, rule in rules:
        if _find(rule, t) and not _singular_vetoed(rule, t) and \
                not _only_a_top_modifier(key, rule, t):
            return key
    return None


def fine_subtype(title, product_type=None) -> Optional[str]:
    """fine_subtype of one title; the type defaults to classify_title's."""
    t = _normalise(title)
    return _fine_subtype(t, product_type or _classify_text(t)[0])


# ------------------------------------------------------------------ colour family
# 18 swatch families (ui_spec 7.3). The enriched `color` is already canonical on 99.5%
# of tagged Fashion rows; Electronics carries more seller noise ("black1", "sliver",
# "rosegold"). multicolor and clear are drawn specially by the UI (a conic gradient, a
# checkerboard), their hex is only a fallback.
COLOR_FAMILIES = {
    "black": "#111111", "blue": "#1F5FBF", "white": "#FFFFFF", "red": "#C40000",
    "grey": "#8C8C8C", "silver": "#C0C0C0", "pink": "#F4A7B9", "green": "#2E7D32",
    "gold": "#D4AF37", "brown": "#6D4C41", "beige": "#E8DCC4", "purple": "#6A1B9A",
    "navy": "#1B2A4A", "yellow": "#F7D600", "orange": "#F57C00", "burgundy": "#800020",
    "multicolor": "#9E9E9E", "clear": "#F5F5F5",
}
COLOR_LABELS = {k: ("Multicolor" if k == "multicolor" else k.capitalize())
                for k in COLOR_FAMILIES}
SPECIAL_SWATCHES = frozenset({"multicolor", "clear"})
# Words that name a family (whole value or inside one: "navy blue", "black1", "rose gold")
_COLOR_WORDS = {
    **{k: k for k in COLOR_FAMILIES},
    "gray": "grey", "charcoal": "grey", "smoke": "grey", "graphite": "grey", "slate": "grey",
    "multi": "multicolor", "multicolour": "multicolor", "multicolored": "multicolor",
    "multicoloured": "multicolor", "colorful": "multicolor", "colourful": "multicolor",
    "rainbow": "multicolor", "assorted": "multicolor", "mix": "multicolor",
    "mixed": "multicolor", "camo": "multicolor", "camouflage": "multicolor",
    "leopard": "multicolor", "floral": "multicolor", "paisley": "multicolor",
    "transparent": "clear",
    "indigo": "blue", "royal": "blue", "sky": "blue", "aqua": "blue", "aquamarine": "blue",
    "turquoise": "blue", "teal": "blue", "cobalt": "blue", "denim": "blue",
    "ruby": "red", "crimson": "red", "scarlet": "red", "firebrick": "red", "coral": "orange",
    "wine": "burgundy", "maroon": "burgundy", "claret": "burgundy", "oxblood": "burgundy",
    "amethyst": "purple", "lavender": "purple", "lilac": "purple", "violet": "purple",
    "plum": "purple", "mauve": "purple", "magenta": "pink", "fuchsia": "pink", "rose": "pink",
    "blush": "pink",
    "brass": "gold", "rosegold": "gold", "golden": "gold", "champagne": "beige",
    "bronze": "brown", "copper": "brown", "walnut": "brown", "coffee": "brown",
    "chocolate": "brown", "mocha": "brown", "chestnut": "brown", "cognac": "brown",
    "stainless": "silver", "steel": "silver", "pewter": "silver", "platinum": "silver",
    "aluminum": "silver", "aluminium": "silver", "chrome": "silver", "silvery": "silver",
    "sliver": "silver", "gunmetal": "grey",
    "khaki": "beige", "tan": "beige", "camel": "beige", "cream": "beige", "ivory": "beige",
    "nude": "beige", "taupe": "beige", "sand": "beige", "wheat": "beige", "natural": "beige",
    "apricot": "orange", "rust": "orange", "tangerine": "orange",
    "mustard": "yellow", "lemon": "yellow", "tricolor": "multicolor",
    "lime": "green", "jade": "green", "olive": "green", "mint": "green",
    "malachite": "green", "emerald": "green", "sage": "green",
    "blac": "black", "balck": "black", "carbon": "black", "jet": "black", "onyx": "black",
}
# A colour word inside a longer value; digits may touch it ("black1", "beige664")
_COLOR_WORD_RE = re.compile(
    r"(?<![a-z])(" + "|".join(sorted((re.escape(w) for w in _COLOR_WORDS), key=len, reverse=True))
    + r")(?![a-z])")
# Words that modify a following colour word ("rose gold", "steel blue", "royal blue")
_COLOR_MODIFIERS = frozenset({"rose", "steel", "slate", "royal", "sky", "aqua", "wine",
                              "sand", "jet", "carbon", "natural", "mint", "sage"})


def color_family(value) -> Optional[str]:
    """The COLOR_FAMILIES key for a raw `color` value, or None (missing, a size, "as
    shown", a pattern name). A value naming several families takes the first ("navy
    blue" -> navy, "black/white" -> black), skipping a modifier another colour word
    follows ("rose gold" -> gold, "steel blue" -> blue)."""
    if not isinstance(value, str):
        return None
    v = " ".join(value.casefold().split())
    if not v:
        return None
    if v in _COLOR_WORDS:
        return _COLOR_WORDS[v]
    words = [m.group(1) for m in _COLOR_WORD_RE.finditer(v)]
    for i, word in enumerate(words):
        if word in _COLOR_MODIFIERS and i + 1 < len(words):
            continue
        return _COLOR_WORDS[word]
    return None


# ------------------------------------------------------------------ shopper gender, audience
# The enriched `gender` tag copies `categ_lvl2_name`, and its "unisex" is the loader's
# FALLBACK (no department, no gender word): only 22 of 37 sampled "unisex" rows were
# unisex and 11 were women's (ballet flats, tennis bracelets, a bikini cover-up,
# "necklace for women18"), so a Women tick missed ~14% of women's items. Kids rows
# were neutral to the ranking, and a few department-filed rows contradict their title
# ("OutTop ... Coats for Women" under Kids, "Girl's ... Skater Dress" under Women).
# shopper_gender keeps the tag and re-reads the title only where it is a fallback or
# contradicted; "for women girls" keyword stuffing stays women.
# Every cue is a needled rule: a title is read for a cue only when the decision needs it
# (a women- or men-tagged title only for a kid word, the usual case), which keeps the
# per-title cost of annotate() small on ~100K Fashion titles.
_DIGIT_T = tuple(f"{d}t" for d in range(10))   # the toddler-size needles ("3t", "2t-4t")
_KID_STRONG = _rule(
    r"\b(?:(?<!stray )(?<!new )kids?|kid's|kids'|toddlers?|toddler's|infants?|infant's|"
    r"newborns?|children|children's|childrens|child's|youth|little (?:kids?|girls?|boys?)|"
    r"big (?:kids?|girls?|boys?)|girl's|boy's|"
    r"bab(?:y|ies|y's)(?![- ]?(?:blue|pink|doll|dolls|shower|tee|tees|breath|yoda|bump|face))|"
    r"preschool|kindergarten)\b|"
    # a plural possessive ends on the apostrophe, where a closing \b never matches:
    # "Boys' Woven Boxer" under Men read adult (2026-10-19 review)
    r"\b(?:girls|boys)'(?!\w)|"
    # a toddler size ("3t", "2t-4t"), not "5 t-shirts"
    r"\b\d{1,2}t\b(?![-'])|"
    r"\b(?:[1-9]|1[0-4])\s?(?:-\s?(?:[1-9]|1[0-6])\s?)?(?:years?|yrs?)\b(?! anniversary)|"
    r"\b\d{1,2}\s?(?:-\s?\d{1,2}\s?)?(?:months?|mos)\b",
    "kid", "girl", "boy", "bab", "year", "yr", "month", "mos", "toddler", "child", "infant",
    "newborn", "youth", "preschool", "kindergarten", *_DIGIT_T)
_KID_WEAK = _rule(r"\b(?:girls?|boys?|teen\s?girls?|teen\s?boys?)\b", "girl", "boy")
_WOMEN_WORD = _rule(
    r"\b(?:women|women's|womens|woman's|(?<!wonder )(?<!spider-)(?<!cat )woman|ladies|"
    r"ladies'|lady's|female|females|feminine|juniors|junior's|maternity|pregnan\w*|moms?|"
    r"mom's|mommy|mama|mother|mother's|wife|girlfriend|grandma|grandmother|granny|bride|"
    r"brides|bridal|bridesmaids?)\b|(?:women|womens|ladies)(?=\d)|\bfor her\b",
    "women", "woman", "ladies", "lady", "female", "feminine", "junior", "maternity", "pregnan",
    "mom", "mama", "mother", "wife", "girlfriend", "grandma", "granny", "bride", "bridal",
    "for her")
_MAN_COMPOUND = (r"(?<!spider-)(?<!spider )(?<!iron )(?<!ant-)(?<!pac-)(?<!he-)(?<!bat)"
                 r"(?<!snow)(?<!rain )(?<!cave )(?<!sand)(?<!candy )(?<!gingerbread )"
                 r"(?<!tin )(?<!mega)(?<!lucky )(?<!old )(?<!family )(?<!pac )")
_MEN_WORD = _rule(
    r"\b(?:men|men's|mens|male|males|gentlem[ae]n|gentlemen's|gents?|husband|"
    r"(?:boyfriend|dad)(?![- ]?(?:jeans?|fit|shirts?|tees?|cardigans?|blazers?|hats?|caps?)\b)|"
    r"dads|daddy|father|father's|grandpa|grandfather|groom|grooms|groomsm[ae]n)\b|"
    r"\b" + _MAN_COMPOUND + r"man(?:'s)?\b(?! of steel| united| utd|'s best friend)|\bfor him\b",
    "men", "man", "male", "gent", "husband", "boyfriend", "dad", "father", "grandpa", "groom",
    "for him")
_ADULT = _rule(r"\b(?:adults?|teens?|teenagers?|teenage|grown[- ]ups?)\b",
               "adult", "teen", "grown")
# Items only women's wear names, read only where the tag is the fallback "unisex"
_WOMEN_CONVENTION = _rule(
    r"\b(?:dress(?:es)?(?!\s*(?:shirts?|pants|shoes?|belts?|socks|suits?|watch))|sundress(?:es)?|"
    r"(?<!dressing )gowns?|skirts?|blouses?|bras?|bralettes?|bikinis?|tankinis?|monokinis?|"
    r"lingerie|panties|camisoles?|lehengas?|sarees?|saris?|anarkalis?|kurtis?|salwars?|"
    r"dupattas?|nightgowns?|babydolls?|ballet flats?|ballet|stilettos?|high heels|"
    r"tennis bracelets?|handbags?|purses?|clutch(?:es)?|wristlets?)\b",
    "dress", "gown", "skirt", "blouse", "bra", "bikini", "tankini", "monokini", "lingerie",
    "panties", "camisole", "lehenga", "saree", "sari", "anarkali", "kurti", "salwar",
    "dupatta", "babydoll", "ballet", "stiletto", "high heels", "tennis bracelet", "handbag",
    "purse", "clutch", "wristlet")
_MEN_CONVENTION = _rule(
    r"\b(?:neckties?|bow ?ties?|cufflinks?|cuff links|tie (?:clips?|bars?)|tuxedos?|"
    r"boxer briefs|boxers|sherwanis?|groomsmen)\b",
    "tie", "cufflink", "cuff link", "tuxedo", "boxer", "sherwani", "groomsmen")
_BABY = _rule(
    r"\b(?:bab(?:y|ies|y's)(?![- ]?(?:blue|pink|doll|dolls|shower|tee|tees|breath|yoda|bump|"
    r"face))|infants?|infant's|newborns?|preemie)\b|"
    r"\b\d{1,2}\s?(?:-\s?\d{1,2}\s?)?(?:months?|mos)\b",
    "bab", "infant", "newborn", "preemie", "month", "mos")
_KID_SIZE = _rule(
    r"\b(?:toddlers?|toddler's|little kids?|big kids?|(?<!stray )(?<!new )kids?|kid's|kids'|"
    # "baby girl's" is a baby's size like "baby girls" (68 rows read kids, months and all);
    # a plural possessive's "girls" matches before its apostrophe
    r"youth|children|(?<!baby )(?:girls?|boys?|girl's|boy's))\b|"
    r"\b\d{1,2}t\b(?![-'])|\b(?:[1-9]|1[0-6])\s?(?:-\s?\d{1,2}\s?)?(?:years?|yrs?)\b",
    "kid", "girl", "boy", "year", "yr", "toddler", "youth", "children", *_DIGIT_T)
# A couple's or an adult pair's item filed under Kids ("His and Hers Matching Couple
# T-Shirts - I Love This Guy and Girl", 2026-10-19 review): the weak "girl" and the
# kids fallback made it a kids' row, so the default adult ask sent a graded-perfect tee
# from #19 to #119 and the reranker read 'audience=kids'. Read only on Kids-tagged titles
# with no strong kid cue. Not "guy" ("Tuff Guys Boys Sleepwear" is a brand)
_ADULT_PAIR = _rule(
    r"\b(?:his (?:and|&|n) hers|hers (?:and|&|n) his|couples?|couple's|"
    r"husband (?:and|&) wife|mr\.? (?:and|&) mrs\.?|boyfriend (?:and|&) girlfriend)\b",
    "his", "hers", "couple", "husband", "mr", "boyfriend")
_GENDER_TAGS = {"women": "women", "men": "men", "kids": "kids", "unisex": "unisex"}


def _has(rule, t: str) -> bool:
    return _find(rule, t) is not None


def _gender_tag(gender, category) -> Optional[str]:
    """The stored tag (women / men / kids / unisex), else read from the category bucket."""
    if isinstance(gender, str):
        tag = _GENDER_TAGS.get(gender.strip().casefold())
        if tag:
            return tag
    if isinstance(category, str):
        c = category.casefold()
        if "women" in c or "ladies" in c:
            return "women"
        if re.search(r"\bmen", c):
            return "men"
        if re.search(r"kid|girl|boy|baby", c):
            return "kids"
        if "unisex" in c:
            return "unisex"
    return None


def _shopper_gender(tag, t: str) -> str:
    """shopper_gender from a stored tag and a normalised title."""
    kid = _has(_KID_STRONG, t)
    if tag == "kids":
        # an adult garment the department filed under Kids ("OutTop ... coats for women")
        if kid:
            return "kids"
        pair = _has(_ADULT_PAIR, t)
        if not pair and _has(_KID_WEAK, t):
            return "kids"
        women, men = _has(_WOMEN_WORD, t), _has(_MEN_WORD, t)
        if women or men:
            return "unisex" if women and men else "women" if women else "men"
        return "unisex" if pair else "kids"
    if tag in ("women", "men"):
        # a children's item filed under an adult department ("girl's ... skater dress 12
        # years"), not "earrings for women girls" (an adult word keeps the tag)
        if kid and not (_has(_WOMEN_WORD, t) or _has(_MEN_WORD, t) or _has(_ADULT, t)):
            return "kids"
        return tag
    # the "unisex" fallback (or no tag): the title decides
    women, men = _has(_WOMEN_WORD, t), _has(_MEN_WORD, t)
    if kid and not (women or men or _has(_ADULT, t)):
        return "kids"
    if women != men:
        return "women" if women else "men"
    if not women:
        women_conv, men_conv = _has(_WOMEN_CONVENTION, t), _has(_MEN_CONVENTION, t)
        if women_conv != men_conv:
            return "women" if women_conv else "men"
    return "unisex"


def _audience(shopper_gender: str, t: str) -> str:
    """adult, kids or baby. A toddler is kids; teen and junior sizes are adult."""
    if shopper_gender != "kids":
        return "adult"
    return "baby" if _has(_BABY, t) and not _has(_KID_SIZE, t) else "kids"


def shopper_profile(title, gender=None, category=None) -> Tuple[str, str]:
    """(shopper_gender, audience) of one Fashion title, as annotate() derives them from
    the row's `gender` tag and `categ_lvl2_name` bucket."""
    t = _normalise(title)
    found = _shopper_gender(_gender_tag(gender, category), t)
    return found, _audience(found, t)


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
    """Add the derived columns IN PLACE and return the same frame:

    - `product_type` / `product_subtype`: classify_title(Product_title);
    - `fine_subtype`: a FINE_SUBTYPE_LABELS key for apparel and footwear, else None;
    - `shopper_gender`: women / men / kids / unisex (the `gender` tag, else the
      `categ_lvl2_name` bucket, re-read from the title where it is the "unisex"
      fallback or contradicts the title);
    - `audience`: adult / kids / baby;
    - `color_family`: a COLOR_FAMILIES key from `color` -- on EVERY vertical, not only
      Fashion (None where the colour is missing or no family).

    The first five are Fashion-only: every other row (and every row of a frame without
    the vertical or title column) gets None. Every column is object dtype holding str
    or None, so a row dict reads None, never NaN. No existing column is modified: the
    embedded fields (app/search.py) and therefore search_text stay byte-identical.

    Never filters, concatenates, resets or reorders: hydration and GET /product use
    the ROW POSITION as catalog_index, so a reordered frame would serve the wrong
    products. Idempotent (the columns are recomputed from the source columns each
    call). Distinct normalised titles and colour values are read once and mapped back."""
    n_rows, index = len(df), df.index
    types, subtypes, fines = [None] * n_rows, [None] * n_rows, [None] * n_rows
    genders, audiences, colors = [None] * n_rows, [None] * n_rows, [None] * n_rows
    columns = df.columns
    if "bsns_vrtcl_name" in columns and "Product_title" in columns:
        titles = df["Product_title"].tolist()
        tags = df["gender"].tolist() if "gender" in columns else [None] * n_rows
        cats = df["categ_lvl2_name"].tolist() if "categ_lvl2_name" in columns else [None] * n_rows
        memo, tag_memo = {}, {}
        for i, vertical in enumerate(df["bsns_vrtcl_name"].tolist()):
            if not is_fashion(vertical):
                continue
            key = _normalise(titles[i])
            found = memo.get(key)
            if found is None:
                type_, subtype = _classify_text(key)
                found = memo[key] = (type_, subtype, _fine_subtype(key, type_))
            types[i], subtypes[i], fines[i] = found
            # NaN / pd.NA read as absent (a CSV's missing tag, a Databricks None)
            raw = (tags[i] if isinstance(tags[i], str) else None,
                   cats[i] if isinstance(cats[i], str) else None)
            tag = tag_memo.get(raw, "")
            if tag == "":
                tag = tag_memo[raw] = _gender_tag(*raw)
            genders[i] = gender = _shopper_gender(tag, key)
            audiences[i] = _audience(gender, key)
    if "color" in columns:
        color_memo = {}
        for i, raw in enumerate(df["color"].tolist()):
            if not isinstance(raw, str):
                continue
            family = color_memo.get(raw, "")
            if family == "":
                family = color_memo[raw] = color_family(raw)
            colors[i] = family
    for name, values in ((TYPE_COLUMN, types), (SUBTYPE_COLUMN, subtypes),
                         (FINE_SUBTYPE_COLUMN, fines), (GENDER_COLUMN, genders),
                         (AUDIENCE_COLUMN, audiences), (COLOR_FAMILY_COLUMN, colors)):
        df[name] = pd.Series(values, index=index, dtype=object)
    if len(df) != n_rows or not df.index.equals(index):
        raise RuntimeError("product_type.annotate changed the catalog's rows or index")
    return df


def annotation_counts(df: pd.DataFrame) -> dict:
    """{column: {value: rows}} for the columns annotate() adds beyond the type, Fashion
    rows only for the Fashion-only ones (for logs and the slow test). Values only."""
    out = {}
    fashion = df[TYPE_COLUMN].notna() if TYPE_COLUMN in df.columns else None
    for name in (FINE_SUBTYPE_COLUMN, GENDER_COLUMN, AUDIENCE_COLUMN, COLOR_FAMILY_COLUMN):
        if name not in df.columns:
            continue
        every_row = name == COLOR_FAMILY_COLUMN or fashion is None
        values = df[name] if every_row else df.loc[fashion, name]
        counts = values.fillna("none").value_counts()
        out[name] = {str(k): int(v) for k, v in counts.items()}
    return out


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
# 'outfit'/'look' name no TYPE, so _tokens() reads them out of the head's nouns ("earrings
# to complete my outfit" asks for earrings). A head that names nothing BUT an outfit is a
# clothing ask ("beach wedding outfit" -> apparel); 'look' never is ("a gift that looks
# expensive"). An outfit bundle (OUTFIT_BUNDLES_ENABLED) never reads requested_types.
_OUTFIT_WORDS_RE = re.compile(r"\b(outfits?|looks?)\b")
_OUTFIT_ASK_RE = re.compile(r"\boutfits?\b")
# 'accessorize this dress' asks for accessories FOR the dress, not the dress
_ACCESSORISE_RE = re.compile(r"\baccessori[sz](?:e|es|ing)\b")
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
    "occasion function reception celebration night evening wear".split())
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
    if head == "jewelry" and _watch_or_band(" ".join(tokens[:at + 1])):
        # The rows' watch rule, read on the ask: a jewellery noun that names a watch or its
        # band ("apple watch bracelet", "pocket watch necklace") asks for what the titles
        # type accessory. Read as jewelry, every watch band ranked below every plain
        # bracelet (2026-10-30 review); "watch charm", "bracelet with watch charm" keep it
        head = "accessory"
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
    found, after_modifier, outfit = set(), False, False
    for i, (joiner, part, type_) in enumerate(read):
        if type_:
            found |= ACCESSORY_FAMILY if type_ == _FAMILY else {type_}
            after_modifier = False
        elif many:
            if _names_untyped_product(part, joiner, i < last_typed, after_modifier):
                return None
            after_modifier = True
            outfit = outfit or bool(_OUTFIT_ASK_RE.search(part))
    if outfit and found:
        # An outfit named NEXT TO a type is more products, in either order ("outfit and
        # shoes for a wedding" like "shoes and an outfit"): the shoes alone would make
        # every garment a violator. Alone or with modifiers it is apparel (the caller).
        return None
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
    headlamp", "gym shorts and an apple watch band") makes it None. A head that names
    nothing but an outfit is apparel ("hiking outfit"), an outfit next to a type is None
    ("outfit and shoes"); an accessorise verb in the head is the accessory family
    ("accessorize this dress") unless a type precedes it. Purely lexical, so
    a lone Electronics/Grocery noun collides ("tea bags" -> bag, "ring light" ->
    jewelry, "apple watch" -> accessory). app/main.py must only act on the result
    inside a Fashion-majority candidate pool, which neutralises those."""
    whole = _normalise_query(query)
    # From 1, so a query that opens with a connector ("for men: dress shoes") keeps it
    # in the head instead of leaving the head empty
    match = _CONNECTOR_RE.search(whole, 1)
    head = whole[:match.start()] if match else whole
    verb = _ACCESSORISE_RE.search(head)
    if verb:
        # The dress named after the verb is what the accessories are FOR; a type named
        # before it is the ask ("boots to accessorize my outfit" -> footwear)
        before = head[:verb.start()]
        found = _segment_types(before, whole) if before.strip() else set()
        if found is None:
            return None
        return frozenset(found) if found else ACCESSORY_FAMILY
    found = _segment_types(head, whole) if head.strip() else set()
    if found is None:
        return None
    if not found and _OUTFIT_ASK_RE.search(head):
        # With outfit bundles off an outfit ask is a flat page: rank its garments first
        found = {"apparel"}
    if not found and match and _is_clothing_ask(whole, whole):
        found = {"apparel"}
    if not found and _names_occasion_attire(head, whole):
        found = {"apparel"}
    if found == {"jewelry"} and match and _jewelry_for_a_watch(head, whole, match):
        found = {"accessory"}
    return frozenset(found) if found else None


# After 'for', a determiner may open the watch's name ("for MY apple watch"); a relative
# word makes the phrase a person instead ("for a man WHO collects watches")
_RELATIVE_WORDS = frozenset("who whom whose that which".split())


def _jewelry_for_a_watch(head: str, whole: str, match) -> bool:
    """'bracelet for apple watch', 'beaded bracelet for my fitbit charge 5': a jewellery
    head FOR a watch is the band the rows type accessory (their title reading: a target
    after 'for'). Only when the phrase after 'for' IS the watch -- it ends on it, sizes and
    model words aside -- and the title reading of the ask says watch or band: "bracelet
    for my watch-loving dad", "bracelet for a man who collects watches" and "earrings for
    apple watch" keep jewelry. One jewelry head only; a multi-item ask keeps its reading."""
    if match.group() != " for ":
        return False
    after = _CONNECTOR_RE.search(whole, match.end() - 1)
    target = whole[match.end():after.start() if after else len(whole)].split()
    while target and _bare(target[0]) in _DETERMINERS:
        target = target[1:]
    if not 1 <= len(target) <= 5 or any(_bare(t) in _RELATIVE_WORDS for t in target):
        return False
    phrase = " ".join(target)
    return (catalog_taxonomy.ends_on_a_watch(phrase)
            and catalog_taxonomy.watch_kind_text(f"{head} for {phrase}") in ("watch", "band"))


# A style or dress code names clothing although no garment noun does ("black tie event
# men", "business casual for men", "gift for my boyfriend who likes streetwear"): these
# came back None, so the untyped window gave 10-13 of 15 seats to one intent (a bow tie,
# cargo pants). Anywhere in the query, like a clothing noun. A bare "black tie" is still
# the tie; "wear leveling" (an SSD) is no style.
_STYLE_WEAR_RE = re.compile(
    r"\b(?:streetwear|street wear|menswear|womenswear|eveningwear|evening wear|partywear|"
    r"party wear|formalwear|formal wear|resortwear|resort wear|festival wear|casualwear|"
    r"casual wear|business (?:casual|formal|professional|attire)|smart casual|"
    r"cocktail (?:attire|wear)|semi[- ]formal (?:attire|wear)|dress code|"
    r"(?:black|white)[- ]tie (?:" + "|".join(sorted(_AFTER_DRESS_CODE)) + r"))\b")
# A wedding-party or occasion role names its clothing ("bridesmaid", "mother of the
# bride", "prom", "groomsmen") -- in the HEAD only, and not when the head asks for a
# gift or a prop for them ("bridesmaid proposal box", "groomsmen gifts"; "gift for a
# bridesmaid" has the role after the connector).
_OCCASION_ROLE_RE = re.compile(
    r"\b(?:bridesmaids?|maids? of honou?r|matrons? of honou?r|(?:mother|father) of the "
    r"(?:bride|groom)|prom|homecoming|quincea[nñ]eras?|groomsm[ae]n|flower girls?)\b")
_NOT_ATTIRE_RE = re.compile(
    r"\b(?:gifts?|presents?|proposals?|box(?:es)?|favou?rs?|cards?|ideas?|decor\w*|games?|"
    r"signs?|invitations?|baskets?|petals?|wands?|mums?|pillows?|cups?|tumblers?|mugs?)\b")
# "something warm to wear" (also "something warm for a ski trip": the core garment of a
# cold outing is clothing, its gloves and beanie come after)
_SOMETHING_WARM_RE = re.compile(
    r"\bsomething (?:warm|warmer|cozy|cosy)\b(?= (?:to wear|for (?:a |an |the |my |our )?"
    r"(?:ski\w*|snowboard\w*|hik\w*|camp\w*|winter|cold|snow\w*|trip|travel\w*|run\w*|walk\w*|"
    r"hunt\w*|fishing|football|game|outdoor\w*)\b))")


def _names_occasion_attire(head: str, whole: str) -> bool:
    if _STYLE_WEAR_RE.search(whole) or _SOMETHING_WARM_RE.search(whole):
        return True
    return bool(_OCCASION_ROLE_RE.search(head)) and not _NOT_ATTIRE_RE.search(head)


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


# ------------------------------------------------------------------ fit, style, audience asks
# Readers of the shopper's words for the window and the rubric (2026-10-19): the fit
# range ("petite work pants"), the shop category ("jeans", not leggings), who it is for
# ("toddler girl party dress"), and whether the shopper is a wedding guest. Lexical and
# English-only like requested_types, so they collide outside Fashion ("iphone 14 plus",
# "tall kitchen trash bags", "baby formula"): app/main.py gates them exactly like the
# type request (a Fashion-majority pool). Each returns None when the query says nothing.
def _ask_head(query) -> str:
    """The part of the query that names the product: before the first connector, and
    before a wear verb or an accessorise verb (see requested_types)."""
    whole = _normalise_query(query)
    match = _CONNECTOR_RE.search(whole, 1)
    head = whole[:match.start()] if match else whole
    for verb_re in (_WEAR_ASK_RE, _ACCESSORISE_RE):
        verb = verb_re.search(head)
        if verb:
            head = head[:verb.start()]
    return head + " "


_FIT_RES = (
    # not "petite earrings" (small jewellery)
    ("petite", re.compile(
        r"\bpetites?\b(?![- ](?:necklaces?|earrings?|pendants?|bracelets?|rings?|studs?|"
        r"charms?|jewel\w*|watch\w*)\b)")),
    ("maternity", re.compile(
        r"\b(?:maternity|pregnan\w*|moms?[- ]to[- ]be|expecting (?:moms?|mothers?)|"
        r"baby bump|bump[- ]friendly|post[- ]?partum|breastfeeding|nursing (?:bras?|tops?|"
        r"tanks?|camis?|dress(?:es)?|gowns?|pajamas?|pjs|sleepwear|nightgowns?|shirts?|"
        r"hoodies?|sweaters?|clothes|clothing|wear|friendly))\b")),
    # "plus" alone is a phone model ("iphone 14 plus") or a conjunction
    ("plus", re.compile(r"\b(?:plus[- ]?sizes?d?|plus[- ]sized|curvy)\b")),
    # not "tall boots" / "knee tall socks" (a shaft height) or a tall vase
    ("tall", re.compile(
        r"\bbig (?:and|&|n) tall\b|"
        r"(?<!knee )(?<!knee-)(?<!thigh )(?<!thigh-)(?<!high )\btall\b(?![- ]?(?:boots?|booties|"
        r"socks?|shafts?|heels?|hats?|vases?|glass(?:es)?|cans?|bags?|kitchen|trash|grass|"
        r"candles?|tumblers?|cups?|plants?|trees?|dressers?|lamps?|shelf|shelves|cabinets?|"
        r"bottles?)\b)")),
)


def requested_fit(query) -> Optional[FrozenSet[str]]:
    """The fit ranges the shopper named, a subset of FITS, or None: "petite",
    "maternity" (also pregnancy, nursing tops, postpartum), "plus" (needs "plus size"
    or "curvy"), "tall" (also "big and tall")."""
    whole = _normalise_query(query)
    found = frozenset(fit for fit, pattern in _FIT_RES if pattern.search(whole))
    return found or None


_MEN_ASK_RE = re.compile(
    r"\b(?:men|men's|mens|man|male|gentlemen|gents|boys?|boy's|husband|boyfriend|dad|father|"
    r"him|his|he|groom|groomsmen)\b")


def implies_women(query) -> bool:
    """Whether a fit word states women's wear ("petite work pants", "maternity dress").
    Fit words only: app/main.py keeps its own women's-garment and people cues."""
    fit = requested_fit(query)
    if not fit or not fit & {"petite", "maternity"}:
        return False
    return not _MEN_ASK_RE.search(_normalise_query(query))


# Compounds first (each consumes its words), then single nouns, where a noun directly
# followed by another is a modifier ("sweater dress", "jean jacket", "boot socks").
# A two-word garment reads as the title rules read it (fine_subtype): "track suit" is a
# set, "night gown" sleepwear, not the bare noun's tailoring or dress -- else the window
# seated blazers ahead of every swimsuit on "swim suit" (2026-10-19 review).
_ASK_COMPOUNDS = tuple((re.compile(pattern), frozenset(keys)) for pattern, keys in (
    (r"\b(?:suit|sport|dinner|tuxedo|tux) (?:jackets?|coats?)\b|\bsport ?coats?\b|"
     r"\b(?:suit|tuxedo|tux) vests?\b", {"suits_blazers"}),
    (r"\b(?:track|sweat|snow|jump|body|romper|" + _ATHLETIC_SUIT_WORDS + r") ?suits?\b",
     {"jumpsuits_sets"}),
    (r"\b(?:wet|rain|space|flight|boiler|birthday|diving|dive) ?suits?\b", ()),  # none
    (r"\b(?:pajama|pyjama|lounge|sleep) (?:pants|shorts|sets?|tops?|shirts?|dress(?:es)?)\b|"
     r"\bnight ?gowns?\b|\bhouse ?dress(?:es)?\b|\bdressing gowns?\b", {"sleep_lounge"}),
    (r"\byoga pants\b", {"leggings", "pants"}),
    (r"\bswim ?suits?\b|\b(?:swim|board) (?:shorts|trunks|tops?|dress(?:es)?|skirts?|briefs)\b|"
     r"\bbikini (?:tops?|bottoms?)\b|\bbathing suits?\b|\bcover[- ]?ups?\b|\brash guards?\b",
     {"swim"}),
    (r"\bboxer (?:shorts|briefs)\b|\bsports? bras?\b", {"lingerie"}),
    (r"\bdress shirts?\b|\bbutton[- ]?(?:down|up)s?(?: shirts?)?\b|"
     r"\b(?:oxford|flannel|hawaiian|aloha|western|camp|tuxedo) shirts?\b", {"blouses_shirts"}),
    (r"\b(?:tank|crop|tube|halter) tops?\b|\bpolo shirts?\b|\bgraphic (?:tees?|t-?shirts?)\b",
     {"tops_tees"}),
    (r"\bsweater vests?\b", {"sweaters"}),
    (r"\bdress (?:pants|slacks|trousers)\b", {"pants"}),
    (r"\b(?:denim|jean) (?:jackets?|vests?)\b|\btrench coats?\b", {"coats_jackets"}),
    (r"\b(?:denim|jean) shorts\b", {"shorts"}),
    (r"\b(?:denim|jean) skirts?\b", {"skirts"}),
    (r"\b(?:running|trail|basketball|training|cross[- ]?training|golf|soccer|football|"
     r"volleyball|track|hiking|gym|workout|athletic) (?:shoes?|sneakers?|trainers?|cleats?)\b|"
     r"\bcross[- ]?trainers?\b|\bcleats?\b", {"athletic"}),
    (r"\b(?:tennis|walking) (?:shoes?|sneakers?)\b", {"athletic", "sneakers"}),
    (r"\b(?:boat|deck|driving) shoes?\b|\bballet flats?\b|\bmary janes?\b", {"flats"}),
    (r"\bhouse shoes?\b", {"slippers"}),
    (r"\bflip[- ]?flops?\b", {"sandals"}),
    (r"\bhigh[- ]?heels?\b", {"heels"}),
    (r"\b(?:dress|water|work) shoes?\b", ()),        # no single shop category
))
_ASK_WORDS = {}
for _key, _words in (
        ("dresses", "dress dresses sundress sundresses gown gowns maxi maxis midi kaftan "
                    "kaftans caftan caftans"),
        ("tops_tees", "t-shirt t-shirts tshirt tshirts tee tees tank tanks cami camis "
                      "camisole camisoles polo polos henley henleys jersey jerseys"),
        ("blouses_shirts", "blouse blouses tunic tunics flannel flannels"),
        ("sweaters", "sweater sweaters cardigan cardigans pullover pullovers jumper jumpers "
                     "turtleneck turtlenecks shrug shrugs"),
        ("hoodies", "hoodie hoodies hoody sweatshirt sweatshirts"),
        ("coats_jackets", "coat coats jacket jackets parka parkas vest vests raincoat "
                          "raincoats windbreaker windbreakers puffer puffers anorak anoraks "
                          "poncho ponchos gilet gilets overcoat overcoats peacoat peacoats "
                          "shacket shackets outerwear"),
        ("suits_blazers", "suit suits tuxedo tuxedos tux tuxes blazer blazers waistcoat "
                          "waistcoats"),
        ("jeans", "jeans jean jeggings"),
        ("pants", "pants pant trousers trouser chinos chino joggers jogger slacks khakis "
                  "sweatpants capris culottes palazzo palazzos cargos"),
        ("shorts", "shorts bermudas"),
        ("leggings", "leggings legging"),
        ("skirts", "skirt skirts skort skorts tutu tutus"),
        ("jumpsuits_sets", "jumpsuit jumpsuits romper rompers playsuit playsuits overalls "
                           "onesie onesies bodysuit bodysuits tracksuit tracksuits sweatsuit "
                           "sweatsuits snowsuit snowsuits co-ord co-ords coord coords"),
        ("swim", "swimsuit swimsuits swimwear bikini bikinis tankini tankinis monokini "
                 "monokinis trunks coverup coverups sarong sarongs beachwear burkini"),
        ("sleep_lounge", "pajamas pajama pyjamas pyjama pjs sleepwear nightgown nightgowns "
                         "nightie nighties robe robes bathrobe bathrobes loungewear "
                         "nightshirt nightshirts"),
        ("lingerie", "bra bras bralette bralettes lingerie underwear panties boxers briefs "
                     "thong thongs shapewear corset corsets bustier bustiers"),
        ("socks_hosiery", "sock socks tights stockings pantyhose hosiery"),
        ("ethnic_wear", "saree sarees sari saris lehenga lehengas kurta kurtas kurti kurtis "
                        "anarkali anarkalis salwar salwars sherwani sherwanis dupatta "
                        "dupattas churidar churidars abaya abayas qipao cheongsam cheongsams "
                        "hanbok hanboks dashiki dashikis choli cholis"),
        ("boots", "boot boots bootie booties wellies wellingtons chukka chukkas"),
        ("sandals", "sandal sandals slides gladiators"),
        ("heels", "heels pumps stiletto stilettos wedges slingback slingbacks"),
        ("flats", "flats loafer loafers moccasin moccasins oxfords brogue brogues derbies "
                  "espadrille espadrilles"),
        ("slippers", "slipper slippers"),
):
    for _word in _words.split():
        _ASK_WORDS[_word] = frozenset({_key})
for _word in ("shirt", "shirts", "top", "tops"):
    _ASK_WORDS[_word] = frozenset({"tops_tees", "blouses_shirts"})
for _word in ("sneaker", "sneakers", "trainers"):
    _ASK_WORDS[_word] = frozenset({"sneakers", "athletic"})
_ASK_WORD_RE = re.compile(
    r"(?<![\w-])(" + "|".join(sorted((re.escape(w) for w in _ASK_WORDS), key=len, reverse=True))
    + r")(?![\w-])")
# 'dress' as a verb ("how to dress for", "what should I dress my toddler in", "dress up")
# or in "dress code" is no garment
_DRESS_VERB_RE = re.compile(
    r"\b(?:to|i|we|you|they|he|she|should|shall|can|could|would|will|must|do|does|did|how)"
    r" dress\b|\bdress (?:up|down|code|codes)\b")
# 'dress' modifying the noun after it is no dress, as in titles (_DRESS_MODIFIED): "dress
# watch for men" and "dress belt" read the dresses subtype and seated dresses first
# (2026-10-30 R2). Read after the compounds, so "dress shirt" and "dress pants" keep
# their own subtypes; "dress suit" and "dress pumps" read the garment after it as before
_DRESS_MODIFIER_ASK_RE = re.compile(r"\bdress(?:es)?(?=[\s-]*" + _DRESS_MODIFIED + r")")
# What follows is what the shopper does NOT want. Not a bare "no": "no show socks" and
# "no iron shirts" are styles.
_NEGATION_RE = re.compile(r"\b(?:not|without|except|instead of|rather than|other than)\b")


def requested_subtypes(query) -> Optional[FrozenSet[str]]:
    """The FINE_SUBTYPE_LABELS keys the shopper named, read from the product part of
    the query like requested_types (a noun after the connector is context: "earrings
    for a wedding dress" names no subtype; one after a negation is excluded: "jeans,
    not leggings"). "shirt" / "top" accept tees and blouses, "sneakers" accepts
    athletic shoes; "shoes", "clothes" or "outfit" name none."""
    head = _ask_head(query)
    negation = _NEGATION_RE.search(head)
    if negation:
        head = head[:negation.start()] + " "
    head = _DRESS_VERB_RE.sub(" ", head)
    found = set()
    for pattern, keys in _ASK_COMPOUNDS:
        for match in pattern.finditer(head):
            found |= keys
        head = pattern.sub(" | ", head)   # a blank no noun can modify across
    head = _DRESS_MODIFIER_ASK_RE.sub(" ", head)
    nouns = list(_ASK_WORD_RE.finditer(head))
    for i, noun in enumerate(nouns):
        nxt = nouns[i + 1] if i + 1 < len(nouns) else None
        if nxt is not None and head[noun.end():nxt.start()].strip(" -") == "":
            continue                      # a modifier of the next noun
        found |= _ASK_WORDS[noun.group(1)]
    whole = _normalise_query(query)
    if not found and _DRESS_CODE_TIE_RE.search(whole) and \
            requested_types(query) == frozenset({"apparel"}):
        men = _MEN_ASK_RE.search(whole) and not _WOMEN_ASK_RE.search(whole)
        return _MEN_FORMAL_SUBTYPES if men else _FORMAL_SUBTYPES
    return frozenset(found) or None


# A black-tie or white-tie dress code names its garments although no garment noun does
# (2026-10-19 review, "black tie event men"): with the apparel request alone the window's
# round-robin went 7-8 rows deep into each tuxedo intent's weak retrieval -- a
# windbreaker, a hi-vis jacket, an Arc'teryx shell, a hair-cutting cape -- and the tail
# after it opened on soft-shell and moto jackets. As named subtypes the tuxedos, dress
# shirts, trousers and gowns come first in each tier; the rest of the clothing follows.
# A men's ask leaves out the gowns (a mother-of-the-bride dress and a tutu underskirt from
# the trousers intent would lead the tail of "black tie event men").
_FORMAL_SUBTYPES = frozenset(
    {"suits_blazers", "blouses_shirts", "pants", "dresses", "jumpsuits_sets", "skirts"})
_MEN_FORMAL_SUBTYPES = frozenset({"suits_blazers", "blouses_shirts", "pants"})
_WOMEN_ASK_RE = re.compile(
    r"\b(?:women|women's|womens|woman|ladies|lady|her|hers|she|wife|girlfriend|mom|mother|"
    r"girls?|bride)\b")
_DRESS_CODE_TIE_RE = re.compile(
    r"\b(?:black|white)[- ]tie (?:" + "|".join(sorted(_AFTER_DRESS_CODE)) + r"|attire|"
    r"outfits?|looks?|clothes|clothing|dress code)\b|"
    r"\b(?:wear|wearing|dress|dressed)\b(?: \S+){0,3}? (?:to|for) (?:an? |the )?"
    r"(?:black|white)[- ]tie\b")


# Who the item is for. Toddlers wear kids' and baby sizes; a teen wears adult or
# kids' sizes; a person's role ("my wife", "groomsmen") or a fit word is an adult.
_NUMBER_WORDS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
    "fifteen sixteen seventeen eighteen nineteen twenty".split())}
_AGE_RE = re.compile(
    r"\b(\d{1,2}|" + "|".join(_NUMBER_WORDS) + r")[- ]?(?:(years?|yrs?|months?|mos?)[- ]?olds?\b|"
    # "my 8 yo", "8yo", "8 y/o", "8 y.o." (not a yo-yo)
    r"(yo|y/o|y\.o)(?![\w/-]))")
# A child's occasion, school grade or wedding-party role names a child although no age
# word does (2026-10-19 review): "first communion dress", "ring bearer outfit" and "back
# to school outfits for 3rd grade" read None and "junior bridesmaid dress" adult, so the
# default adult audience (app/main.py) sent every kids' row of a 15-of-15 kids head out
# of the rerank window behind adult club dresses and bridal gowns
_CHILD_OCCASION_RE = re.compile(
    r"\b(?:communions?|ring bearers?|junior bridesmaids?|flower girls?|kindergarten\w*|"
    r"pre-?k|preschool\w*|elementary school\w*|grade school\w*|middle school\w*|"
    r"(?:[1-8](?:st|nd|rd|th)|first|second|third|fourth|fifth|sixth|seventh|eighth)"
    r"[- ]grade(?:rs?)?)\b(?![- ](?:teachers?|teacher's|teaching|staff)\b)")
# Christening and baptism gowns are a baby's (or a young child's)
_INFANT_OCCASION_RE = re.compile(r"\b(?:christening|baptism|baptismal|baptizing)\b")
# The bridesmaid of "junior bridesmaid" is no adult cue
_JUNIOR_ROLE_RE = re.compile(r"\bjunior bridesmaids?\b")
_KIDS_ASK_RE = re.compile(
    r"\b(?:(?<!stray )(?<!new )kids?|kid's|kids'|children|children's|childrens|child|child's|"
    r"(?<!golden )(?<!spice )(?<!gilmore )(?<!mean )(?<!powerpuff )(?<!cover )(?<!it )"
    r"(?<!material )(?<!pin up )(?<!pin-up )(?:girls?|girl's|girls')"
    r"(?![- ](?:night|nights|trip|trips|weekend|power|boss|gang)\b)|"
    r"(?<!the )(?<!backstreet )(?<!beastie )(?:boys?|boy's|boys')"
    r"(?![- ](?:night|nights|shorts?|cut)\b)|"
    r"(?:sons?|daughters?)(?!'s (?:wedding|graduation|engagement|rehearsal|college))|"
    r"grand(?:children|child|kids?|sons?|daughters?)|nieces?|nephews?|youth|tweens?|"
    r"preschoolers?|little ones?)\b")
_BABY_ASK_RE = re.compile(
    r"\b(?:bab(?:y|ies|y's)(?![- ]?(?:blue|pink|doll|dolls|shower|tee|tees|breath|yoda|bump|"
    r"face|shark|boomer))|infants?|newborns?|preemies?)\b")
_TODDLER_ASK_RE = re.compile(r"\b(?:toddlers?|toddler's)\b|\b\d{1,2}t\b(?![-'])")
_TEEN_ASK_RE = re.compile(r"\b(?:teens?|teen's|teenagers?|teenage|juniors)\b")
_ADULT_ASK_RE = re.compile(
    r"\b(?:adults?|grown[- ]ups?|women|women's|womens|(?<!wonder )(?<!spider-)(?<!cat )woman|"
    r"ladies|lady|men|men's|mens|"
    r"gentlemen|gents|wife|husband|hubby|boyfriend|girlfriend|moms?|mother|mama|dad|father|"
    r"grandma|grandmother|grandpa|grandfather|fiance|fiancee|bride|groom|bridesmaids?|"
    r"groomsm[ae]n|coworkers?|petite|maternity|pregnant|plus[- ]?size)\b|"
    r"\b" + _MAN_COMPOUND + r"man\b")


def requested_audience(query) -> Optional[FrozenSet[str]]:
    """Who the shopper is buying for, a subset of AUDIENCES, or None: kids, girls, boys,
    a son, a "7 year old" -> kids; baby, infant, newborn, "6 month old" -> baby; a
    toddler -> kids and baby; a teen -> adult and kids; an adult cue (women's, men's,
    "for my wife", petite) -> adult. Several cues add up ("mother daughter dresses")."""
    whole = _normalise_query(query)
    found = set()
    if _KIDS_ASK_RE.search(whole) or _CHILD_OCCASION_RE.search(whole):
        found.add("kids")
    if _BABY_ASK_RE.search(whole):
        found.add("baby")
    if _TODDLER_ASK_RE.search(whole) or _INFANT_OCCASION_RE.search(whole):
        found |= {"kids", "baby"}
    if _TEEN_ASK_RE.search(whole):
        found |= {"adult", "kids"}
    if _ADULT_ASK_RE.search(_JUNIOR_ROLE_RE.sub(" ", whole)):
        found.add("adult")
    for number, years, short in _AGE_RE.findall(whole):
        unit = years or short    # "yo" / "y/o" is years
        n = int(number) if number.isdigit() else _NUMBER_WORDS[number]
        if unit.startswith("mo") or n <= 2:
            found |= {"baby"} if unit.startswith("mo") else {"baby", "kids"}
        elif n <= 12:
            found.add("kids")
        elif n <= 17:
            found |= {"adult", "kids"}
        else:
            found.add("adult")
    return frozenset(found) or None


# The wedding-guest reading (2026-10-15 rubric semantics, made deterministic): a
# shopper dressing for someone else's wedding, party or gala, so bridal gowns,
# bridesmaid dresses and swimwear do not fit. Never when the shopper asks for the
# wedding party's own clothes (bride, bridal, wedding dress / gown / costume, lehenga,
# bridesmaid, groom, groomsmen, "my wedding") -- the reranker had read "wedding costume"
# as a guest ask and capped the gowns. A parent of the couple is a guest.
_PARENT_OF_RE = re.compile(
    r"\b(?:mother|father|mom|dad|parents?|grandmother|grandma|grandfather|grandpa) of the "
    r"(?:bride|groom)\b|\b(?:bride|groom)'s (?:mother|mom|father|dad|grandmother|grandma|"
    r"parents?)\b")
_BRIDAL_PARTY_RE = re.compile(
    r"\b(?:brides?|bride's|bridal|bridegrooms?|bridesmaids?|maids? of honou?r|"
    r"matrons? of honou?r|grooms?|groom's|groomsm[ae]n|lehengas?|lehnga|"
    r"wedding (?:dress(?:es)?|gowns?|costumes?|sarees?|saris?|lehengas?)|"
    r"(?:my|our) (?:own )?wedding)\b")
_GUEST_RE = re.compile(
    r"\bwedding guests?\b|\bguests? (?:at|of|for) (?:an? |the |my |our )?(?:[\w'-]+ ){0,2}?"
    r"weddings?\b|\bwedding (?:outfits?|attire|clothes|cloths|clothing|wear|looks?|ensembles?)\b")
_AT_EVENT_RE = re.compile(
    r"\b(?:for|to|at|attend|attending|going to|invited to)\s+"
    r"(?:(?:a|an|the|my|our|his|her|their|this|that|[\w-]+'s)\s+)?(?:[\w'-]+\s+){0,4}?"
    r"(?:weddings?|receptions?|galas?|part(?:y|ies))\b")
_NOT_DRESSY_PARTY_RE = re.compile(
    r"\b(?:pool|beach|swim|pajama|pyjama|costume|halloween|toga|foam|slumber|luau|"
    r"hot tub|bachelorette|bachelor)\s+part(?:y|ies)\b")


def is_wedding_guest_ask(query) -> bool:
    """Whether the shopper is plainly a GUEST dressing for a wedding, party or gala:
    "wedding guest dress", "cloths for a beach wedding in summer", "beach wedding
    outfit", "what to wear to a gala". False for any ask for the wedding party's own
    clothes, for a pool or costume party (swimwear and costumes fit there), and when
    no clothing is asked for ("earrings for a wedding")."""
    whole = _PARENT_OF_RE.sub(" ", _normalise_query(query))
    if _BRIDAL_PARTY_RE.search(whole):
        return False
    if _GUEST_RE.search(whole):
        # A named guest is a clothing ask unless the shopper names only other things:
        # "earrings for a wedding guest", "wedding guest clutch" and "wedding guest
        # shoes" were guests (2026-10-19 review) while "earrings for a wedding" was not.
        # Bare "wedding guest" names no type and stays one
        types = requested_types(query)
        return types is None or "apparel" in types
    event = _AT_EVENT_RE.search(whole)
    if event is None or _NOT_DRESSY_PARTY_RE.search(event.group()):
        return False
    types = requested_types(query)
    return bool(types and "apparel" in types)
