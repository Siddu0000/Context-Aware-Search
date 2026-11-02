"""Device asks: the shopper named a DEVICE itself ("laptop for college", "apple watch",
"gift for a vintage watch collector"), and which rows of a pool ARE that device.

The Electronics / gift counterpart of product_type.requested_types (2026-10-27, the Oct 27
meeting: "gift for a vintage watch collector" showed watch boxes, winders and cleaning
kits on page 1 and the watches on page 2). Nothing in main.py knew "device vs accessory":
a wristwatch and its winder are both Fashion `accessory` rows, the rubric caps "an
accessory FOR the named product", and about 60% of Electronics rows are accessories that
NAME a device ("Case for iPhone 12"). The row side reads app/catalog_taxonomy.py's
load-time columns (`watch_kind`, `item_category`, `item_role`) -- no second classifier.

- requested_device(query) -> DeviceAsk(family, gift, accessory_asked) | None. A closed
  family list (FAMILIES, owner call D2): watch, laptop, headphones, phone, tablet, camera,
  speaker. NOT power bank (a power bank is itself a charger) and NOT tracker / AirTag (0
  AirTags in the catalog: the rubric and the weak-match banner handle that supply gap).
  A gift persona ("gift for a <X> collector / lover / fan", "... who loves <X>") names X,
  gift=True (owner call D1: X itself is the gift). The family noun must be the HEAD of
  the ask (2026-10-27 review): an accessory noun after it ("watch winder", "laptop
  sleeve", "apple watch band", "camera bag") asks for the accessory (accessory_asked=True,
  main.py does nothing); any other word after it but a model / spec / descriptor word
  ("speaker wire", "headphone pads", "watch cap") and a coordination with anything but
  the same device ("phone or tablet stand", "laptop and mouse") read None, and so does a
  word before it that names another device ("drawing tablet", "cordless phone", "vr
  headset"; 2026-10-30 review). A care / use
  phrase about a device the shopper owns ("stop my laptop from overheating", "keep my
  phone alive", "watch movies ...") is no device ask: None.
- is_core(row, family): the row IS the device, from the load-time columns and the
  title's head (a device category alone also holds cases, books and landlines).
- mentions(row, family): the row's title is about the family (the pool gate).
- role(row, family): CORE / UNKNOWN / ACCESSORY, the window's and the ads' tiers.
- addon_rank(row, family): storage / display (0) before other add-ons (1) before tools,
  care, parts, cables and replacement items (2): the order of a gift's reserved seats.
- constraint(ask): the deterministic [product_type] text the reranker is sent.

Pure re (no pandas, no I/O, no LLM), English-only and hand-maintained like
product_type.py and the query guard; tables in tests/test_device_role.py. Lexical, so
main.py acts on a reading only inside a pool where the family leads the raw head and at
least one core row exists (_effective_device): "apple watch" is a device ask in an
Electronics pool and does nothing where no watch row exists.
"""

import re
from functools import lru_cache
from typing import NamedTuple, Optional

FAMILIES = ("watch", "laptop", "headphones", "phone", "tablet", "camera", "speaker")

# Role tiers of a row under a device ask, best first (main.py partitions on them)
CORE, UNKNOWN, ACCESSORY = 0, 1, 2
ROLE_NAMES = {CORE: "core", UNKNOWN: "unknown", ACCESSORY: "accessory"}


class DeviceAsk(NamedTuple):
    family: str            # one of FAMILIES
    gift: bool             # a gift for someone who collects / loves the family
    accessory_asked: bool  # the shopper asked for an accessory FOR the device


def _norm(text) -> str:
    """Casefolded, curly apostrophe straightened, whitespace collapsed; non-str -> ''."""
    if not isinstance(text, str):
        return ""
    return " ".join(text.casefold().replace("’", "'").split())


# ------------------------------------------------------------------ family nouns
# 'watch' is no noun when an object or a party follows ("watch movies", "watch party",
# "watch the game"); catalog_taxonomy.watch_kind reads titles with its own verb list
_WATCH_NOT_VERB = (r"(?!\s+(?:movies?|films?|tv|television|shows?|videos?|netflix|youtube|"
                   r"sports?|football|games?|the|a|an|my|your|our|over|out|party|parties|"
                   r"list|me|him|her|them|it|this|us|live|online|anywhere)\b)")
_FAMILY_NOUNS = {
    "watch": (r"wrist ?watch(?:es)?|smart ?watch(?:es)?|pocket ?watch(?:es)?|timepieces?|"
              r"chronographs?|(?:fitness|activity) trackers?|watch(?:es)?" + _WATCH_NOT_VERB),
    "laptop": (r"laptops?|notebook (?:computers?|pcs?)|macbooks?(?: (?:air|pro))?|"
               r"chromebooks?|ultrabooks?"),
    "headphones": (r"head ?phones?|head ?sets?|ear ?buds?|ear ?phones?|"
                   r"airpods(?: (?:pro|max))?|iems?"),
    "phone": (r"smart ?phones?|cell ?phones?|mobile phones?|android phones?|iphones?|"
              r"phones?"),
    "tablet": r"tablets?|ipads?(?: (?:pro|air|mini))?|kindles?|e-?readers?",
    "camera": r"cameras?|camcorders?|dslrs?|gopros?|action cams?|dash ?cams?",
    "speaker": r"speakers?|sound ?bars?|soundbars?|boom ?box(?:es)?",
}
FAMILY_RE = {family: re.compile(r"\b(?:" + pattern + r")\b")
             for family, pattern in _FAMILY_NOUNS.items()}

# An accessory noun after the device noun makes the accessory the head ("watch winder",
# "iphone 13 case", "airpods pro case"): English compounds are right-headed
_ACCESSORY_NOUN_RE = re.compile(
    r"\b(?:bands?|straps?|bracelets?|links?|loops?|cases?|covers?|skins?|shells?|bumpers?|"
    r"sleeves?|bags?|backpacks?|pouch(?:es)?|holders?|stands?|mounts?|docks?|cradles?|"
    r"chargers?|charging|cables?|cords?|adapters?|batter(?:y|ies)|power banks?|"
    r"protectors?|tempered glass|films?|box(?:es)?|winders?|displays?|organi[sz]ers?|"
    r"trays?|rolls?|pillows?|cushions?|storage|tools?|kits?|repair|cleaners?|cleaning|"
    r"cloths?|polish(?:ing)?|parts?|replacements?|ear ?pads?|ear ?tips?|tips|keyboards?|"
    r"mouse|mice|styluses|stylus|pens?|lens(?:es)?|lens caps?|tripods?|gimbals?|grips?|"
    r"lights?|filters?|memory cards?|sd cards?|remotes?|controllers?|decals?|stickers?|"
    r"keychains?|lanyards?|hooks?|locks?|cooling pads?|fans?|hubs?|chains?|fobs?|"
    r"crystals?|bezels?|crowns?|movements?|glass|wallets?)\b")

# A care / use phrase about a device the shopper owns opens the request: a fix or an
# add-on for it, never the device ("stop my laptop from overheating" wants a cooling pad).
# Not 'watch' (the noun pattern refuses the verb) and not 'find' / 'buy' ("find a laptop").
_OWNER_VERB_RE = re.compile(
    r"^(?:(?:how (?:to|do i|can i|should i)|ways to|best way to|what to use to) )?"
    r"(?:stop|keep|fix|repair|protect|charge|recharge|clean|cool|connect|mount|carry|store|"
    r"organi[sz]e|hold|hang|attach|secure|prop|power|upgrade|speed up|boost|extend|replace|"
    r"listen|lose|never|make|help|stream|sync|pair|unlock|reset|backup|back up)\b")
# "my laptop keeps overheating", "her phone", "case for my iphone 14": a possessive right
# before the device noun is the owner talking about their own device
_OWNED_RE = re.compile(r"\b(?:my|our|his|her|their|your)\s+(?:[\w'-]+\s+){0,2}$")
# Where the head phrase ends: the text before the first of these names the product. And /
# or / & / + / slash do not end it, they JOIN its parts ("headphones or earbuds"): every
# part is read (_phrase_reading)
_PHRASE_END_RE = re.compile(
    r" (?:for|with|under|below|less than|that|which|to go with|to match|from|without|on|"
    r"to|in|at|near|by|around|over|so|because|but|while|when|if) ")
_COORDINATOR_RE = re.compile(r" (?:and|or|&|\+|/) ")
# Trailing specs that are not a second noun ("laptop 16gb", "15 inch", "2 pack")
_TRAILING_RE = re.compile(r"(?:\s+(?:\d[\w.$/'\"-]*|inch(?:es)?|in|gb|tb|mm|pack|pcs))+$")
# The only words a device name may carry AFTER its noun (owner call, 2026-10-27 review):
# model / variant words ("iphone 14 pro max", "apple watch series 9", "airpods pro 2"),
# descriptor and spec words of the Amazon-style keyword query ("headphones noise
# cancelling", "watch automatic mens", "iphone unlocked black"; sizes and numbers are
# _TRAILING_RE's) and shopping words ("watch gift", "laptop deals"). Any other word after
# the device noun is the head of a compound -- a part or an add-on FOR the device ("speaker
# wire", "headphone pads", "camera microphone", "laptop desk", "phone ring", "watch face",
# "watch cap", "headphone amp", "camera flash", "laptop ram") -- so the reading fails closed
# (None: the page the shopper had before 2026-10-27), whether or not _ACCESSORY_NOUN_RE
# knows the word. Until the 2026-10-27 review one unknown word was allowed and each of
# those read as the device, seated the device over what was asked for and sent the device
# constraint.
_TAIL_WORDS = frozenset((
    # model / variant
    "pro", "max", "plus", "mini", "ultra", "se", "air", "lite", "series", "gen",
    "generation", "edition", "classic", "active", "sport", "x", "xl", "xr", "xs", "s", "e",
    "fe", "new", "latest", "paperwhite", "oasis", "hero",
    # descriptors and specs
    "unlocked", "renewed", "refurbished", "wireless", "bluetooth", "noise", "cancelling",
    "canceling", "cancellation", "waterproof", "gaming", "smart", "digital", "analog",
    "analogue", "automatic", "mechanical", "vintage",
    "men", "mens", "men's", "women", "womens", "women's", "kids", "kid's", "kids'",
    "black", "white", "silver", "gold", "rose", "pink", "blue", "red", "green", "grey",
    "gray", "brown", "purple", "yellow", "orange", "navy", "beige",
    # a computer named as one ("laptop computer", "notebook pc")
    "computer", "computers", "pc",
    # shopping words
    "gift", "gifts", "idea", "ideas", "deal", "deals", "sale", "cheap", "best"))

# A word BEFORE the family noun that makes it another device (2026-10-30 review): the ask's
# twin of the title rules' _OTHER_DEVICE_RE['tablet'] and _LANDLINE_RE. "drawing tablet",
# "writing tablet for kids", "landline phone", "cordless phone" and "vr headset" read as
# tablet / phone / headphones asks; the gate passed on one stray core row in each pool, so
# Android kids tablets were seated over the LCD writing tablets asked for, a Samsung mobile
# over every landline, gaming headsets over every VR headset -- while the rows themselves
# are other devices (a Wacom is computer_accessories, catalog_taxonomy files VR headsets
# under gaming, is_core refuses landlines). Such an ask reads None: the page of before
# 2026-10-27. Per family: (the noun forms the words modify, the words). Not "paper" ("e-paper
# tablet" is an e-reader, D2's tablet) and not "3d" ("3d surround gaming headset" is audio)
_OTHER_DEVICE_ASK = {
    "tablet": (r"tablets?",
               r"lcd|writing|drawing|graphics?|pen|art|doodle|sketch(?:ing)?|signature|"
               r"digitizer|e-?writer|wacom|huion|xp-?pen|gaomon|ugee|veikk"),
    "phone": (r"phones?",
              r"landlines?|cordless|corded|desk|home|house|office|conference|ip|voip|dect|"
              r"satellite|speaker|rotary|retro|wall|caller id|call center"),
    # only a headSET: "vr headphones" are headphones for a VR headset
    "headphones": (r"head ?sets?",
                   r"vr|virtual reality|augmented reality|mixed reality|ar|oculus|quest"),
}
_OTHER_DEVICE_ASK_RE = {family: (re.compile(noun), re.compile(r"\b(?:" + words + r")\b"))
                        for family, (noun, words) in _OTHER_DEVICE_ASK.items()}

# Who the gift is for: "gift for a vintage watch collector", "present for my dad who loves
# cameras", "watch collector gift". The persona noun list is closed and hand-maintained
_GIFT_RE = re.compile(r"\b(?:gifts?|presents?|stocking stuffers?)\b")
_PERSONA_NOUNS = (r"collectors?|lovers?|enthusiasts?|fans?|addicts?|geeks?|nerds?|"
                  r"aficionados?|buffs?|junkies?|connoisseurs?|fanatics?|obsessives?")
_PERSONA_RE = re.compile(
    r"\bfor (?:a |an |my |the |our |your |his |her )?(?P<x>[\w'-]+(?: [\w'-]+){0,3}?) "
    r"(?:" + _PERSONA_NOUNS + r")\b")
_PERSONA_GIFT_RE = re.compile(
    r"^(?:a |an |the )?(?P<x>[\w'-]+(?: [\w'-]+){0,3}?) (?:" + _PERSONA_NOUNS + r")(?:'s?)? "
    r"(?:gifts?|presents?)\b")
_LIKES_RE = re.compile(
    r"\bwho (?:loves?|likes?|collects?|is into|enjoys?|is obsessed with|is crazy about) "
    r"(?P<x>.+)$")


def _rightmost_family(text: str):
    """(family, match) of the right-most family noun in `text`, or (None, None).
    English compounds are right-headed: 'iphone camera' names a camera."""
    best = None
    for family, rx in FAMILY_RE.items():
        for m in rx.finditer(text):
            if best is None or (m.end(), m.end() - m.start()) > (best[1].end(),
                                                                 best[1].end() - best[1].start()):
                best = (family, m)
    return best if best else (None, None)


def _head_reading(head: str) -> Optional[DeviceAsk]:
    """The device a head phrase names (gift=False), or None."""
    head = _TRAILING_RE.sub("", head).strip(" ,.!?;:-")
    if not head or _OWNER_VERB_RE.search(head):
        return None
    family, m = _rightmost_family(head)
    if family is None:
        return None
    if _OWNED_RE.search(head[:m.start()]):
        return None
    other = _OTHER_DEVICE_ASK_RE.get(family)
    if other and other[0].fullmatch(m.group(0)) and other[1].search(head[:m.start()]):
        return None                                # "drawing tablet", "cordless phone"
    tail = head[m.end():].strip(" ,.!?;:-")
    if _ACCESSORY_NOUN_RE.search(tail):
        return DeviceAsk(family, False, True)      # "laptop sleeve", "watch winder"
    words = tail.split()
    if len(words) > 4 or any(w not in _TAIL_WORDS and not w[:1].isdigit() for w in words):
        return None                                # the device is not the head noun
    return DeviceAsk(family, False, False)


def requested_device(query) -> Optional[DeviceAsk]:
    """The device family the shopper asked for, or None (module docstring)."""
    q = _norm(query)
    if not q:
        return None
    if _GIFT_RE.search(q):
        m = _PERSONA_RE.search(q) or _PERSONA_GIFT_RE.search(q) or _LIKES_RE.search(q)
        if m:
            reading = _phrase_reading(m.group("x").strip())
            if reading is not None:
                return reading._replace(gift=True)
        # no persona naming a device ("gift for my dad", "gift for a coffee lover"): the
        # plain reading decides ("watch gift for my dad" names the watch itself)
    return _phrase_reading(q)


def _phrase_reading(text: str) -> Optional[DeviceAsk]:
    """The reading of the head phrase (the text before the first _PHRASE_END_RE
    connector). A coordinated phrase is a device ask only when EVERY part reads as the
    same device ("headphones or earbuds", "smart watch and fitness tracker"); any other
    part -- an accessory ("iphone and apple watch charger", "phone or tablet stand",
    "headphones and earbuds case": the noun at the END is what is asked for, for every
    device before it), another device ("iphone and ipad"), an add-on ("laptop and
    mouse") -- fails closed: None, the page the shopper had before 2026-10-27."""
    cut = _PHRASE_END_RE.search(text, 1)
    parts = _COORDINATOR_RE.split(text[:cut.start()] if cut else text)
    readings = [_head_reading(part) for part in parts]
    if len(readings) == 1:
        return readings[0]
    first = readings[0]
    if first is None or first.accessory_asked or any(r != first for r in readings[1:]):
        return None
    return first


def names_device(text, family: str) -> bool:
    """Whether a phrase (a search intent) asks for the device itself of `family`."""
    reading = requested_device(text)
    return reading is not None and reading.family == family and not reading.accessory_asked


# ------------------------------------------------------------------ rows
def _label(row: dict, column: str) -> str:
    """A load-time label column as a str ('' for None, NaN, absent)."""
    value = row.get(column)
    return value if isinstance(value, str) else ""


_LAPTOP_NOUN_RE = re.compile(
    r"\b(?:laptops?|notebooks?|chromebooks?|macbooks?|ultrabooks?|netbooks?|2-in-1s?)\b")
_CAMERA_NOUN_RE = re.compile(
    r"\b(?:cameras?|cams?|camcorders?|dslrs?|slrs?|mirrorless|gopros?|dash ?cams?)\b")
_SPEAKER_NOUN_RE = re.compile(
    r"\b(?:speakers?|sound ?bars?|soundbars?|sub ?woofers?|subwoofers?|boom ?box(?:es)?|"
    r"loud ?speakers?)\b")
# A security camera reads smart_home (or optics: "night vision"), a dash cam car; and a
# car or marine speaker car (2026-10-30 review: catalog_taxonomy files "car speakers" and
# "marine speakers" there, so its 124 speakers and subwoofers -- "Alpine SPJ-161C2 6\"
# Coaxial Car Speakers" -- were accessories (81) or unknown (43) on "car speakers", behind
# the car speakers filed audio; _not_a_speaker keeps its head units and empty enclosures out)
_CAMERA_CATEGORIES = frozenset({"cameras", "smart_home", "car", "optics"})
_SPEAKER_CATEGORIES = frozenset({"audio", "smart_home", "car"})
_CATEGORY_OF = {"headphones": "headphones", "phone": "phones", "tablet": "tablets"}
# Electronics rows that are add-ons whatever they name (catalog_taxonomy roles)
_ADDON_ROLES = frozenset({"accessory", "part", "consumable", "service"})


def is_core(row: dict, family: str) -> bool:
    """The row IS the device (load-time columns + its title; contract 3.1). A core row is
    seated first, leads every display band and sort group, counts as on type, passes the
    ad gate and goes to the reranker as ` | role=device`, so it must not be an accessory
    (owner call, 2026-10-27 review).
    - watch: watch_kind 'watch' on any vertical (a Fashion wristwatch and an Electronics
      smartwatch alike; catalog_taxonomy reads the title's segments and targets).
    - every other family needs ALL of: item_role 'device'; the family's category
      (headphones / phones / tablets; laptop = computers with a laptop noun, not a
      desktop; camera = cameras, or a camera device filed under smart_home / car / optics:
      security cameras, dash cams; speaker = an audio, smart-home or car device with a
      speaker noun, not a radio, a head unit, a microphone or an empty subwoofer
      enclosure); the family noun in the title's HEAD, before
      the first "for / compatible / fits / works with / designed for / replacement"
      (_names_in_head: "Replacement Battery for iPhone" names no phone there); and a
      title that is not FOR the device (_for_the_device: "SingPad Apple iPad 2/3/4 Case",
      "Tablet Pc Handle for iPad", "iPad in easy steps", a book). A phone must also be a
      MOBILE phone (D2: iPhone, smartphone, cell phone; _not_a_mobile_phone), not a
      landline, satellite or VoIP phone.
    Never a Grocery row (a "chocolate pocket watch")."""
    if not _category_core(row, family):
        return False
    if family == "watch":
        return True
    if _label(row, "item_role") != "device":
        return False
    return _title_is_the_device(_norm(row.get("Product_title")), family)


@lru_cache(maxsize=65536)
def _title_is_the_device(title: str, family: str) -> bool:
    """is_core's title part for a normalised title. Memoised per (title, family): a
    request reads each pool row's role several times (window, display, sort groups,
    match quality, ads) and pools repeat titles across requests."""
    if not _names_in_head(title, family):
        return False
    if family == "phone" and _not_a_mobile_phone(title):
        return False
    if family == "speaker" and _not_a_speaker(title):
        return False
    return not _for_the_device(title, family)


def _category_core(row: dict, family: str) -> bool:
    """is_core's column part: the load-time category (or watch_kind) alone."""
    if _grocery(row):
        return False
    if family == "watch":
        return _label(row, "watch_kind") == "watch"
    category = _label(row, "item_category")
    if not category:
        return False
    if family in _CATEGORY_OF:
        return category == _CATEGORY_OF[family]
    title = _norm(row.get("Product_title"))
    if family == "laptop":
        return category == "computers" and bool(_LAPTOP_NOUN_RE.search(title))
    if family == "camera":
        return category == "cameras" or (
            category in _CAMERA_CATEGORIES and _label(row, "item_role") == "device"
            and bool(_CAMERA_NOUN_RE.search(title)))
    if family == "speaker":
        return (category in _SPEAKER_CATEGORIES and _label(row, "item_role") == "device"
                and bool(_SPEAKER_NOUN_RE.search(title)))
    return False


# Where a title's head ends: what follows names the device the product is FOR. A
# "desktop replacement" is a laptop class, not a replacement part ("ROG G703GX Desktop
# Replacement Gaming Laptop")
_TITLE_TARGET_RE = re.compile(
    r"(?<![\w-])(?:for|compatible|fits|works with|designed for|made for|"
    r"(?<!desktop )replacement)(?![\w-])")
# Device names a title's head may use besides the query's family nouns (FAMILY_RE): the
# model lines a listing names instead ("Samsung Galaxy Tab A7", "Fire HD 10", "Moto G
# Power", "Boox Ebook Reader", "Instax Mini 9")
_HEAD_NOUN_EXTRA = {
    # a bare "2-in-1" only next to a computer spec: it also names plug adapters, air
    # dusters and a tank model kit filed under computers
    "laptop": (r"notebooks?|netbooks?|"
               r"2-in-1(?=.*\b(?:\d+ ?(?:gb|tb)|ssd|ram|intel|ryzen|core i\d))"),
    "headphones": r"in-ear monitors?",
    "phone": r"galaxy (?:s|note|a|z|j|m) ?\d+\w*|pixel \d\w*|moto [gez]\w*|oneplus",
    "tablet": r"galaxy tabs?|fire hd|fire \d+|surface (?:pro|go)|nook|kobo|ebook readers?",
    "camera": r"cams?|webcams?|slrs?|mirrorless|instax",
    "speaker": r"sub ?woofers?|loud ?speakers?",
}
_HEAD_NOUN_RE = {f: re.compile(FAMILY_RE[f].pattern + (
    r"|\b(?:" + _HEAD_NOUN_EXTRA[f] + r")\b" if f in _HEAD_NOUN_EXTRA else ""))
    for f in FAMILY_RE}


def _names_in_head(title: str, family: str) -> bool:
    """Whether a normalised title names a `family` device before its first target
    connector, read as written or with its parentheses dropped: "(Compatible for
    Hikvision) Outdoor PTZ IP Camera" names its camera after them, "Animation Kit (Camera,
    Software, Book) for Windows" inside them."""
    for text in (title, _TITLE_PAREN_RE.sub(" ", title)):
        cut = _TITLE_TARGET_RE.search(text)
        if _HEAD_NOUN_RE[family].search(text[:cut.start()] if cut else text):
            return True
    return False


# ------------------------------------------------------------------ the title veto
# 2026-10-27 review round. catalog_taxonomy's categories file many add-ons INSIDE the
# device's category (its item_role is the category's: every tablets row is a "device"), and
# a row marked core is seated first, leads every display band and sort group, counts as
# on-type and goes to the reranker as ` | role=device`. Judged seeded samples of the
# category-only predicate (60 rows a family, build/NOTES_B1.md "Review round"): devices
# tablet 40, phone 12 (landlines, satellite and VoIP phones, car receivers, mounts, calling
# cards), speaker 48, camera 51, laptop 54, headphones 59 -- "SingPad Apple iPad 2/3/4
# Case", "Tablet Pc Handle for iPad", "iPad in easy steps" (a book) were all `role=device`
# on "ipad"; the family noun in the head (_names_in_head) alone kept most of them (phone
# precision 0.48 -> 0.54, tablet 0.73 -> 0.76 on 120 judged rows each). Watches keep
# watch_kind alone (catalog_taxonomy reads their segments; 56 of 60 judged). A row is FOR
# the device when its title says so, read from the HEAD of the title, not from any
# accessory word in it: real tablets list "6000mAh Battery, IPS Screen" and real earbuds a
# "Charging Case".
_B, _E = r"(?<![\w-])", r"(?![\w-])"   # word edges that keep "ear-hook" and "i-flash" whole
# Where the head phrase ends: what follows is what the row is for or comes with
_TITLE_CONNECT_RE = re.compile(
    r"(?<![\w-])w/|" + _B + r"(?:for|with|compatible (?:with|for|to)|fits|including|"
    r"includes?|included|supports?|works with|designed for|made for|pour)" + _E + r"|\s\+\s")
# A bare "compatible" names what follows as the product: "FinePix XP20 Digital Camera
# Compatible USB 2.0 Cable Cord", "Cell Phone Compatible 8GB Micro SDHC Memory Card"
_BARE_COMPATIBLE_RE = re.compile(r"\bcompatible\b(?! (?:with|for|to)\b)")
_TITLE_FOR_RE = re.compile(_B + r"(?:for|compatible (?:with|for)|fits|works with|"
                           r"designed for|made for|to fit|pour)" + _E)
_TITLE_SEP_RE = re.compile(r"\s[-–—]\s|[,;|]|\s/\s|\s&\s")
_TITLE_PAREN_RE = re.compile(r"\([^()]*\)|\[[^\[\]]*\]")
# Who or where a "for ..." phrase names, not what: "Camera for Kids ... Toddler Camera",
# "Moto G Power ... Made for US by Motorola"
_FOR_STOP_RE = re.compile(
    _B + r"(?:kids?|children|childrens|toddlers?|boys?|girls?|men|women|adults?|teens?|"
    r"seniors?|beginners?|home|travel|outdoor|indoor|office|cars?|business|school|"
    r"students?|gaming|gamers?|running|sports?|gym|us|by)" + _E)
# Device names a "for ..." phrase may use besides the family nouns ("Case for Galaxy Tab")
_FOR_DEVICE_EXTRA = {
    "tablet": r"galaxy tab|fire hd|fire tablet|surface pro|nook|kobo",
    "phone": r"galaxy|pixel|android|ios|motorola|moto",
    "headphones": r"airpods|beats|bose",
    "laptop": r"macbook|chromebook|thinkpad|dell|hp|lenovo|asus|acer",
    "camera": r"gopro|canon|nikon|dslr|mirrorless|fujifilm|sony a\d+",
    "speaker": r"echo dot|sonos|jbl|bose",
}
_FOR_DEVICE_RE = {f: re.compile(FAMILY_RE[f].pattern + r"|\b(?:" + extra + r")\b")
                  for f, extra in _FOR_DEVICE_EXTRA.items()}
# Phone and tablet titles name the device in the head of their add-ons too ("Mobile Phone
# for Camera Smartphone Lens"), so their "for <device>" is read even then; the others'
# only when the head names no device ("Earbuds for iPhone Headphones" are earbuds)
_ALWAYS_FOR = frozenset({"phone", "tablet"})
_ADDON_TITLE_RE = re.compile(
    _B + r"(?:cases?|covers?|sleeves?|skins?|shells?|bumpers?|folios?|padfolios?|etuis?|"
    r"holders?|holsters?|stands?|mounts?|docks?|docking|cradles?|chargers?|cables?|cords?|"
    r"adapters?|protectors?|tempered glass|straps?|grips?|handles?|pillows?|cushions?|bags?|"
    r"pouch(?:es)?|backpacks?|attaches?|crossbody|wallets?|decals?|stickers?|sunshades?|"
    r"slipcases?|"
    r"hoods?|tripods?|gimbals?|rigs?|cages?|dolly|dollies|sliders?|lens(?:es)?|filters?|"
    r"organi[sz]ers?|hangers?|hooks?|brackets?|wires?|replacements?|spare|repair|parts?|"
    r"mouse|mice|fans?|microphones?|mics?|locators?|flash drives?|sticks?|memory cards?|"
    r"sd cards?|microsd|ear ?pads?|ear ?tips?|tips|spacers?|tubes?|grills?|knobs?|remotes?|"
    r"video lights?|light panels?|cleaning|cleaners?|boosters?|boxes|box|locks?|"
    r"accessor(?:y|ies)|bases?|armbands?|wedges?|signs?|daypacks?|diaphragms?|voice coils?|"
    r"awg|transmitters?|caps?|cool(?:er|ing) pads?|cooling mats?|"
    # a housing is an add-on ("GoPro Protective Housing"), a metal one the device's body
    # ("Full Metal Housing Dual Driver ... In-ear", "Zinc Alloy Metal Housing ... Backup Camera")
    r"(?<!metal )(?<!alloy )(?<!aluminum )(?<!aluminium )housings?)" + _E)
# Words that name the device itself, or a spec of it, in that family's titles: a "DSLR
# Camera 18-55mm Lens Kit", "Earbuds with Microphone", the EliteBook Folio, a "Surface
# Mount Speaker"
_ADDON_EXCEPT = {"camera": frozenset({"lens", "lenses"}),
                 "headphones": frozenset({"mic", "mics", "microphone", "microphones", "grip",
                                          "grips"}),
                 "laptop": frozenset({"folio", "folios"}),
                 "speaker": frozenset({"mount", "mounts"})}
# Add-ons that make a compound when they come right BEFORE the device noun ("Case iPad 4",
# "Sunshade Hood Base iPad Mini", "Ear Pads Headphones Foam")
_ADDON_BEFORE_RE = re.compile(
    _B + r"(?:cases?|covers?|skins?|sleeves?|protectors?|chargers?|cables?|holders?|stands?|"
    r"adapters?|straps?|bags?|bases?|pads?|lens(?:es)?)" + _E)
# Parts right AFTER the device noun ("Laptop Screen 15.6", "iPhone Battery")
_PART_AFTER_RE = re.compile(
    r"\s*(?:screens?|lcds?|batter(?:y|ies)|ram|memory|keyboards?|motherboards?|hinges?|"
    r"hard drives?|ssds?|power suppl(?:y|ies)|bezels?|housings?|digitizers?)" + _E)
# Not the device at all: a book, a plan, a dummy -- or pills counted in tablets ("BETA-7
# Beta-Alanine, 180 Tablets" is an Electronics row in the tablets category)
_NOT_A_PRODUCT_RE = re.compile(
    r"\b(?:for dummies|in easy steps|user guide|manual|handbook|paperback|hardcover|"
    r"protection plan|warranty plan|calling cards?|dummy|fake|decoy|replica|"
    r"(?<!android )(?<!windows )\d{2,} (?:ct |count )?tablets)\b")
# Another device named before the family noun (or with none): the row is that device
# ("MP3 Player ... Built-in Speaker", "Car Stereo Radio ... support Backup Camera",
# "Smartphone Based Navigation System", "Green Board LCD Tablet ... eWriter")
_OTHER_DEVICE_RE = {f: re.compile(_B + r"(?:" + words + r")" + _E) for f, words in {
    "speaker": r"radios?|turntables?|record players?|monitors?|walkie|talkies?|recorders?|"
               r"players?|mp3|mp4|head ?phones?|earphones?|earbuds?|headsets?|beanies?|hats?|"
               r"amplifiers?|receivers?|microphones?|glasses|sunglasses|speakerphones?",
    "camera": r"stereos?|radios?|phones?|booths?|drones?|quadcopters?|tablets?",
    "phone": r"receivers?|stereos?|radios?|projectors?|navigation|recorders?|dvrs?|rings?|"
             r"tags?|trackers?",
    "tablet": r"controllers?|keyboards?|transmitters?|wacom|e-?writers?|writing boards?|"
              r"autoradios?|stereos?|head ?units?|navigation|"
              r"(?:lcd|writing|drawing|graphics?|pen|paper)"
              r"(?: electronic| writing)? tablets?",
}.items()}
# A camera title whose first clause is a recorder with no camera named, and which lists
# no cameras at all, is the recorder alone (a DVR); "... DVR System with Ten Security
# Cameras" stays a camera system
_RECORDER_RE = re.compile(_B + r"(?:dvrs?|nvrs?|recorders?)" + _E)
_OTHER_FAMILIES_RE = {f: re.compile("|".join("(?:" + rx.pattern + ")"
                                             for g, rx in FAMILY_RE.items() if g != f))
                      for f in FAMILY_RE}
# D2's phone is a MOBILE phone (iPhone, smartphone, cell phone): the phones category also
# holds landlines, cordless, satellite and VoIP phones, call-center headsets and calling
# cards; on its judged sample this check alone kept 12 of 12 handsets that carry a cue
_MOBILE_CUE_RE = re.compile(
    r"\b(?:smart ?phones?|cell ?phones?|cellphones?|mobile phones?|iphones?|android phones?|"
    r"flip phones?|unlocked|prepaid|galaxy (?:s|note|a|z|j|m) ?\d+\w*|pixel \d)\b")
_LANDLINE_RE = re.compile(
    r"\b(?:landlines?|cordless|corded|telephones?|handsets?|desk phones?|ip phones?|voip|"
    r"satellite|speakerphones?|call center|walkie|retro)\b")
_PHONE_CROSS_RE = re.compile(
    r"\b(?:tablets?|ipads?|galaxy tab|earbuds?|ear ?buds|headsets?|head ?phones?|earphones?)\b")


_YOUR_DEVICE_RE = {f: re.compile(r"\b(?:your|my) (?:[\w'-]+ ){0,2}(?:" + rx.pattern + r")")
                   for f, rx in _HEAD_NOUN_RE.items()}


def _addons(text: str, family: str) -> list:
    """The add-on noun matches in `text` that are add-ons for this family."""
    exempt = _ADDON_EXCEPT.get(family, frozenset())
    return [m for m in _ADDON_TITLE_RE.finditer(text) if m.group(0) not in exempt]


def _not_a_mobile_phone(title: str) -> bool:
    cut = _TITLE_CONNECT_RE.search(title)
    head = title[:cut.start()] if cut else title
    return (not _MOBILE_CUE_RE.search(title) or bool(_LANDLINE_RE.search(title))
            or bool(_PHONE_CROSS_RE.search(head)))


# Car audio the speaker family must not take as a speaker (2026-10-30 review, with the car
# category): of its 147 new core rows 5 were head units and 18 subwoofer enclosures that do
# not say they come loaded (122 of the 124 left are speakers; the other two, a baffle and a
# speaker ring kit, are a residue audio-filed rows share). Both read in the title's first
# clause, the right-headed compound idiom: a head unit named
# before any speaker unit ("Double Din Car Stereo with ... Carplay", "1/2 DIN ... Graphic
# Equalizer"; not a "Car Stereo Speaker", nor "SUBWOOFER CAR Audio CAR Stereo RMS"), an
# enclosure with no speaker unit after it ("Bbox Dual Sealed 10 Inch Subwoofer Enclosure",
# "MTX Audio SLH12U Unloaded Subwoofer Enclosure") unless it comes loaded ("... Loaded
# Subwoofer Enclosure", "Powered Car Subwoofer Enclosure"). The DIN words also took three
# audio-filed "1/2 DIN ... Equalizer" rows off the speakers
_HEAD_UNIT_RE = re.compile(
    _B + r"(?:(?:half|single|double|1/2|1|2)[ -]?dins?|(?:car|marine|boat|auto) "
    r"stereos?(?! (?:speakers?|sub ?woofers?|tweeters?|coaxial|component)))" + _E)
_ENCLOSURE_RE = re.compile(_B + r"enclosures?" + _E)
_SPEAKER_UNIT_RE = re.compile(_B + r"(?:speakers?|sub ?woofers?|subs|woofers?|tweeters?)" + _E)
_LOADED_RE = re.compile(_B + r"(?:loaded|powered|amplified|active)" + _E)


def _not_a_speaker(title: str) -> bool:
    text = _TITLE_PAREN_RE.sub(" ", title)
    cut = _TITLE_CONNECT_RE.search(text)
    # an unspaced "+" joins a bundle too: "Car Stereo Speakers+(2) Enclosures"
    head = (text[:cut.start()] if cut else text).split("+")[0]
    first = next((c for c in _TITLE_SEP_RE.split(head) if c.strip()), "")
    unit, stereo = _SPEAKER_UNIT_RE.search(first), _HEAD_UNIT_RE.search(first)
    if stereo and (unit is None or stereo.start() < unit.start()):
        return True
    boxes = list(_ENCLOSURE_RE.finditer(first))
    return (bool(boxes) and not _LOADED_RE.search(title)
            and not _SPEAKER_UNIT_RE.search(first, boxes[-1].end()))


def _for_the_device(title: str, family: str) -> bool:
    """Whether a normalised title is FOR a `family` device rather than one: a "for
    <device>" phrase, a title that opens with an add-on, an add-on as the head of its
    first 8 words, another device named first, or an add-on / part compounded with a
    device noun (memoised through _title_is_the_device)."""
    family_re = FAMILY_RE[family]
    if _NOT_A_PRODUCT_RE.search(title):
        return True
    # A listing that addresses the shopper's own device is an add-on for it ("Attaches
    # Directly to Your Smartphone", "Protect Your Samsung Galaxy Tab 3")
    if _YOUR_DEVICE_RE[family].search(title):
        return True
    cut = _TITLE_CONNECT_RE.search(title)
    head = _TITLE_PAREN_RE.sub(" ", title[:cut.start()] if cut else title)
    clauses = [c.strip() for c in _TITLE_SEP_RE.split(head) if c.strip()]
    first = clauses[0] if clauses else ""
    # "Case for iPad", "Lens for Canon DSLR Camera": a device noun within 4 words of "for"
    if family in _ALWAYS_FOR or not family_re.search(head):
        device = _FOR_DEVICE_RE[family]
        for clause in _TITLE_SEP_RE.split(title):
            for m in _TITLE_FOR_RE.finditer(clause):
                rest = clause[m.end():]
                stop = _TITLE_CONNECT_RE.search(rest)
                words = (rest[:stop.start()] if stop else rest).split()[:4]
                for k, word in enumerate(words):
                    if _FOR_STOP_RE.fullmatch(word):
                        break
                    if device.match(" ".join(words[k:])):
                        return True
    # "Decals Stickers ... Laptops", "USB Finger Mouse ... Laptop": opens with an add-on;
    # "Tablet Pc Handle", "Laptop Stand Ergonomic": the head of the first 8 words is one
    words = first.split()
    if _addons(" ".join(words[:3]), family) and not family_re.match(first):
        return True
    opening = " ".join(words[:8])
    nouns = [(m.end(), False) for m in family_re.finditer(opening)] + \
        [(m.end(), True) for m in _addons(opening, family)]
    if nouns and max(nouns)[1]:
        return True
    other = _OTHER_DEVICE_RE.get(family)
    if other is not None:
        found, own = other.search(first), family_re.search(first)
        if found and (family == "phone" or own is None or found.start() < own.start()):
            return True
        if family != "camera":
            # ... or before the first family noun anywhere in the head ("Kenwood ... USB &
            # CD Receiver ... iPod/iPhone"); a camera's head lists its own "Car DVR"
            found, own = other.search(head), family_re.search(head)
            if found and (own is None or found.start() < own.start()):
                return True
    if family == "camera" and _RECORDER_RE.search(first) and not family_re.search(first) \
            and not re.search(r"\bcameras\b", title):
        return True
    bare = _BARE_COMPATIBLE_RE.search(title)
    if bare:
        rest = _TITLE_SEP_RE.split(title[bare.end():], 1)[0]
        stop = _TITLE_CONNECT_RE.search(rest)
        if _addons(" ".join((rest[:stop.start()] if stop else rest).split()[:6]), family):
            return True
    # An add-on just after a device noun (3 words) or a strong one just before it: the
    # family's own noun up to the first clause naming it ("Camera, Camera Case, Tripod" is
    # a bundle), another family's noun in any clause ("Tablet Slim - 10.1 Laptop Bag")
    own_named = False
    for clause in clauses:
        for rx in (_OTHER_FAMILIES_RE[family],) if own_named else (family_re,
                                                                   _OTHER_FAMILIES_RE[family]):
            for m in rx.finditer(clause):
                after = clause[m.end():]
                if _addons(" ".join(after.split()[:3]), family) or _PART_AFTER_RE.match(after):
                    return True
                before = " ".join(clause[:m.start()].split()[-1:])
                if _ADDON_BEFORE_RE.fullmatch(before) and \
                        before not in _ADDON_EXCEPT.get(family, frozenset()):
                    return True
        own_named = own_named or bool(family_re.search(clause))
    return False


def _grocery(row: dict) -> bool:
    vertical = row.get("bsns_vrtcl_name")
    return isinstance(vertical, str) and vertical.strip().casefold() == "grocery"


def mentions(row: dict, family: str) -> bool:
    """The row is about the family: its device, or something that names it. For watches
    catalog_taxonomy's reader decides (it refuses "watch movies", "bird watching"). A
    Grocery row never is ("Dishwasher Tablets", a "chocolate watch"): food is no device
    and no add-on for one, so a Grocery pool can never pass the device gate."""
    if _grocery(row):
        return False
    if family == "watch":
        return bool(_label(row, "watch_kind"))
    return is_core(row, family) or bool(FAMILY_RE[family].search(_norm(row.get("Product_title"))))


def role(row: dict, family: str) -> int:
    """CORE (the device), ACCESSORY (names the family without being it, or an Electronics
    accessory / part / consumable of any kind: a screen cleaner kit on a watch ask), else
    UNKNOWN (no evidence either way: a wallet, a desktop, a grocery row)."""
    if is_core(row, family):
        return CORE
    if mentions(row, family) or _label(row, "item_role") in _ADDON_ROLES:
        return ACCESSORY
    return UNKNOWN


_STORAGE_CATEGORIES = frozenset({"cases", "bags", "mounts"})
_CARE_CATEGORIES = frozenset({"tools", "supplies", "components", "cables", "power",
                              "batteries", "screen_protectors"})
_WATCH_ADDON_RANK = {"storage": 0, "band": 1, "protection": 1, "other": 1}


def addon_rank(row: dict, family: str) -> int:
    """0 storage / display, 1 other add-ons, 2 tools, care, parts, cables and replacement
    items (owner call D1: what a collector would least like to unwrap)."""
    kind = _label(row, "watch_kind")
    if family == "watch" and kind and kind != "watch":
        return _WATCH_ADDON_RANK.get(kind, 2)       # care_tool, part, charger -> 2
    category = _label(row, "item_category")
    if category in _STORAGE_CATEGORIES:
        return 0
    if category in _CARE_CATEGORIES or kind in ("care_tool", "part", "charger"):
        return 2
    return 1


# ------------------------------------------------------------------ the reranker's text
# Deterministic per family, never the shopper's words; the examples sit in parentheses
# like main._type_constraint's. No braces: the constraints block is rendered as is.
_WANT = {
    "watch": "a watch itself (wristwatches, pocket watches, smartwatches)",
    "laptop": "a laptop computer itself (laptops, notebooks, chromebooks)",
    "headphones": "headphones or earbuds themselves (over-ear headphones, earbuds, headsets)",
    "phone": "a phone itself (smartphones, cell phones)",
    "tablet": "a tablet or e-reader itself (tablets, e-readers)",
    "camera": "a camera itself (digital, mirrorless, action and security cameras, camcorders)",
    "speaker": "a speaker itself (portable, bluetooth and smart speakers, soundbars)",
}
_ACCESSORIES = {
    "watch": ("watch boxes, winders, display cases, straps, bands, chargers, tools, repair "
              "kits, cleaning and polishing supplies"),
    "laptop": ("laptop bags, backpacks, sleeves, cases, chargers, stands, cooling pads, "
               "skins and keyboard covers"),
    "headphones": "cases, ear pads, ear tips, cables, adapters, stands and chargers",
    "phone": "cases, screen protectors, chargers, cables, mounts and holders",
    "tablet": "cases, covers, keyboards, screen protectors, styluses, chargers and stands",
    "camera": ("bags, cases, mounts, tripods, straps, lenses, batteries, memory cards and "
               "cleaning kits"),
    "speaker": "mounts, stands, cases, covers and cables",
}


def constraint(ask: DeviceAsk) -> dict:
    """{'type': 'product_type', 'value': ...} for the reranker's constraints block: a
    plain ask caps every accessory for the device; a gift ask (owner call D1) puts the
    device first, storage and display below it, and caps tools, care, parts and
    replacement items (scoped "for it": the coffee of a coffee lover is never capped)."""
    want = _WANT[ask.family]
    if ask.gift:
        value = (f"the gift is {want}; storage and display pieces for it rank below every "
                 f"real one; tools, repair kits, cleaning and polishing supplies, spare "
                 f"parts, cables and replacement items for it violate it")
    else:
        value = f"{want}; accessories FOR it ({_ACCESSORIES[ask.family]}) violate it"
    return {"type": "product_type", "value": value}
