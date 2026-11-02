"""Offline tests for app/catalog_taxonomy.py (2026-10-27) and its app/search.py hook.

No network, no LLM, no catalog unless CAS_SLOW_TESTS=1. Runs under pytest
(`python -m pytest tests/test_catalog_taxonomy.py`) or plainly
(`python tests/test_catalog_taxonomy.py`).

Pinned here: the published vocabularies and rail groups (contract 3.1: the rail API and
the ranking read them), a hand-checked title table per Electronics and Grocery category
(original case and lowercased), the documented misreads of the 2026-10-27 held-out read
as PINNED wrong answers (a fix shows up as a deliberate table change), the watch kinds
with the 2026-10-27 reader fixes, the facet readers (diet tags, works with, condition,
brand), the pattern-derived index (sound on every table title; the derivation's own
units), annotate_catalog on a Databricks-shaped frame (object dtype, None, NaN, pd.NA),
and that it never moves or edits a row. The app/search.py hook: _annotate_types adds
the columns after product_type and logs counts only; the LOCAL derived-column cache
(.cache/taxonomy_<csv md5>_<rules digest>.npz) round-trips exactly, keys on the catalog,
the rules file, its format and the Python minor version, is never served for another
frame (a row-count change, a corrupt file, other columns: a miss), never keyed by a
rules file edited after its import (2026-10-30), never written without a csv hash
(Databricks) and never fails a boot. CAS_SLOW_TESTS=1 (PRODUCTS_CSV=
data/products_enriched.csv) adds the whole catalog: coverage within the measured ranges,
the index soundness on every segment from every token start, the watch reader's needles,
the fast reading against the exhaustive one (a counted residue, all of it finditer's
overlap rule), the watches/watch_kind identity and the timing.
"""

import contextlib
import logging
import os
import random
import re
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import app.config as cfg  # noqa: E402
from app import catalog_taxonomy as ct  # noqa: E402
from app import search as S  # noqa: E402


class _Skipped(Exception):
    """Raised by _skip() under the plain runner (it reports it), pytest or not."""


_PLAIN_RUNNER = False  # set by __main__: pytest.skip() there would abort the whole run


def _skip(reason):
    if _PLAIN_RUNNER:
        raise _Skipped(reason)
    try:
        import pytest
    except ImportError:
        raise _Skipped(reason)
    pytest.skip(reason)


@contextlib.contextmanager
def patched(obj, attrs):
    """Set attributes for one test and always restore them."""
    old = {k: getattr(obj, k) for k in attrs}
    for k, v in attrs.items():
        setattr(obj, k, v)
    try:
        yield
    finally:
        for k, v in old.items():
            setattr(obj, k, v)


# ------------------------------------------------------------------ the contract
ELECTRONICS_KEYS = (
    "cases cables components power audio camera_accessories headphones computers mounts "
    "storage bags computer_accessories av_accessories batteries cameras backdrops "
    "screen_protectors tv_video watch_accessories smart_home car networking optics monitors "
    "tablets watches supplies lighting tools office service gaming phones").split()
GROCERY_KEYS = ("candy sauces snacks spices fresh bakery tea beverages coffee cake_decor "
                "baking breakfast pantry nuts canned gifts non_food protein").split()


def test_vocabularies_are_the_contract():
    assert set(ct.ELECTRONICS_CATEGORIES) == set(ELECTRONICS_KEYS) and len(ELECTRONICS_KEYS) == 33
    assert set(ct.GROCERY_CATEGORIES) == set(GROCERY_KEYS) and len(GROCERY_KEYS) == 18
    assert not set(ct.ELECTRONICS_CATEGORIES) & set(ct.GROCERY_CATEGORIES)
    assert ct.CATEGORY_LABELS == {**ct.ELECTRONICS_CATEGORIES, **ct.GROCERY_CATEGORIES}
    assert ct.UNKNOWN not in ct.CATEGORY_LABELS
    for label in ct.CATEGORY_LABELS.values():
        assert isinstance(label, str) and label.strip() == label and label
    assert ct.GROCERY_CATEGORIES["non_food"] == "Kitchen, pet & other non-food"
    roles = {ct._ELECTRONICS_TABLE[k][1] for k in ELECTRONICS_KEYS}
    assert roles == set(ct.ELECTRONICS_ROLES)
    assert {ct._GROCERY_TABLE[k][1] for k in GROCERY_KEYS} == set(ct.GROCERY_ROLES)
    assert set(ct.ROLE_LABELS) == set(ct.ELECTRONICS_ROLES) | set(ct.GROCERY_ROLES)
    assert ct.ADDED_COLUMNS == ("item_category", "item_role", "watch_kind", "diet_tags",
                                "works_with", "item_condition", "brand")
    assert ct.WATCH_KINDS == ("watch", "band", "storage", "care_tool", "protection", "charger",
                              "part", "other")
    assert set(ct.WORKS_WITH_LABELS) == {"apple", "samsung", "google", "amazon", "microsoft",
                                         "playstation", "nintendo", "pc", "camera", "fitness"}
    assert ct.WORKS_WITH_LABELS["pc"] == "Dell, HP, Lenovo & PCs"
    assert ct.CONDITION_LABELS == {"renewed": "Renewed / refurbished", "used": "Used"}
    assert len(ct.DIETARY_VALUES) == 15 and set(ct.DIETARY_LABELS) == set(ct.DIETARY_VALUES)
    assert ct.DIETARY_LABELS["sugar free"] == "Sugar free / no added sugar"


def test_rail_groups_cover_every_key_once():
    members = [k for _label, keys in ct.ELECTRONICS_RAIL_GROUPS.values() for k in keys]
    assert len(members) == len(set(members))
    assert set(members) == set(ELECTRONICS_KEYS) - {"service"}   # plans are never offered
    assert len(ct.ELECTRONICS_RAIL_GROUPS) == 21
    for group, (label, keys) in ct.ELECTRONICS_RAIL_GROUPS.items():
        assert re.fullmatch(r"[a-z][a-z_]{0,31}", group), group
        assert label and keys
    assert ct.ELECTRONICS_RAIL_GROUPS["computers_monitors"] == (
        "Computers & monitors", ("computers", "monitors"))
    assert set(ct.GROCERY_RAIL_GROUPS) == set(GROCERY_KEYS)
    assert all(keys == (k,) for k, (_l, keys) in ct.GROCERY_RAIL_GROUPS.items())


def test_rail_group_lookup_and_one_category_param():
    """The rail's `category` param serves both departments (contract 3.3): Electronics
    sends group keys, Grocery its 18 keys, so the two vocabularies must not collide."""
    assert not set(ct.ELECTRONICS_RAIL_GROUPS) & set(ct.GROCERY_RAIL_GROUPS)
    assert set(ct.ELECTRONICS_GROUP_OF) == set(ELECTRONICS_KEYS) - {"service"}
    for key, group in ct.ELECTRONICS_GROUP_OF.items():
        assert key in ct.ELECTRONICS_RAIL_GROUPS[group][1]
    assert ct.ELECTRONICS_GROUP_OF["monitors"] == "computers_monitors"
    assert ct.ELECTRONICS_GROUP_OF["watches"] == "watches"


def test_module_is_pure_and_self_contained():
    """No I/O and no app import: the local taxonomy cache keys on THIS file's bytes
    (app/search.py), so a rule kept in another module would be served stale. The one
    clock read is IMPORTED_AT (when the rules were loaded, for that key); no classifier
    reads it."""
    import ast
    tree = ast.parse(Path(ct.__file__).read_text(encoding="utf-8"))
    imported, clock = set(), []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in ("open", "eval", "exec"), node.func.id
        elif (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
              and node.value.id == "time"):
            clock.append(node.attr)
    assert imported <= {"math", "re", "typing", "pandas", "sre_constants", "sre_parse",
                        "time"}, imported
    assert not any(name.startswith("app") for name in imported)
    assert clock == ["time"], clock
    assert sum(isinstance(n, ast.Name) and n.id == "IMPORTED_AT" for n in ast.walk(tree)) == 1
    assert isinstance(ct.IMPORTED_AT, float) and ct.IMPORTED_AT <= time.time()


# ------------------------------------------------------------------ title tables
# (title, item_category, item_role): every Electronics category at least once, plus the
# 2026-10-27 post-read fixes (model-only heads, "plus", bundle "with", lamps, "fit").
ELECTRONICS_TITLES = [
    ("Wireless Earbuds Bluetooth 5.3 Headphones with Charging Case", "headphones", "device"),
    ("Monoprice Wireless 8-Inch 110-Watt Powered Subwoofer - Black (110544)", "audio", "device"),
    ("Apple iPhone 12, 64GB, Black - Unlocked (Renewed)", "phones", "device"),
    ("Lenovo IdeaPad 3 15.6 inch Laptop, 8GB RAM, 256GB SSD", "computers", "device"),
    ("Samsung Galaxy Tab A 10.1 inch 128GB with 64GB Memory Card", "tablets", "device"),
    ('Acer RT280K Abmiipx 28" Ultra HD 4K2K (3840 x 2160) HDR Ready TN Monitor', "monitors",
     "device"),
    ("Roku Express HD Streaming Media Player", "tv_video", "device"),
    ("Digital Concepts 6.1MP Digital Camera", "cameras", "device"),
    ("SAMSUNG Galaxy Watch Active2 (Silicon Strap + Aluminum Bezel) Bluetooth - International",
     "watches", "device"),
    ("Ring Video Doorbell with HD Video", "smart_home", "device"),
    ("Intel Dual Band Wireless-N 7260 Network Adapter (7260.HMWNBWB.R)", "networking", "device"),
    ("Xbox Wireless Controller - Carbon Black", "gaming", "device"),
    ("Lowrance LCX-28C HD 7-Inch Waterproof Marine GPS and Chartplotter (50/200 kHz Transducer)",
     "car", "device"),
    ("HP OfficeJet Pro 9015 All-in-One Wireless Printer", "office", "device"),
    ("PEKAK 10x25 Small Compact Mini Binoculars for Adults", "optics", "device"),
    ("Rechargeable LED Flashlight, 2000 Lumens", "lighting", "device"),
    ("Silicone Case for iPhone 12", "cases", "accessory"),
    ("Tempered Glass Screen Protector for Samsung Galaxy S21", "screen_protectors", "accessory"),
    ("ELECBRAiN Octagon Tip Laptop Ac Adapter Charger Replacement for Inspiron 1545 1546 PA-21 "
     "with Power Cord", "power", "accessory"),
    ("Energizer AA Batteries, 24 Pack", "batteries", "consumable"),
    ("CESS-001L Spade Fork Plug to Speakon Female Jack Adapter for Speaker Cable - 2 Pack "
     "(Large Spade)", "cables", "accessory"),
    ("Rosewill True 1.3 M Effective Pixels USB Webcam (RCM-8163)", "computer_accessories",
     "accessory"),
    ('Morel CAW 428 4" Classic Woofer', "components", "part"),
    ("SanDisk 128GB Ultra microSDXC Memory Card", "storage", "accessory"),
    ("Manfrotto Compact Action Aluminium 5-Section Tripod", "camera_accessories", "accessory"),
    ("MEHOFOTO Summer Beach Tropical Flamingo Theme Photography Backdrop Props", "backdrops",
     "accessory"),
    ("Mountainsmith Network Laptop Bag, Anvil Grey", "bags", "accessory"),
    ("Premium Aluminum Dual Arm Monitor Mount Stand-Full Motion Gas Spring Monitor Riser",
     "mounts", "accessory"),
    ("Kameleon 8-in-1 Universal Remote Control", "av_accessories", "accessory"),
    ("Small S Bands Replacement with Metal Clasps Replacement for Fitbit Flex Only",
     "watch_accessories", "accessory"),
    ("HP 62XL Black High-yield Ink Cartridge", "supplies", "consumable"),
    ("iFixit Pro Tech Toolkit - Electronics, Smartphone, Computer & Tablet Repair Kit", "tools",
     "accessory"),
    ("ASURION 1 Year Floorcare Extended Protection Plan ($0 - $49.99)", "service", "service"),
    # post-read fixes: a model-only head defers to the accessory noun; "plus" is no cut
    ("iPhone 7 Plus, iPhone 8 Plus, Silicone Cover", "cases", "accessory"),
    ("iPhone 6 Plus Cases", "cases", "accessory"),
    ("Electrified NP-04LP / 60002027 Replacement Lamp with Housing for NEC Projectors",
     "components", "part"),
    ("Case Fit All-New Fire HD 10 Plus Tablet", "cases", "accessory"),   # "fit <brand>"
    ("Secure Fit Sport Earphones", "headphones", "device"),               # a bare "fit"
    ("Slim Fit Folio Case", "cases", "accessory"),
    # watches are watch_kind "watch" rows (one source)
    ("Kalawen Watch Winder for Rolex", "watch_accessories", "accessory"),
    ("Apple Watch 42mm Stainless Steel Case with Black Sport Band", "watches", "device"),
    ("RCA Lyra 512MB Sport MP3 Player with Pedometer/Stopwatch", "audio", "device"),
    ("Huami Amazfit Charger ,Itian Amazfit Huami Charging Dock Station Cradle Holder", "power",
     "accessory"),
    ("Mystery Gadget Model X1", "unknown", None),
    # a device named after "for" is what the product is FOR, never the product, even when
    # the head names nothing the rules know (907 catalog titles depend on this skip; the
    # rows above all name their noun first, so these two keep it tested): unknown, not
    # tablets / audio
    ("STM Jacket for iPad", "unknown", None),
    ("Speck iGuy Mini for iPod mini", "unknown", None),
    # 2026-10-30 R2. A box or storage case FOR watches is a watch accessory whatever
    # container noun follows (both read cases); a smartwatch's protective "watch case" is a
    # protective case (it read watch_accessories as storage); a watch stand stays a stand
    ("Watch Box 12 Slot Display Case", "watch_accessories", "accessory"),
    ("Watch Storage Case for 10 Watches", "watch_accessories", "accessory"),
    ("HoneyGirl Watch Case for Fitbit Versa 3, Scratch-Proof Cover Protective Bumper Shell",
     "cases", "accessory"),
    ("BMBEAR Bling Apple Watch Case Compatible with Apple Watch 38mm Series 3 2, Diamond "
     "Rhinestone Full Protective Cover", "cases", "accessory"),
    ("SWISS PART Watch Stand Display Charging Universal 2 in 1 Desktop Dock Station Band "
     "Bracket Holder Compatible for Apple iWatch iPhone ipad Tablets Grey", "mounts", "accessory"),
    # ... and a cleaner is never the device it cleans (real catalog titles, some shortened:
    # each read the device it names, headphones / cameras / audio / tv_video / computers)
    ("6 in 1 Keyboard Cleaner Kit Earphone Cleaner Kit with Brush Spong Pen Nibs for "
     "Electronic Cleaning Cellphone Hole Earbuds Monitor (White)", "tools", "accessory"),
    ("Bluetooth Earbuds Cleaning Pen 3 in 1 Cleaner Kit Earphone Brush, Multi-Function "
     "Portable Soft Cleaning Brush Pen Tools for Airpods Pro", "tools", "accessory"),
    ("Charm &Magic Professional Camera Cleaning Set with Air Blower Cleaner Lens Cleaning Pen "
     "Cleaning Cloths for Cleaning Your Camera Lens", "tools", "accessory"),
    ("LituFoto K9 Camera Air Blowers, Lens Cleaning Blaster Dust Cleaner Removes Dust and Dirt "
     "from DSLR Camera", "tools", "accessory"),
    ("VSGO DDG-1 Camera Cleaning Gloves Professional Anti-Static Cleaning Gloves White "
     "Dust-Free Compatible for Camera", "tools", "accessory"),
    ("Vinyl Record Cleaner Carbon Fiber Brush Turntables LP Deep Cleaning Roller Professional "
     "and Effective Vinyl Cleaning Solution", "tools", "accessory"),
    ("Scotch High Performance VCR Head Cleaner (Pack of 5)", "tools", "accessory"),
    ("DIDnDID Mini Desktop Vacuum Cleaner Portable USB Vacuum Cleaner for Desktop", "tools",
     "accessory"),
    ("Oductsen Computer Vacuum Cleaner Air Duster 2-in-1, Cordless Mini Keyboard Vacuum",
     "tools", "accessory"),
    ("vershine Mini Computer Vacuum Keyboard Cleaner PC Laptop Brush Dust Cleaning Kit "
     "Computer Vacuums", "tools", "accessory"),
    # ... while a device bundled WITH a cleaning kit, a vacuum-tube amplifier, an ear-wax
    # camera and a camcorder's "wipe" stay devices
    ("Canon PowerShot SX730 HS (Black) Camera, Camera Case, Lexar 16GB UHS, Memory Card "
     "Wallet, Ritz Gear Tripod, Cleaning Kit", "cameras", "device"),
    ("SAMSUNG Galaxy Tab A Lite 7.0-inch 8GB Tablet PC Wi-Fi, White Bundle with 16GB microSD "
     "Memory Card, 3 Stylus Pens, in-Ear Headphone, Hardshell Case and Cleaning Kit", "tablets",
     "device"),
    ("Beats Solo3 Wireless On-Ear Headphones - Class 1 Bluetooth Headphones, Built-in "
     "Microphone with Bonus Cleaning Cloth - Red", "headphones", "device"),
    ("LeTkingok Mini Vacuum Tube Amplifier- HiFi Desktop Power Amp Valve Amplifier 3W2 (Black)",
     "audio", "device"),
    ("ScopeAround Otoscope Ear Camera with 4.5\" IPS Screen, 1920x1080 FHD Smart Visual Ear "
     "Cleaner with Camera Tool Kit", "cameras", "device"),
    ("Panasonic 4K video camera WX970M wipe takes lightweight 447g black HC-WX970M-K", "cameras",
     "device"),
    # only a DEVICE category moves: a cleaner an accessory rule reads keeps that category
    ("VSGO Professional Lens Cleaning Pen with Soft Lens Brush & Nano Optical Carbon Compatible "
     "for Rift S/HTC Vive/Cosmos Headset, Drone, Cameras Lens", "camera_accessories",
     "accessory"),
    # an AirPods case holding a watch strap is no watch storage (watch_kind storage)
    ("Fivemix Ash Python Shaped Leather Airpods Pro Case - Shockproof Case with Black "
     "Accessories - Pair of EarTips - Airpod Watch Strap Holder - Magnetic Strip Airpod "
     "Holder - Pair of Ear Hooks and Brush", "cases", "accessory"),
    # 2026-10-30 F2 (the Phase C live run's residue, real catalog titles). A strap that
    # names a camera and no watch is a camera strap (the bare "strap" read
    # watch_accessories: 3 on page 1 of "gift for a photographer"); a strap naming no
    # camera, or a band whose title names a watch, keeps the watch accessories
    ("Eorefo Camera Strap Leather Universal Shoulder Neck Belt Strap for Mirrorless and DSLR "
     "Camera,Flower Red.", "camera_accessories", "accessory"),
    ("NEEWER Faux Leather Hand Wrist Pad Strap for DSLR Camera (EA-01W)", "camera_accessories",
     "accessory"),
    ("CHMETE Camera Neck Shoulder Belt Strap (Neoprene x2)", "camera_accessories", "accessory"),
    ("Silicone Strap for Xiaomi Mi Band 4", "watch_accessories", "accessory"),
    ("Replacement Band for Kids Smart Watch with Camera", "watch_accessories", "accessory"),
    # ... and a case listed like its tablet is the case (the kids-tablet case was a core
    # tablet on "tablet for kids", page 2), while a tablet that lists its specs or its
    # Kid-Proof Case, or comes WITH a cover, stays the tablet
    ('Armorex Dragon Touch Alldaymall NeuTab Chromo Inc 7" Kids Tablet, Durable [Shockproof] '
     "[Impact Resistant] [Kids Safe] Rugged Silicone Cover (Hot Pink)", "cases", "accessory"),
    ("Elegant Choise Amazon Kindle Fire 8 Tablet (7th 8th Generation, 2017 2018 Release), "
     "Kindle Fire HD 8 Case, Heavy Duty Full Body Protection Shockproof Kickstand Case "
     "Cover(Red)", "cases", "accessory"),
    ("Touchfire iPad 2, 3, 4 Keyboard Storage Case", "cases", "accessory"),
    ("HiGrace Kids Tablet 10 inch, Android 12 Quad core Tablet for Kids (Ages 3-12), 5000mAh, "
     "32GB ROM, Dual Camera, WiFi, Parental Control, Kid-Proof Case-Blue", "tablets", "device"),
    ("Samsung Galaxy Tab A SM-T550 9.7-Inch Tablet (16 GB, Smoky-Titanium) W/Book Cover "
     "(Certified Refurbished)", "tablets", "device"),
    ("Fire 7 Kids Tablet, 7 inch Display, Kid-Proof Case", "tablets", "device"),
    ("Kids Tablet 7 inch with Silicone Cover", "tablets", "device"),
]
# The 2026-10-27 held-out read's error classes, PINNED as read today (the right answer
# in the comment): a rule fix moves them on purpose, with this table.
ELECTRONICS_KNOWN_MISREADS = [
    # product-line-only title, read off a part (computers)
    ("Dell Inspiron 15 5000 15.6in Touchscreen Trulife LED-Backlit FHD i5570-7487SLV-PUS Intel "
     "8th Gen i7-8550U 12GB 2TB HDD", "storage"),
    # "&" is no cut (cameras)
    ("Nikon D3200 Digital SLR Camera & 18-55mm G VR DX AF-S Zoom Lens (Black)",
     "camera_accessories"),
    # stray words: "Skin Care" (no category), "Cable Box" (tv_video), "HDTV Amplifier" (antenna)
    ("Firefly DE300 Digital Dermatoscope for Telehealth, Telemedicine, Dermatology, "
     "Trichology, and Skin Care.", "cases"),
    ("Cable Box", "cables"),
    ("TV Antenna-2022 Updated Amplified HD Digital TV Antenna-Long Range 200+ Miles "
     "Reception-Powerful HDTV Amplifier", "audio"),
    # a heart-rate chest strap: the category rules said "watches", the watch reader does not
    ("Bryton Rider 320E Heart Rate Monitor Bundle, GPS Cycle Computer", "unknown"),
    # a cleaner whose device list runs on past a comma after "for": the list's later parts
    # are no target segments, so "Tablets" decides there (tools; 2026-10-30 R2, the one
    # catalog title of its kind; its head names no tablet, so it is never core)
    ("am Get Clean Refill (200 ml) Refill Liquid for All refillable am Get Clean Products, "
     "Cleaning Spray for Phones, Tablets and Laptops, Portable and Eco-Friendly", "tablets"),
    # the camera-strap rule (2026-10-30 F2) reads the DECIDING segment's strap: a camera bag
    # named before its straps reads camera_accessories (bags), and a polishing cloth that
    # lists a camera lens keeps the watch accessories' "polishing cloth" (tools)
    ("Modovo Canvas Leisure DSLR SLR Camera Bag Messenger Bag Adjustable Bottom Straps for "
     "Tripod Removable Inside Pad (No Tripod Included), Brown", "camera_accessories"),
    ("MOSISO Polishing Cloth, 2PCS Soft Microfiber Cloth Electronic Screen Cleaning Cloths "
     "Compatible with iPhone,Compatible with MacBook,Compatible with iPad,Tablets,Laptop "
     "Notebook Computer,Camera Lens", "watch_accessories"),
]
GROCERY_TITLES = [
    ("Sugar Free Creamy Chocolate Coffee-Mate Coffee Creamer 10.2 OZ", "coffee"),
    ("Vonbee Premium Hibiscus Tea Piramid Tea Bags, 60 Count", "tea"),
    ("Woodford Reserve Spiced Cherry Bourbon Barrel Aged Cocktail Bitters & Bourbon Cherries "
     "2 Pack", "beverages"),
    ("Snak Club Energizer Trail Mix, 24 Ounce", "snacks"),
    ("Bellino - Italian Dark Chocolate Egg, (2)- 5.3 oz. Pkgs.", "candy"),
    ("LARDER BAKING COMPANY Country Boule French Bread", "bakery"),
    ("Carbon's Golden Malted Pancake & Waffle Flour Mix, Original, 32-Ounces (Pack of 2)",
     "baking"),
    ("Fusion Black Truffle Sea Salt, 4.0-Ounce Jar", "spices"),
    ("Hickory Farm Horseradish Sauce", "sauces"),
    ("Barilla Spaghetti Pasta, 16 oz", "pantry"),
    ("Kellogg's Winter Variety Pop Tarts (58.2 oz, 32 ct.)", "breakfast"),
    ("Campbell's Chunky, Creamy Chicken Noodle Soup, 540ml (Imported from Canada)", "canned"),
    ("Wonderful Pistachios, Roasted & Salted, 16 oz", "nuts"),
    ("Optimum Nutrition Gold Standard 100% Whey Protein Powder", "protein"),
    ("Fresh Strawberries by La Chatelaine Farms Per Pound", "fresh"),
    ("Torie & Howard Organic Hard Candy 5 Tin Gift Pack", "gifts"),
    ("Klionjor No. 2 Rhinestone Metal Plate Cake Topper 2nd Anniversary", "cake_decor"),
    ("Stainless Steel Tea Infuser for Loose Leaf Tea", "non_food"),
    ("Cold Brew Coffee Plus Collagen Protein", "coffee"),   # "plus" stays a Grocery cut
    ("Mystery Brand Original, 12 oz", "unknown"),
    # 2026-10-30: "press" singular (the rules read only "presse(s)"): kitchen tools
    ("Cookie Press Kit with 20 Discs", "non_food"),
    ("Coffee Press 34 oz", "non_food"),
    ("Cookie Presses Set", "non_food"),
]
GROCERY_KNOWN_MISREADS = [
    ("Cadbury Dairy Milk Fruit & Nut 180G (Pack of 2)", "nuts"),          # candy: a flavour word
    ("Gluten Free Apple Cinnamon Oven Baked Bars", "spices"),             # snacks: weak "bars"
    ("My T Fine Pudding, Chocolate, 3.125-Ounce (Pack of 12)", "candy"),  # a neighbour
    ("Dean Jacob's Habanero Seasoning Glass Grinder Mill ~ 2.2 oz.", "non_food"),  # spices
]


def _frame_of(rows, vertical):
    return pd.DataFrame({"bsns_vrtcl_name": [vertical] * len(rows),
                         "Product_title": [r[0] for r in rows]})


def test_electronics_title_table_any_case():
    assert {c for _t, c, _r in ELECTRONICS_TITLES} == set(ELECTRONICS_KEYS) | {"unknown"}
    for title, category, role in ELECTRONICS_TITLES:
        for shape in (title, title.lower(), title.upper()):
            assert (ct.electronics_category(shape) or "unknown") == category, (shape, category)
    df = ct.annotate_catalog(_frame_of(ELECTRONICS_TITLES, "Electronics"))
    assert df["item_category"].tolist() == [c for _t, c, _r in ELECTRONICS_TITLES]
    assert df["item_role"].tolist() == [r for _t, _c, r in ELECTRONICS_TITLES]


def test_grocery_title_table_any_case():
    assert {c for _t, c in GROCERY_TITLES} == set(GROCERY_KEYS) | {"unknown"}
    df = ct.annotate_catalog(_frame_of(GROCERY_TITLES, "grocery"))   # a lowercase vertical
    assert df["item_category"].tolist() == [c for _t, c in GROCERY_TITLES]
    for title, category in GROCERY_TITLES:
        assert (ct.grocery_category(title.lower()) or "unknown") == category, title
        role = df.loc[df["Product_title"] == title, "item_role"].iloc[0]
        assert role == (None if category == "unknown" else ct._GROCERY_TABLE[category][1])


def test_known_misreads_stay_pinned():
    for title, today in ELECTRONICS_KNOWN_MISREADS:
        assert (ct.electronics_category(title) or "unknown") == today, title
    for title, today in GROCERY_KNOWN_MISREADS:
        assert (ct.grocery_category(title) or "unknown") == today, title
    for title, today in WATCH_KNOWN_MISREADS:
        assert ct.watch_kind(title) == today, title
        assert ct.electronics_category(title) == "watches", title


def test_fast_reading_equals_the_exhaustive_one_on_the_tables():
    for tax, rows in ((ct.ELECTRONICS_TAXONOMY, ELECTRONICS_TITLES + ELECTRONICS_KNOWN_MISREADS),
                      (ct.GROCERY_TAXONOMY, GROCERY_TITLES + GROCERY_KNOWN_MISREADS)):
        for row in rows:
            t = ct._norm(row[0])
            assert tax.classify_text(t)[0] == tax.classify_text_ref(t), row[0]
            assert tax.classify_text(t)[0] == tax.classify_text_all(t), row[0]


_CLEANERS = [t for t, c, _r in ELECTRONICS_TITLES
             if c == "tools" and ("clean" in t.lower() or "vacuum" in t.lower())]


def test_a_cleaner_is_never_the_device_it_cleans():
    """The category RULES read the device a cleaner names (the right-most device noun of
    the deciding segment); _electronics_read files the row with the tools when THAT
    segment sells a cleaner, so a cleaning kit bundled after a device keeps the device.
    The substring hints are only a prefilter: every _CLEANER_PRODUCT match holds one."""
    tax = ct.ELECTRONICS_TAXONOMY
    assert len(_CLEANERS) >= 10
    for title in _CLEANERS:
        t = ct._norm(title)
        assert tax.role(tax.classify_text(t)[0]) == "device", title    # what the rules read
        assert ct._electronics_read(t)[0] == "tools", title
    rows = [r[0] for r in ELECTRONICS_TITLES + ELECTRONICS_KNOWN_MISREADS]
    with patched(ct, {"_CLEANER_HINT": ("",)}):
        relaxed = [ct.electronics_category(t) for t in rows]
    assert relaxed == [ct.electronics_category(t) for t in rows]
    for phrase in ("cleaner kit", "cleaning pen", "vhs head cleaner", "camera air blowers",
                   "air duster", "compressed air", "desktop vacuum", "cleaning gloves",
                   "deep cleaning roller", "mini keyboard vacuum"):
        found = ct._CLEANER_PRODUCT.search(phrase)
        assert found and any(h in found.group() for h in ct._CLEANER_HINT), phrase
    for phrase in ("mini vacuum tube amplifier", "smart visual ear cleaner", "ear cleaning "
                   "endoscope", "wipe takes", "& dust brush", "easy to clean", "clean cloth"):
        assert not ct._CLEANER_PRODUCT.search(phrase), phrase


def test_watch_storage_and_smartwatch_cases_follow_the_watch_reader():
    """Electronics: the rules' right-most container noun filed a watch box as a phone-style
    case, and a smartwatch's protective "watch case" with the watch accessories; the watch
    reader's kind (storage / protection) decides both (2026-10-30 R2). Only a container
    category moves: an AirPods case that holds a watch strap reads storage but names no
    watch storage, and a watch stand stays a stand."""
    tax = ct.ELECTRONICS_TAXONOMY
    for title, rules, kind, filed in (
            ("Watch Box 12 Slot Display Case", "cases", "storage", "watch_accessories"),
            ("Watch Storage Case for 10 Watches", "cases", "storage", "watch_accessories"),
            ("HoneyGirl Watch Case for Fitbit Versa 3, Scratch-Proof Cover Protective Bumper "
             "Shell", "watch_accessories", "protection", "cases"),
            ("Fivemix Ash Python Shaped Leather Airpods Pro Case - Shockproof Case with Black "
             "Accessories - Pair of EarTips - Airpod Watch Strap Holder - Magnetic Strip "
             "Airpod Holder - Pair of Ear Hooks and Brush", "cases", "storage", "cases"),
            ("Kollea Aluminum Apple Watch Stand Charging Station Bracket Docking Cradle for "
             "All Apple Watch, Carbon Bamboo", "mounts", "charger", "mounts")):
        t = ct._norm(title)
        assert tax.classify_text(t)[0] == rules, title
        assert ct._electronics_read(t) == (filed, kind), (title, ct._electronics_read(t))
    df = ct.annotate_catalog(pd.DataFrame({
        "bsns_vrtcl_name": ["Electronics", "Fashion"],
        "Product_title": ["Watch Box 12 Slot Display Case"] * 2}))
    assert df["item_category"].tolist() == ["watch_accessories", None]
    assert df["item_role"].tolist() == ["accessory", None]
    assert df["watch_kind"].tolist() == ["storage", "storage"]


def test_a_strap_naming_a_camera_and_no_watch_is_a_camera_strap():
    """The watch accessories rule's bare "strap" is the right-most noun of a camera strap's
    head ("... Shoulder Neck Belt Strap"); _electronics_read files the row with the camera
    accessories when the title names a camera, the watch reader finds no watch and the
    DECIDING segment holds the strap (2026-10-30 F2: 36 catalog rows, 3 of them on page 1
    of "gift for a photographer"). The rules themselves are untouched."""
    tax = ct.ELECTRONICS_TAXONOMY
    for title, filed in (
            ("Eorefo Camera Strap Leather Universal Shoulder Neck Belt Strap for Mirrorless and "
             "DSLR Camera,Flower Red.", "camera_accessories"),
            # the camera named only after "for"
            ("NEEWER Faux Leather Hand Wrist Pad Strap for DSLR Camera (EA-01W)",
             "camera_accessories"),
            ("Silicone Strap for Xiaomi Mi Band 4", "watch_accessories"),           # no camera
            ("Replacement Band for Kids Smart Watch with Camera", "watch_accessories"),  # a watch
            # a camera named, but the deciding segment sells a cloth, not a strap
            ("MOSISO Polishing Cloth, 2PCS Soft Microfiber Cloth Electronic Screen Cleaning "
             "Cloths Compatible with iPhone,Compatible with MacBook,Compatible with iPad,"
             "Tablets,Laptop Notebook Computer,Camera Lens", "watch_accessories"),
            ("Microfiber Polishing Cloths 2 Pack, Strap Safe, for DSLR Camera Lens",
             "watch_accessories")):
        t = ct._norm(title)
        assert tax.classify_text(t)[0] == "watch_accessories", title    # what the rules read
        assert ct._electronics_read(t)[0] == filed, (title, ct._electronics_read(t))
    for phrase in ("camera", "dslr cameras", "camcorder", "mirrorless", "gopro", "slr"):
        assert ct._CAMERA_NAMED.search(phrase), phrase
    for phrase in ("cam buckle strap", "webcam", "camo strap", "cameraman"):
        assert not ct._CAMERA_NAMED.search(phrase), phrase
    df = ct.annotate_catalog(pd.DataFrame({
        "bsns_vrtcl_name": ["Electronics", "Fashion"],
        "Product_title": ["CHMETE Camera Neck Shoulder Belt Strap (Neoprene x2)"] * 2}))
    assert df["item_category"].tolist() == ["camera_accessories", None]
    assert df["item_role"].tolist() == ["accessory", None]
    assert df["watch_kind"].tolist() == [None, None]


def test_a_tablet_case_listed_like_its_tablet_is_a_case():
    """A tablets row whose plain segments after the deciding one end on a case or cover is
    that case, unless the title lists the device's specs or its Kid-Proof Case; a bundle
    "with" a cover, a later non-case noun and every other device category keep the device
    (2026-10-30 F2: 21 catalog rows, the kids-tablet case a core tablet on "tablet for
    kids")."""
    tax = ct.ELECTRONICS_TAXONOMY
    for title, rules, filed in (
            ('Armorex Dragon Touch Alldaymall NeuTab Chromo Inc 7" Kids Tablet, Durable '
             "[Shockproof] [Impact Resistant] [Kids Safe] Rugged Silicone Cover (Hot Pink)",
             "tablets", "cases"),
            ("Touchfire iPad 2, 3, 4 Keyboard Storage Case", "tablets", "cases"),
            ("Kids Tablet 7 inch, Rugged Silicone Cover", "tablets", "cases"),
            ("HiGrace Kids Tablet 10 inch, Android 12 Quad core Tablet for Kids (Ages 3-12), "
             "5000mAh, 32GB ROM, Dual Camera, WiFi, Parental Control, Kid-Proof Case-Blue",
             "tablets", "tablets"),
            ("Fire 7 Kids Tablet, 7 inch Display, Kid-Proof Case", "tablets", "tablets"),
            ("Kids Tablet 7 inch with Silicone Cover", "tablets", "tablets"),
            ("Kids Tablet 7 inch, Compatible with Rugged Case", "tablets", "tablets"),
            ("Kids Tablet, Rugged Silicone Cover, Stylus Pen", "tablets", "tablets"),
            ("Wireless Earbuds, Touch Control, Charging Case", "headphones", "headphones"),
            ("Instant Camera, Protective Case", "cameras", "cameras")):
        t = ct._norm(title)
        assert tax.classify_text(t)[0] == rules, title                  # what the rules read
        assert ct._electronics_read(t)[0] == filed, (title, ct._electronics_read(t))
    for phrase in ("32GB", "2 GB", "4GB RAM", "5000mAh", "Android 12", "Quad-Core", "quad core",
                   "1.8GHz", "WiFi", "Wi-Fi", "Bluetooth", "Bundle", "(Renewed)",
                   "Certified Refurbished", "Kid-Proof Case", "Kids-Proof Case"):
        assert ct._DEVICE_SPEC.search(ct._norm(phrase)), phrase
    for phrase in ('7" Kids Tablet', "Rugged Silicone Cover", "Auto Sleep/Wake", "10.1 inch",
                   "Kickstand Case Cover", "Kids Safe", "Kid-Proof Cover"):
        assert not ct._DEVICE_SPEC.search(ct._norm(phrase)), phrase


def test_a_band_before_a_named_smartwatch_is_the_band():
    """A band word right before a smartwatch maker's line name ("Leather Band Apple Watch",
    "for" left out) is the band for it, also with the connector glued on ("Strap-for
    Samsung Galaxy Watch"); a generic watch name, or watch signals with no accessory
    signal, keep the watch (2026-10-30 F2: 4 catalog rows, the Goton band a core watch on
    "apple watch band")."""
    for title, kind in (
            ("Goton Leather Band Apple Watch Series 4 3 2 1", "band"),
            ("Sport Loop iWatch Series 7", "band"),
            ("Leather Watch Band Strap-for Samsung Galaxy Watch 4", "band"),
            ("Silicone Band Garmin Forerunner 245", "band"),
            ("Leather Band Watch", "watch"),
            ("Mesh Band Smart Watch for Women", "watch"),
            ("Leather Band Apple Watch Series 3 GPS 38mm", "watch"),      # a watch signal
            ("Always On Display Galaxy Watch 4 Classic 46mm Smartwatch", "watch"),  # no band
            ("Apple Watch Series 8 GPS 41mm Midnight Aluminum Case with Sport Band", "watch")):
        assert ct.watch_kind(title) == kind, (title, ct.watch_kind(title))
    assert not ct.ends_on_a_watch("leather band apple watch")
    assert ct.ends_on_a_watch("apple watch")
    df = ct.annotate_catalog(pd.DataFrame({
        "bsns_vrtcl_name": ["Fashion", "Electronics"],
        "Product_title": ["Goton Leather Band Apple Watch Series 4 3 2 1"] * 2}))
    assert df["watch_kind"].tolist() == ["band", "band"]
    assert df["item_category"].tolist() == [None, "watch_accessories"]


def test_fast_reading_follows_the_unindexed_one_where_finditer_differs():
    """finditer skips a match overlapping an earlier one, so the finditer reference loses
    "lens mount adapter" to the shorter "mount adapter" (cables); the unindexed reading
    (every rule from every token start) is what the fast reading implements."""
    tax = ct.ELECTRONICS_TAXONOMY
    t = ct._norm("Fotodiox Pro Lens Mount Adapter - Miranda SLR Lens to Canon EOS")
    assert tax.classify_text(t)[0] == tax.classify_text_all(t) == "camera_accessories"
    assert tax.classify_text_ref(t) == "cables"


# ------------------------------------------------------------------ watches
WATCH_TITLES = [
    ("Vintage Automatic Mechanical Pocket Watch with Chain", "watch"),
    ("Smart Watch for Men with Bluetooth Call", "watch"),
    ("Kenneth Cole New York Mens Stainless Steel Bracelet Casual Watch", "watch"),
    ("Analog Quartz Watch Leather Band", "watch"),      # the watch's own band
    ("Women Watch with Gift Box", "watch"),             # "gift box" is packaging
    ("Fitbit Charge 5 Advanced Fitness Tracker", "watch"),
    ("Letsfit - Fitness Tracker with Heart Rate Monitor", "watch"),
    ("Leather Watch Band 20mm Quick Release", "band"),
    ("Wanme Bands for Amazfit Band 5 Replacement Strap", "band"),
    ("Leather Replacement Bracelet Strap Wristband for Apple Watch 38mm", "band"),
    ("Kalawen Watch Winder for Rolex", "storage"),      # the meeting's junk, first row
    ("JQUEEN Six Automatic Watch Winder", "storage"),
    ("Watch Box for Men 12 Slot", "storage"),
    ("Microfiber Polishing Cloth Compatible with Apple Watch", "care_tool"),
    ("Watch Repair Tool Kit, Watch Link Remover", "care_tool"),
    ("Screen Protector for Apple Watch Series 7 45mm", "protection"),
    ("Charger for Apple Watch, Magnetic Charging Cable", "charger"),
    ("Watch Battery Replacement CR2032 for Watches", "part"),
    ("Pocket Watch Chain Fob", "other"),
    # 2026-10-27 reader fixes: a protection plan, a list after "for", verbs, golf GPS units
    ("Allstate 3-Year Watch Protection Plan", "other"),
    ("TALK WORKS Lightning Cable for Apple iPhone 13, iPad, AirPods, Watch", "other"),
    ("Canary Pro Indoor Home Security Camera | 24/7 Watch Live", None),
    ("Garmin Approach G80 Handheld Golf GPS", None),
    ("RCA Lyra 512MB Sport MP3 Player with Pedometer/Stopwatch", None),   # a feature word
    ("Huami Amazfit Charger", None),
    ("Fitbit Flyer", None),
    ("Yo-kai Watch 2", None),
    ("Bird Watching Binoculars", None),
    ("Knit Watch Cap Beanie", None),
    ("Cord Winder for Headphones", None),
    ("Weight Watchers Snack Bars", None),
    ("Red Dress", None),
    # 2026-10-30 review: a tool kit, game masks, knit caps, a case two-pack and Grocery
    # brands read as the watch itself (B1 then seated the kit as the gift on the meeting's
    # query). Real catalog titles, some shortened
    ("SE Tools 16 PCS Watch Tool Kit Plus Adjustable Watch Wrench All in One", "care_tool"),
    ("Watch Battery Replacement Tool Kit Watch Opener Watch Press", "care_tool"),
    ("Watch Dogs Wrench Mask with Rivet, Punk Devil Cosplay Game Legion Ghost Death PVC "
     "Hacker Masks for Halloween Party", None),
    ("Kuberas Watch Dogs Legion Winston Pig Mask King of Hearts Skeleton", None),
    ("Geyoga 6 Pieces Fleece Hats Soft Beanie Watch Caps Winter Helmet Liner", None),
    ("Home Prefer Men's Winter Outdoor Watch Hat Camo Knitted Fleece Lined Beanie", None),
    ("Women Men Winter Beanie Hat - White Cashmere Wool Acrylic Thick Knit Knitted Warm Watch "
     "Stocking Skull Knit Caps", None),
    # a protective case two-pack for a Garmin: storage until 2026-10-30 R2 ("watch case"
    # next to a smartwatch name is its protective case); a watch case without one stays
    # storage
    ("[2 Pack] TenCloud Tactix 7 Watch Cases Accessories Compatible with Garmin tactix "
     "7/tactix 7 Pro Case (No Screen Protector)", "protection"),
    ("Watch Case for Men 6 Slots", "storage"),
    # 2026-10-30 R2: the reviewers' residue. Smartwatch "watch cases" (real catalog
    # titles, shortened) read storage; a box / storage case for watches stays storage, as
    # does a case named before glass and a tray, and a bare size is no smartwatch
    ("HoneyGirl Watch Case for Fitbit Versa 3, Scratch-Proof Cover Protective Bumper Shell",
     "protection"),
    ("Apple Watch Case Series 2 Series 1, BESTeck Rugged Protective Case with Strap Bands",
     "protection"),
    ("Watch Box 12 Slot Display Case", "storage"),
    ("Watch Storage Case for 10 Watches", "storage"),
    ("LEO LION Men's Jewelry Box Organizer - Watch and Jewelry Box for Men - Wooden Watch "
     "Case with Glass Top and Jewelry Tray Below", "storage"),
    ("Watch Case 40mm Stainless Steel Replacement for SKX007", "storage"),
    # ... a strap's hole punch, a slap bracelet and a multitool adapter read as watches
    # (B1's judged core samples); a slap WATCH is one, and a charger adapter a charger
    ("NEW REVOLVING LEATHER PUNCH WATCH BELT HOLE PUNCH RED", "care_tool"),
    ("Baban Revolving Punch Plier Kit, Leather hole punch set for Belts, Watch Bands, Paper, "
     "Plastic, Fabric", "care_tool"),
    ("1Pc Women's Feather Slap Bracelet Watch Patting Circle Slap Ring Wrist Decoration Cuff "
     "Sleeves Slap Bangle Hair Accessories", "other"),
    ("Tocosy Slap on Watch Silicone Quartz Number Sport Wrist Band for Children Girl Lady Women "
     "Men Boy Unisex Slap Kids Watches", "watch"),
    ("Bundle Monster 12 Piece Jelly Silicone Slap Watch Mixed Color Set with Interchangable "
     "Faces - Dark Colors", "watch"),
    ("BestTechTool watch adapter compatible with LEATHERMAN TREAD LT - BTT adapter (compatible "
     "with Garmin Fenix5X-Fenix 3 (see description), Black, TREAD LT)", "part"),
    ("Itramax Charger Adapter for Garmin Watch 3Pack,Type-C USB Cable Charging for Fenix 7",
     "charger"),
    ("Sea Watch Premium New England Clam Chowder", None),
    ("Aerial Resupply Coffee Fire Watch Ground Coffee Medium Roast", None),
    ("Yokai-watch and I make the specter 3D Gummy!", None),
    ("Lay’s 2020 The Voice Taste Watch Win Pick Your Favorite Flavor Chips!", None),
    # known residuals: steampunk jewellery made of movements reads as a part
    ("Steampunk Watch Movements Ring", "part"),
    # 2026-10-30 F2: a band named right before a smartwatch maker's line name is the band
    # ("for" left out; the Goton band, a Fashion row, was a core watch on "apple watch
    # band"), the Bepack band a known misread until then; a generic watch name, or watch
    # signals with no accessory signal, keep the watch
    ("Goton Leather Band Apple Watch Series 4 3 2 1, Retro Leather Watch Strap Compatible "
     "Apple Watch Band 40mm 38mm Leather Men Women Stainless Steel Buckles (Dark Brown, "
     "40mm/38mm)", "band"),
    ("Ecute Quick Release-22mm Width Design Leather Watch Band Strap-for Samsung Galaxy Watch "
     "46mm/Gear S3 Frontier/Classic Smart Watch - Vintage Color Rose", "band"),
    ("Bepack Bands Fitbit Charge 2, Soft TPU Durable Adjustable Replacement Sport Strap Band "
     "Fitbit Charge 2 Smart Watch Heart Rate Fitness Wristband", "band"),
    ("Nylon Strap Samsung Galaxy Watch 4", "band"),
    ("Leather Band Watch", "watch"),
    ("Mesh Band Smart Watch for Women", "watch"),
    ("Leather Band Apple Watch Series 3 GPS 38mm", "watch"),
    ("Winnie The Pooh Unisex Easy Reader Quartz Analog 38mm Leather Strap Watch", "watch"),
]
# Watch-reader error classes PINNED as read today (the right kind in the comment), from
# the 2026-10-30 review's catalog scan: the first segment names a Fitbit and nothing
# after it, so the accessory sold for it reads as the device. No rule fix: deferring a
# model-only first segment to a later accessory also flipped two real Garmin Forerunner
# bundles. All three are Electronics "watches" rows (one source) and so core for B1 (a
# fourth, the Bepack band named right before the Fitbit, reads band since 2026-10-30 F2).
WATCH_KNOWN_MISREADS = [
    ("Fitbit Charge, Charging Cable", "watch"),                                    # charger
    ("Fitbit Ionic Case,Fitian Replacement Shock-proof and Shatter-resistant TPU Soft "
     "Accessory Rugged Protective Case Frame Cover Shell for Fitbit Ionic", "watch"),  # protection
    ("Fitbit Zip Clip By XXSCY®, 2pcs Replacement Clip for Fitbit Zip Only (No tracker) "
     "(Black&Black)", "watch"),                                                    # part
]


def test_watch_kind_table():
    for title, kind in WATCH_TITLES:
        assert ct.watch_kind(title) == kind, (title, ct.watch_kind(title), kind)
        assert ct.watch_kind(title.upper()) == kind, title
    assert {k for _t, k in WATCH_TITLES} == set(ct.WATCH_KINDS) | {None}
    for odd in (None, float("nan"), pd.NA, "", 42):
        assert ct.watch_kind(odd) is None


def test_ends_on_a_watch():
    """The phrase after 'for' in a jewellery ask (product_type: "bracelet for apple watch"
    asks for a band) is the watch only when it names a watch LAST."""
    for phrase in ("apple watch", "my fitbit charge 5", "galaxy watch 4 44mm", "iwatch se",
                   "apple watch series 7 45mm", "garmin forerunner 245", "smartwatch",
                   "galaxy watch 4 classic", "watches", "apple watch 41 mm",
                   "apple watch 38/40mm", "apple watch se 40mm gps", "apple watch ultra 2 49mm",
                   "apple watch 4.5mm "):
        assert ct.ends_on_a_watch(phrase), phrase
    for phrase in ("my watch-loving dad", "watch lovers", "a watch collector", "watch band",
                   "watch winder", "dad", "", "bird watching", "huami amazfit charger",
                   "apple watch 38/40mm lovers", "apple watch series 7 fans"):
        assert not ct.ends_on_a_watch(phrase), phrase
    # One reading per size token (2026-10-30): the first draft's two numeric readings of
    # "45mm" / "45 mm" backtracked exponentially on a run of sizes before another word
    # (22 sizes: ~6 s; this takes well under a millisecond)
    for size in ("45mm", "45 mm"):
        started = time.perf_counter()
        assert not ct.ends_on_a_watch("apple watch " + f"{size} " * 22 + "lovers")
        assert ct.ends_on_a_watch("apple watch" + f" {size}" * 22)
        assert time.perf_counter() - started < 1.0, size


def test_watch_kind_is_read_on_every_vertical():
    df = pd.DataFrame({"bsns_vrtcl_name": ["Fashion", "Electronics", "Grocery", None],
                       "Product_title": ["Kalawen Watch Winder for Rolex"] * 3 + ["Wrist Watch"]})
    ct.annotate_catalog(df)
    assert df["watch_kind"].tolist() == ["storage", "storage", "storage", "watch"]
    assert df["item_category"].tolist() == [None, "watch_accessories", "unknown", None]


def test_electronics_watches_are_exactly_watch_kind_watch():
    titles = [t for t, _k in WATCH_TITLES] + [r[0] for r in ELECTRONICS_TITLES]
    df = ct.annotate_catalog(_frame_of([(t,) for t in titles], "Electronics"))
    assert ((df["item_category"] == "watches") == (df["watch_kind"] == "watch")).all()
    assert (df["item_category"] == "watches").sum() >= 6


# ------------------------------------------------------------------ facet readers
def test_diet_tags():
    assert ct.diet_tags("gluten free,organic,kosher") == ("gluten free", "organic", "kosher")
    # DIETARY_VALUES order, each once; case and hyphens folded; unknown words dropped
    assert ct.diet_tags("Vegan, Gluten-Free, gluten free, bogus") == ("gluten free", "vegan")
    assert ct.diet_tags("non-gmo") == ("non-gmo",) == ct.diet_tags("Non GMO")
    for empty in (None, float("nan"), pd.NA, "", "  ", "bogus"):
        assert ct.diet_tags(empty) == ()
    assert all(ct.diet_tags(v) == (v,) for v in ct.DIETARY_VALUES)


WORKS_WITH_TITLES = [
    ("Silicone Case for iPhone 12 Pro Max", ("apple",)),
    ("USB C Charger for Samsung Galaxy S21, iPhone 15, iPad Pro", ("apple", "samsung")),
    ("Remote for Fire TV Stick 4K", ("amazon",)),
    ("Kindle Paperwhite Cover", ("amazon",)),
    ("Controller Charging Dock for PS5 DualSense", ("playstation",)),
    ("Joy-Con Grips for Nintendo Switch OLED", ("nintendo",)),
    ("Laptop Battery for Dell Latitude E7450", ("pc",)),
    ("LP-E6 Battery for Canon EOS 5D", ("camera",)),
    ("Lens Cap for Sony A7 III", ("camera",)),
    ("Bands for Fitbit Versa 2 and Garmin Venu", ("fitness",)),
    ("Type Cover for Microsoft Surface Pro 7", ("microsoft",)),
    ("USB-C Hub for MacBook Pro, Surface, Dell XPS", ("apple", "microsoft", "pc")),
    ("Pixel 7 Screen Protector", ("google",)),
    ("Xbox Series X Cooling Fan", ("microsoft",)),
    ("4-Port Surface Mount Box", ()),             # not a Microsoft Surface
    ("Gaming Mouse Pad, Smooth Surface", ()),
    ("Universal Phone Mount", ()),
    ("Charger (made_by_asus)", ("pc",)),          # '_' separates tokens
]


def test_works_with_table():
    for title, keys in WORKS_WITH_TITLES:
        assert ct.works_with_text(ct._norm(title)) == keys, (title, keys)


def test_works_with_only_on_accessories_parts_and_consumables():
    rows = [("Silicone Case for iPhone 12",), ("Apple iPhone 12, 64GB, Black - Unlocked",),
            ("HP 62XL Black Ink Cartridge",), ("ASURION Protection Plan for iPhone",)]
    df = ct.annotate_catalog(_frame_of(rows, "Electronics"))
    assert df["item_role"].tolist() == ["accessory", "device", "consumable", "service"]
    assert df["works_with"].tolist() == [("apple",), (), ("pc",), ()]
    grocery = ct.annotate_catalog(_frame_of([("Apple Cider for iPhone fans",)], "Grocery"))
    assert grocery["works_with"].tolist() == [()]


CONDITION_TITLES = [
    ("Apple iPhone 8, 64GB, Gold - Fully Unlocked (Renewed)", "renewed"),
    ("HP EliteBook 840 G3 (Certified Refurbished)", "renewed"),
    ("Olympus 10MP Digital Camera (Red) Used - Good", "used"),
    ("Used Chromebook 11-3120", "used"),
    ("[Used] EOS 10D camera", "used"),
    ("Pre-Owned Nikon D750", "used"),
    ("Bitcoin miner ready to ship - used", "used"),
    ("Cable Ties used for fastening", None),          # a verb
    ("Antenna for attic, rv used, 65 miles", None),
    ("Can be used with a vintage pc", None),
    ("Brand New Laptop", None),
]


def test_condition_table():
    for title, condition in CONDITION_TITLES:
        assert ct.condition_text(ct._norm(title)) == condition, title
    df = ct.annotate_catalog(pd.DataFrame({
        "bsns_vrtcl_name": ["Electronics", "Fashion", "Electronics"],
        "Product_title": ["Apple iPhone 8 (Renewed)", "Renewed Denim Jacket",
                          "Protection Plan for Used/Refurbished Laptops"]}))
    assert df["item_condition"].tolist() == ["renewed", None, None]   # Electronics, no plans


def test_brand_reads_the_specs_only_for_chip_vendor_stores():
    specs = "Processor: 2.4 GHz core_i5; Chipset Brand: Intel; Brand: ASUS; Series: EEE PC 900"
    assert ct.brand_of("Electronics", "intel", specs) == "ASUS"
    assert ct.brand_of("Electronics", " NVIDIA ", "Brand: EVGA") == "EVGA"
    assert ct.brand_of("Electronics", "intel", "Chipset Brand: Intel; RAM: 8 GB") == "intel"
    assert ct.brand_of("Electronics", "intel", None) == "intel"
    assert ct.brand_of("Electronics", "Lenovo", "Brand: HP") == "Lenovo"   # not a chip store
    assert ct.brand_of("Grocery", "intel", "Brand: X") == "intel"           # Electronics only
    assert ct.brand_of("Electronics", None, "Brand: X") is None
    assert ct.brand_of("Electronics", float("nan"), "Brand: X") is None
    assert ct.brand_of("Electronics", "amd", "Brand: ; Series: x") == "amd"


# ------------------------------------------------------------------ the index
def test_token_keys():
    assert ct._token_key("case") == "case"
    assert ct._token_key("s21ultra") == "s0"
    assert ct._token_key("38mm") == "0"
    assert ct._token_key("ddr4") == "ddr0"
    assert ct._token_key("12") == "0"


def test_pattern_index_derivation():
    heads, firsts, maxlen = ct._pattern_index(r"\b(?:cases?|covers?)\b")
    assert heads.exact == {"case", "cases", "cover", "covers"} == firsts.exact and maxlen == 1
    heads, firsts, maxlen = ct._pattern_index(r"\bhead ?phones?\b")
    assert heads.exact == {"headphone", "headphones", "phone", "phones"}
    assert firsts.exact == {"headphone", "headphones", "head"} and maxlen == 2
    heads, firsts, maxlen = ct._pattern_index(r"\biphone(?: \d+)?\b")
    assert heads.exact == {"iphone", "0"} and firsts.exact == {"iphone"} and maxlen == 2
    assert ct._pattern_index(r"\bddr\d(?:l)?\b")[0].exact == {"ddr0"}
    # a \w run after letters is a prefix key; with nothing before it, any token
    heads = ct._pattern_index(r"\bgalaxy tab(?: [as]\d*\w*)?\b")[0]
    assert heads.prefixes == ("a", "s") and "tab" in heads.exact and not heads.anything
    assert heads.accepts("active") and heads.accepts("s7") and not heads.accepts("pro")
    assert ct._pattern_index(r"\bmodel \w+\b")[0].anything
    # an unbounded repeat reads as the cap; a lookaround narrows nothing here
    assert ct._pattern_index(r"\biphone(?: (?:pro|max))*\b")[2] == ct._CAP
    assert ct._pattern_index(r"\bcases?(?<!pc case)(?= for)\b")[0].exact == {"case", "cases"}
    # beyond the derivation: the rule runs from every token
    assert ct._pattern_index(r"(a)\1") is None
    rule = ct._Rule("cases", r"(case)\1")
    assert rule.heads.anything and rule.firsts.anything and rule.anchored


def test_every_rule_has_a_derived_index_and_few_are_anchored():
    """An anchored rule tries every token it can start on (slower): pinned, so a pattern
    edit that makes another rule anchored is noticed."""
    for tax in (ct.ELECTRONICS_TAXONOMY, ct.GROCERY_TAXONOMY):
        for r in tax.rules:
            assert not (r.heads.anything and r.firsts.anything), (tax.name, r.key)
    assert [(r.key, r.weak) for r in ct.ELECTRONICS_TAXONOMY.anchored] == [
        ("cameras", False), ("components", False)]
    assert [(r.key, r.weak) for r in ct.GROCERY_TAXONOMY.anchored] == [("fresh", True)]


_FROM_EVERY_POSITION = {}


def _from_every_position(rule):
    """The rule as a lookahead: one finditer reports its match from EVERY position (what
    rule.pattern.match(seg, pos) gives there), overlapping ones included."""
    look = _FROM_EVERY_POSITION.get(rule.pattern.pattern)
    if look is None:
        look = _FROM_EVERY_POSITION[rule.pattern.pattern] = re.compile(
            "(?=(" + rule.pattern.pattern + "))", rule.pattern.flags)
    return look


def _index_misses(tax, texts):
    """(rule, match) pairs the index would not find: a match ending on a token its heads
    refuse, starting on one its firsts refuse, or longer than its maxlen. Every match from
    every TOKEN START, as the fast reading tries them (2026-10-30 review: finditer alone
    skips overlapping matches -- 44,550 Electronics and 34,209 Grocery all-start matches
    on 2026-10-27's catalog -- which only a 3,000-title sample then checked)."""
    bad = []
    for t in texts:
        for seg, target in tax.iter_segments(t):
            if not seg:
                continue
            seg_tokens = ct._TOKEN_RE.findall(seg)
            if not seg_tokens:
                continue
            starts = set(ct._token_starts(seg, seg_tokens))
            for r in tax.rules:
                for m in _from_every_position(r).finditer(seg):
                    if m.start() not in starts:
                        continue
                    toks = ct._TOKEN_RE.findall(m.group(1))
                    if not toks:
                        continue
                    if not (r.heads.accepts(toks[-1]) and r.firsts.accepts(toks[0])
                            and len(toks) <= r.maxlen):
                        bad.append((tax.name, r.key, m.group(1)))
    return bad


def test_index_check_sees_overlapping_matches():
    """The soundness check must see a match finditer skips: "cable ties" overlaps "usb
    cable", and an index built for "usb cable" alone would hide it."""
    import types
    heads, firsts, maxlen = ct._pattern_index(r"\busb cable\b")
    rule = types.SimpleNamespace(key="cables", pattern=re.compile(r"\b(?:usb cable|cable ties?)\b"),
                                 heads=heads, firsts=firsts, maxlen=maxlen)
    tax = types.SimpleNamespace(name="probe", rules=[rule],
                                iter_segments=lambda t: iter([(t, False)]))
    assert [m.group() for m in rule.pattern.finditer("usb cable ties")] == ["usb cable"]
    assert _index_misses(tax, ["usb cable ties"]) == [("probe", "cables", "cable ties")]
    assert _index_misses(tax, ["usb cable"]) == []


def test_index_never_hides_a_match_on_the_tables():
    rows = ELECTRONICS_TITLES + ELECTRONICS_KNOWN_MISREADS + WATCH_TITLES
    assert not _index_misses(ct.ELECTRONICS_TAXONOMY, [ct._norm(r[0]) for r in rows])
    rows = GROCERY_TITLES + GROCERY_KNOWN_MISREADS
    assert not _index_misses(ct.GROCERY_TAXONOMY, [ct._norm(r[0]) for r in rows])


# ------------------------------------------------------------------ annotate_catalog
NAN = float("nan")


def _spark_frame(index=None):
    """A Databricks-shaped frame (toPandas: object dtype, None for missing), with a NaN
    and a pd.NA where a CSV or an Arrow column would put them, and every vertical."""
    rows = [  # vertical, title, dietary, store, specs
        ("Electronics", "Silicone Case for iPhone 12", None, "Spigen", None),
        ("Electronics", "Lenovo Yoga 7i 14 inch Laptop (Renewed)", None, "intel",
         "RAM: 16 GB; Chipset Brand: Intel; Brand: Lenovo"),
        ("Electronics", "Kalawen Watch Winder for Rolex", None, "KALAWEN", None),
        ("Electronics", "Garmin Forerunner 245 GPS Running Smartwatch", None, "Garmin", None),
        ("Grocery", "Organic Green Tea Bags, 100 Count", "organic,gluten free", "Numi", None),
        ("Grocery", "Mystery Brand Original, 12 oz", NAN, NAN, NAN),
        ("Fashion", "Kenneth Cole Mens Bracelet Watch", None, "Kenneth Cole", None),
        ("Fashion", "Red Dress", pd.NA, None, None),
        (None, "Wrist Watch", None, None, None),
        ("electronics", None, None, "amd", "Brand: MSI"),
    ]
    df = pd.DataFrame(rows, columns=["bsns_vrtcl_name", "Product_title", "dietary", "store",
                                     "specs"], dtype=object)
    for col in ("prod_description", "color", "material", "occasion"):
        df[col] = None
    if index is not None:
        df.index = index
    return df


SPARK_EXPECTED = {
    "item_category": ["cases", "computers", "watch_accessories", "watches", "tea", "unknown",
                      None, None, None, "unknown"],
    "item_role": ["accessory", "device", "accessory", "device", "food", None, None, None,
                  None, None],
    "watch_kind": [None, None, "storage", "watch", None, None, "watch", None, "watch", None],
    "diet_tags": [(), (), (), (), ("gluten free", "organic"), (), (), (), (), ()],
    "works_with": [("apple",), (), (), (), (), (), (), (), (), ()],
    "item_condition": [None, "renewed", None, None, None, None, None, None, None, None],
    "brand": ["Spigen", "Lenovo", "KALAWEN", "Garmin", "Numi", None, "Kenneth Cole", None,
              None, "MSI"],
}


def test_annotate_catalog_on_a_databricks_shaped_frame():
    df = _spark_frame()
    out = ct.annotate_catalog(df)
    assert out is df
    assert tuple(df.columns[-7:]) == ct.ADDED_COLUMNS
    for column, want in SPARK_EXPECTED.items():
        assert df[column].tolist() == want, (column, df[column].tolist())
        assert df[column].dtype == object, column
    for column in ("item_category", "item_role", "watch_kind", "item_condition", "brand"):
        assert all(v is None or isinstance(v, str) for v in df[column]), column
    for column in ("diet_tags", "works_with"):
        assert all(isinstance(v, tuple) for v in df[column]), column
    row = df.iloc[5].to_dict()                        # never NaN in a row dict
    assert row["item_role"] is None and row["brand"] is None and row["diet_tags"] == ()
    assert set(df["item_category"].dropna()) <= set(ct.CATEGORY_LABELS) | {"unknown"}


def test_annotate_catalog_never_moves_or_edits_a_row():
    index = pd.Index([40, 3, 17, 9, 0, 21, 5, 8, 1, 77])
    df = _spark_frame(index=index)
    before = df.copy()
    ct.annotate_catalog(df)
    assert df.index.equals(index) and len(df) == len(before)
    pd.testing.assert_frame_equal(df[before.columns], before)    # sources untouched
    for column, want in SPARK_EXPECTED.items():
        assert df[column].tolist() == want, column              # by position, not label
    again = ct.annotate_catalog(df.copy())
    pd.testing.assert_frame_equal(again, df)                     # idempotent


def test_annotate_catalog_reads_pandas_string_dtype():
    """read_csv gives pandas' str dtype with NaN; the local frame must read the same."""
    df = _spark_frame().astype({"bsns_vrtcl_name": "str", "Product_title": "str",
                                "store": "str", "specs": "str", "dietary": "str"})
    ct.annotate_catalog(df)
    for column, want in SPARK_EXPECTED.items():
        assert df[column].tolist() == want, column


def test_annotate_catalog_degenerate_frames():
    empty = ct.annotate_catalog(pd.DataFrame(columns=["bsns_vrtcl_name", "Product_title"]))
    assert tuple(empty.columns[-7:]) == ct.ADDED_COLUMNS and len(empty) == 0
    no_vertical = ct.annotate_catalog(pd.DataFrame({"Product_title": ["Wrist Watch"]}))
    assert no_vertical["item_category"].tolist() == [None]
    assert no_vertical["watch_kind"].tolist() == ["watch"]
    assert no_vertical["brand"].tolist() == [None] and no_vertical["diet_tags"].tolist() == [()]
    no_title = ct.annotate_catalog(pd.DataFrame({"bsns_vrtcl_name": ["Electronics"]}))
    assert no_title["item_category"].tolist() == ["unknown"]
    assert no_title["watch_kind"].tolist() == [None]


def test_taxonomy_counts_shape():
    df = ct.annotate_catalog(_spark_frame())
    counts = ct.taxonomy_counts(df)
    assert counts["item_category"]["Electronics"] == {"cases": 1, "computers": 1,
                                                      "watch_accessories": 1, "watches": 1,
                                                      "unknown": 1}
    assert counts["item_category"]["Grocery"] == {"tea": 1, "unknown": 1}
    assert counts["watch_kind"] == {"watch": 3, "storage": 1}
    assert counts["diet_tags"] == {"gluten free": 1, "organic": 1}
    assert counts["works_with"] == {"apple": 1}
    assert counts["item_condition"] == {"renewed": 1}
    assert counts["brand_from_specs"] == 2
    assert ct.taxonomy_counts(pd.DataFrame({"x": [1]})) == {}
    # counts and keys only: no title text reaches the log line
    assert "Silicone" not in repr(counts) and "Kalawen" not in repr(counts)


# ------------------------------------------------------------------ the app/search.py hook
class _Records(logging.Handler):
    def __init__(self):
        super().__init__(logging.DEBUG)
        self.records = []

    def emit(self, record):
        self.records.append(record)


@contextlib.contextmanager
def _search_log():
    handler, logger = _Records(), logging.getLogger("app.search")
    old_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        yield handler.records
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)


def _expected(df):
    return {c: df[c].tolist() for c in ct.ADDED_COLUMNS}


def test_annotate_types_derives_the_taxonomy_after_product_type():
    from app import product_type as pt
    df = _spark_frame()
    with tempfile.TemporaryDirectory() as tmp, patched(S, {"CACHE_DIR": Path(tmp)}), \
            _search_log() as records:
        S._annotate_types(df)                   # no csv_hash (Databricks): no cache file
        assert not list(Path(tmp).iterdir())
    for column, want in SPARK_EXPECTED.items():
        assert df[column].tolist() == want, column
    columns = list(df.columns)
    assert columns.index(pt.TYPE_COLUMN) < columns.index("item_category")
    assert df["product_type"].tolist()[6] == "accessory"   # the bracelet watch: not jewelry
    lines = [r.getMessage() for r in records]
    taxonomy = [m for m in lines if m.startswith("Catalog taxonomy derived in ")]
    assert len(taxonomy) == 1 and "'item_category'" in taxonomy[0], lines
    for title in _spark_frame()["Product_title"].dropna():   # counts only, never a title
        assert all(title not in m for m in lines), title


def test_taxonomy_cache_round_trip():
    df_miss, df_hit = _spark_frame(), _spark_frame()
    with tempfile.TemporaryDirectory() as tmp, patched(S, {"CACHE_DIR": Path(tmp)}):
        S._annotate_taxonomy(df_miss, "c0ffee000001")
        files = list(Path(tmp).iterdir())
        assert [p.name for p in files] == [S._taxonomy_cache_path("c0ffee000001").name]
        assert re.fullmatch(r"taxonomy_c0ffee000001_[0-9a-f]{12}\.npz", files[0].name)

        def refuse(df):
            raise AssertionError("derived again: the cache should have been hit")

        with patched(S.catalog_taxonomy, {"annotate_catalog": refuse}), \
                _search_log() as records:
            S._annotate_taxonomy(df_hit, "c0ffee000001")
        assert any(r.getMessage().startswith("Taxonomy cache hit: ") for r in records)
        # plain arrays and JSON: loads without pickle, no object arrays
        with np.load(files[0], allow_pickle=False) as stored:
            assert all(stored[name].dtype.kind in "iu" for name in stored.files)
    assert _expected(df_hit) == _expected(df_miss) == SPARK_EXPECTED
    for column in ct.ADDED_COLUMNS:
        assert df_hit[column].dtype == object, column
    assert all(isinstance(v, tuple) for c in ("diet_tags", "works_with") for v in df_hit[c])
    assert tuple(df_hit.columns[-7:]) == ct.ADDED_COLUMNS
    pd.testing.assert_frame_equal(df_hit, df_miss)


def test_taxonomy_cache_key_moves_with_the_catalog_the_rules_and_python():
    with tempfile.TemporaryDirectory() as tmp:
        with patched(S, {"CACHE_DIR": Path(tmp)}):
            base = S._taxonomy_cache_path("aaaaaaaaaaaa")
            assert base.parent == Path(tmp)
            assert S._taxonomy_cache_path("bbbbbbbbbbbb") != base          # another CSV
            with patched(S, {"_TAXONOMY_CACHE_FORMAT": "test"}):           # another format
                assert S._taxonomy_cache_path("aaaaaaaaaaaa") != base
            import types
            other_python = types.SimpleNamespace(version_info=(3, 99, 0))
            with patched(S, {"sys": other_python}):                       # another `re`
                assert S._taxonomy_cache_path("aaaaaaaaaaaa") != base
            # one byte more in the rules module (here: a comment) is another key -- for a
            # file no newer than the import (an edit after it: the next test)
            edited = Path(tmp) / "catalog_taxonomy.py"
            edited.write_bytes(Path(ct.__file__).read_bytes() + b"\n# edited\n")
            before_import = ct.IMPORTED_AT - 60
            os.utime(edited, (before_import, before_import))
            with patched(ct, {"__file__": str(edited)}):
                assert S._taxonomy_cache_path("aaaaaaaaaaaa") not in (base, None)
            assert S._taxonomy_cache_path("aaaaaaaaaaaa") == base          # deterministic
    # the app's cache directory is the repo's .cache/ (gitignored), never the tree itself
    assert cfg.CACHE_DIR == ROOT / ".cache" and S.CACHE_DIR == cfg.CACHE_DIR


def _load_copy(path, name):
    """A fresh import of a copy of the rules module (another process, in effect)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rules_edited_after_import_are_never_cached_under_the_new_key():
    """2026-10-30 review: the key hashed the rules FILE when the cache was read or written,
    but the columns came from the rules imported earlier, so an edit in between filed the
    old rules' columns under the new rules' name and every later process served them.
    Now a file newer than its import gives no key: derive, write nothing. The next process
    (a fresh import of the edited file) derives with the NEW rules and caches them."""
    original = Path(ct.__file__).read_bytes()
    edit = (b"\n_unedited_annotate = annotate_catalog\n\n\ndef annotate_catalog(df):\n"
            b"    _unedited_annotate(df)\n    df['brand'] = 'edited'\n    return df\n")
    with tempfile.TemporaryDirectory() as tmp:
        rules = Path(tmp) / "rules_copy.py"
        rules.write_bytes(original)
        old_rules = _load_copy(rules, "cas_rules_before_the_edit")
        rules.write_bytes(original + edit)              # edited after old_rules' import
        edited_at = old_rules.IMPORTED_AT + 5           # (explicit: coarse file clocks)
        os.utime(rules, (edited_at, edited_at))
        cache = Path(tmp) / "cache"
        with patched(S, {"CACHE_DIR": cache, "catalog_taxonomy": old_rules}), \
                _search_log() as records:
            assert S._taxonomy_cache_path("c0ffee000004") is None
            df = _spark_frame()
            S._annotate_taxonomy(df, "c0ffee000004")
        assert _expected(df) == SPARK_EXPECTED          # the old rules ran...
        assert not cache.exists() or not list(cache.iterdir())   # ...and filed nothing
        warnings = [r.getMessage() for r in records if r.levelno == logging.WARNING]
        assert warnings == ["Taxonomy cache off: app/catalog_taxonomy.py is unreadable or "
                            "newer than its import; deriving."], warnings
        new_rules = _load_copy(rules, "cas_rules_after_the_edit")
        edited_at = new_rules.IMPORTED_AT - 5           # the next process started after it
        os.utime(rules, (edited_at, edited_at))
        with patched(S, {"CACHE_DIR": cache, "catalog_taxonomy": new_rules}):
            for _pass in ("miss", "hit"):
                df = _spark_frame()
                S._annotate_taxonomy(df, "c0ffee000004")
                assert set(df["brand"]) == {"edited"}, _pass
                assert len(list(cache.iterdir())) == 1
    # an unreadable rules file is no key either, never a failed boot
    with patched(ct, {"__file__": str(Path(tmp) / "gone.py")}):
        assert S._taxonomy_digest() is None


_CSV_CHANGED = "Taxonomy cache off: the catalog changed while it was read; deriving."


def test_a_catalog_rewritten_during_the_load_is_never_cached_under_its_new_hash():
    """2026-10-30 review, the catalog half of the key: load_index hashes the CSV AFTER
    reading it, so a catalog rewritten in between (an enrich run during a boot) filed the
    OLD rows' columns under the NEW catalog's md5 and every later boot served them. A
    file whose size or mtime moved across the read and hash gives the taxonomy no key:
    derive, write nothing, say so. The control: the unchanged (rewritten) catalog caches."""
    state = ("_df", "_embeddings", "_loaded_backend", "_loaded_fields", "_embedded_fields")
    saved, backend, real_hash = {k: getattr(S, k) for k in state}, cfg.SEARCH_BACKEND, S._file_hash

    def rewritten_then_hashed(path):
        with open(path, "a", encoding="utf-8") as f:     # after read_csv, before the hash
            f.write("Electronics,USB C Charging Cable,,,,,,,\n")
        return real_hash(path)

    class _Embedder:
        model_name = "fake/model"

    def fake_embed(texts):
        return np.ones((len(texts), 4), dtype=np.float32)

    try:
        cfg.SEARCH_BACKEND = "local"
        with tempfile.TemporaryDirectory() as tmp:
            csv = Path(tmp) / "catalog.csv"
            _spark_frame().to_csv(csv, index=False)
            for run, file_hash in (("rewritten", rewritten_then_hashed), ("control", real_hash)):
                cache = Path(tmp) / run
                cache.mkdir()
                S._df = None
                with patched(S, {"PRODUCTS_CSV": csv, "CACHE_DIR": cache, "embed_text": fake_embed,
                                 "get_default_embedder": _Embedder, "_file_hash": file_hash}), \
                        _search_log() as records:
                    S.load_index()
                taxonomy = [p.name for p in cache.iterdir() if p.name.startswith("taxonomy_")]
                warnings = [r.getMessage() for r in records if r.levelno == logging.WARNING]
                if run == "rewritten":
                    assert taxonomy == [] and _CSV_CHANGED in warnings, (taxonomy, warnings)
                    assert _expected(S._df) == SPARK_EXPECTED       # derived from the rows read
                else:
                    assert _CSV_CHANGED not in warnings, warnings
                    assert taxonomy == [f"taxonomy_{real_hash(csv)}_{S._taxonomy_digest()}.npz"]
                    assert len(S._df) == 11 and S._df["item_category"].iloc[-1] == "cables"
    finally:
        for k, v in saved.items():
            setattr(S, k, v)
        cfg.SEARCH_BACKEND = backend


def test_taxonomy_cache_is_never_served_stale():
    """A file that does not describe THIS frame is a miss: the frame is derived again
    and the file rewritten. Covers a row-count change, a corrupt file and a file listing
    other columns."""
    with tempfile.TemporaryDirectory() as tmp, patched(S, {"CACHE_DIR": Path(tmp)}):
        S._annotate_taxonomy(_spark_frame(), "c0ffee000002")
        path = S._taxonomy_cache_path("c0ffee000002")
        # another frame under the same name (cannot happen with a real md5): miss
        smaller = _spark_frame().iloc[:6].copy()
        with _search_log() as records:
            S._annotate_taxonomy(smaller, "c0ffee000002")
        assert any(r.getMessage().startswith("Taxonomy cache miss: ") for r in records)
        assert smaller["item_category"].tolist() == SPARK_EXPECTED["item_category"][:6]
        # a corrupt file: a warning naming the exception TYPE only, then a fresh derivation
        path.write_bytes(b"not an npz file")
        df = _spark_frame()
        with _search_log() as records:
            S._annotate_taxonomy(df, "c0ffee000002")
        warnings = [r.getMessage() for r in records if r.levelno == logging.WARNING]
        assert warnings and warnings[0].startswith("Taxonomy cache unreadable ("), warnings
        assert _expected(df) == SPARK_EXPECTED
        rebuilt = _spark_frame()
        S._annotate_taxonomy(rebuilt, "c0ffee000002")               # the rewrite is valid
        assert _expected(rebuilt) == SPARK_EXPECTED
        # a file whose columns are not ADDED_COLUMNS (a format the code no longer writes)
        with patched(ct, {"ADDED_COLUMNS": ct.ADDED_COLUMNS[:-1]}):
            S._save_taxonomy_cache(_spark_frame().pipe(ct.annotate_catalog), path)
        assert not S._load_taxonomy_cache(_spark_frame(), path)
        assert not [p for p in Path(tmp).iterdir() if p.suffix == ".tmp"]


def test_taxonomy_cache_write_failure_is_no_error():
    df = _spark_frame()
    with tempfile.TemporaryDirectory() as tmp:
        blocker = Path(tmp) / "not_a_dir"
        blocker.write_text("x")
        with patched(S, {"CACHE_DIR": blocker / "cache"}), _search_log() as records:
            S._annotate_taxonomy(df, "c0ffee000003")
    assert _expected(df) == SPARK_EXPECTED
    assert any(r.getMessage().startswith("Taxonomy cache not written (") for r in records)


# ------------------------------------------------------------------ slow: the real catalog
def _real_catalog():
    if os.getenv("CAS_SLOW_TESTS") != "1":
        _skip("set CAS_SLOW_TESTS=1 (and PRODUCTS_CSV) to read the real catalog")
    if not Path(cfg.PRODUCTS_CSV).exists():
        _skip(f"no catalog at {cfg.PRODUCTS_CSV}")
    wanted = ["bsns_vrtcl_name", "Product_title", "dietary", "store", "specs"]
    return pd.read_csv(cfg.PRODUCTS_CSV, low_memory=False, usecols=lambda c: c in wanted)


def test_real_catalog_coverage_index_and_residue_slow():
    """The whole catalog: coverage within the 2026-10-27 measured ranges, the watches /
    watch_kind identity, no match the index would hide on any segment, the watch
    reader's substring needles hiding nothing, and the fast reading against the
    exhaustive one: a counted residue (finditer skips overlapping phrases, e.g. "lens
    mount adapter"), on every title of which the fast reading equals the unindexed
    one (every rule from every token start), as on a random sample."""
    df = _real_catalog()
    started = time.perf_counter()
    ct.annotate_catalog(df)
    elapsed = time.perf_counter() - started
    counts = ct.taxonomy_counts(df)
    print(f"\n  annotate_catalog: {len(df):,} rows in {elapsed:.1f}s; {counts}")
    vertical = df["bsns_vrtcl_name"].map(ct.vertical_of)
    elec, groc = vertical == ct.ELECTRONICS, vertical == ct.GROCERY
    if elec.any():
        typed = (df.loc[elec, "item_category"] != "unknown").mean()
        devices = (df.loc[elec, "item_role"] == "device").mean()
        print(f"  Electronics typed {typed:.1%}, devices {devices:.1%}")
        assert 0.92 <= typed <= 0.96, typed
        assert 0.22 <= devices <= 0.26, devices
        watches = df.loc[elec, "item_category"] == "watches"
        assert (watches == (df.loc[elec, "watch_kind"] == "watch")).all()
        assert 550 <= watches.sum() <= 700, watches.sum()
        brand_moved = (df.loc[elec, "brand"] != df.loc[elec, "store"]) & df.loc[elec, "store"].notna()
        assert 1500 <= brand_moved.sum() <= 1800, brand_moved.sum()
    if groc.any():
        typed = (df.loc[groc, "item_category"] != "unknown").mean()
        print(f"  Grocery typed {typed:.1%}")
        assert 0.89 <= typed <= 0.93, typed
        tagged = df.loc[groc, "diet_tags"].map(bool).mean()
        assert 0.33 <= tagged <= 0.37, tagged
    assert not df.loc[~(elec | groc), "item_category"].notna().any()
    kinds = df["watch_kind"].value_counts()
    print(f"  watch kinds {kinds.to_dict()}")
    assert 1900 <= kinds.get("watch", 0) <= 2200
    assert 3000 <= kinds.get("band", 0) <= 3400
    assert elapsed < 60, f"annotate_catalog took {elapsed:.1f}s"

    # the watch reader's substring needles (_W_NEEDLES) hide no title: reading every
    # title with a needle test that always passes gives the same kinds
    titles = sorted({ct._norm(t) for t in df["Product_title"]})
    with_needles = [ct.watch_kind_text(t) for t in titles]
    with patched(ct, {"_W_NEEDLE_RE": re.compile("")}):
        hidden = [t for t, kind in zip(titles, with_needles) if ct.watch_kind_text(t) != kind]
    assert not hidden, hidden[:5]
    # ... nor do the cleaner hints (_CLEANER_HINT): a cleaner they hid would keep its device
    # category, so every Electronics device title reads the same with a hint test that
    # always passes
    devices = sorted({ct._norm(t) for t in df.loc[elec & (df["item_role"] == "device"),
                                                  "Product_title"]})
    with patched(ct, {"_CLEANER_HINT": ("",)}):
        hidden = [t for t in devices if ct._electronics_read(t)[0] == "tools"]
    print(f"  cleaner hints checked on {len(devices):,} Electronics device titles")
    assert not hidden, hidden[:5]

    # The documented residue (159 Electronics, 153 Grocery titles on 2026-10-27's catalog;
    # the same on 2026-10-30's rules) plus a small margin: a rule edit that adds more
    # departures from the finditer reading fails here, with ten of the residue's titles
    rng = random.Random(1027)
    for tax, rows, documented in ((ct.ELECTRONICS_TAXONOMY, elec, 159),
                                  (ct.GROCERY_TAXONOMY, groc, 153)):
        texts = sorted({ct._norm(t) for t in df.loc[rows, "Product_title"]})
        if not texts:
            continue
        started = time.perf_counter()
        missed = _index_misses(tax, texts)
        print(f"  {tax.name}: index checked from every token start of {len(texts):,} titles "
              f"in {time.perf_counter() - started:.0f}s")
        assert not missed, missed[:5]
        residue = [t for t in texts if tax.classify_text(t)[0] != tax.classify_text_ref(t)]
        print(f"  {tax.name}: fast != exhaustive (finditer) on {len(residue)} of {len(texts):,} "
              f"titles (documented {documented})")
        assert len(residue) <= documented + 6, (len(residue), residue[:10])
        # all of it is finditer's overlap rule: the unindexed reading agrees with the fast
        # one there and on a random sample (on every title: 0 of 198,904 differ, 2026-10-27;
        # the index check above is what proves the fast reading complete on every title)
        sample = residue + rng.sample(texts, min(3000, len(texts)))
        unequal = [t for t in sample if tax.classify_text(t)[0] != tax.classify_text_all(t)]
        assert not unequal, unequal[:5]


# ------------------------------------------------------------------ plain runner
if __name__ == "__main__":
    _PLAIN_RUNNER = True
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = skipped = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except _Skipped as e:
            skipped += 1
            print(f"SKIP  {name}: {e}")
        except Exception as e:  # noqa: BLE001 -- report every failure, then exit non-zero
            failed += 1
            print(f"FAIL  {name}: {type(e).__name__}: {str(e)[:500]}")
    print(f"\n{len(tests) - failed - skipped} passed, {failed} failed, {skipped} skipped")
    sys.exit(1 if failed else 0)
