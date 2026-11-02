"""Offline tests for app/device_role.py (2026-10-27, DEVICE_ROLE_RANKING): the device-ask
reader, the core / accessory predicates on catalog_taxonomy's load-time columns, the
gift add-on order and the reranker's constraint text. main.py's use of them (gate,
window, display, sorts, ads, match quality, row field, knob, cache key, the replay of the
recorded 10-15 watch capture) is in tests/test_ranking.py.

No network, no LLM, no catalog unless CAS_SLOW_TESTS=1 (PRODUCTS_CSV=
data/products_enriched.csv), which adds the whole catalog: core rows per family within
the measured ranges, every core row mentions its family, no Grocery row is ever a device
or about one. Runs under pytest (`python -m pytest tests/test_device_role.py`) or plainly
(`python tests/test_device_role.py`).
"""

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

import app.config as cfg  # noqa: E402
from app import catalog_taxonomy as ct  # noqa: E402
from app import device_role as dr  # noqa: E402


class _Skipped(Exception):
    """Raised by _skip() under the plain runner (it reports it), pytest or not."""


_PLAIN_RUNNER = False


def _skip(reason):
    if _PLAIN_RUNNER:
        raise _Skipped(reason)
    try:
        import pytest
    except ImportError:
        raise _Skipped(reason)
    pytest.skip(reason)


D = dr.DeviceAsk
# ------------------------------------------------------------------ the reader
# The 49 queries of the 2026-10-27 offline study (ranking_offline/reader_check.py), with
# the owner calls applied: tracker / AirTag and power bank are NOT families (D2), and
# "watch winder" / "watch box for men" are accessory asks (the prototype read them None
# because its owner-verb list held "watch", which also refused "watch for men")
READER = {
    "gift for a vintage watch collector": D("watch", True, False),
    "gift for a watch lover": D("watch", True, False),
    "airtag": None,
    "apple airtag": None,
    "airtag holder": None,
    "laptop for college": D("laptop", False, False),
    "laptop backpack": D("laptop", False, True),
    "laptop bag": D("laptop", False, True),
    "noise cancelling headphones for travel": D("headphones", False, False),
    "wireless earbuds": D("headphones", False, False),
    "apple watch": D("watch", False, False),
    "apple watch band": D("watch", False, True),
    "smart watch for women": D("watch", False, False),
    "watch movies on my bedroom wall": None,
    "watch party snacks": None,
    "stop my laptop from overheating": None,
    "keep my phone alive on a weekend camping trip": None,
    "never lose my keys again": None,
    "listen to music in the shower": None,
    "gift for a coffee lover": None,
    "a gift for someone who loves spicy food": None,
    "gift for my boyfriend who likes streetwear": None,
    "gift for my dad": None,
    "birthday gift for a 10 year old": None,
    "power bank": None,
    "outdoor security camera": D("camera", False, False),
    "camera bag": D("camera", False, True),
    "phone case": D("phone", False, True),
    "case for my iphone 14": None,
    "best laptop under $500": D("laptop", False, False),
    "15 inch laptop": D("laptop", False, False),
    "gaming headset": D("headphones", False, False),
    "kids tablet": D("tablet", False, False),
    "present for a headphone enthusiast": D("headphones", True, False),
    "gift for my husband who loves cameras": D("camera", True, False),
    "mens automatic watch": D("watch", False, False),
    "vintage watch": D("watch", False, False),
    "watch winder": D("watch", False, True),
    "watch box for men": D("watch", False, True),
    "something to keep my drinks cold at the beach": None,
    "ring light": None,
    "smart ring": None,
    "bluetooth speaker for the shower": D("speaker", False, False),
    "laptop stand": D("laptop", False, True),
    "dress watch for men": D("watch", False, False),
    "running watch with gps": D("watch", False, False),
    "gift for a gamer": None,
    "kindle": D("tablet", False, False),
    "kindle case": D("tablet", False, True),
    # the electronics battery and the page-type rows
    "watch for men": D("watch", False, False),
    "women's watch": D("watch", False, False),
    "smart watch for running": D("watch", False, False),
    "iphone": D("phone", False, False),
    "iphone 14 pro max": D("phone", False, False),
    "camera for vlogging": D("camera", False, False),
    "tablet for kids": D("tablet", False, False),
    "bluetooth speaker for the beach": D("speaker", False, False),
    "wireless earbuds under $50": D("headphones", False, False),
    "gift for a photographer": None,          # a persona, not a device noun
    "power bank for camping": None,
    # gift phrasings
    "watch collector gift": D("watch", True, False),
    "gift for someone who collects watches": D("watch", True, False),
    "gift ideas for a camera lover": D("camera", True, False),
    "gift for a watch collector who has everything": D("watch", True, False),
    "watch gift for my dad": D("watch", False, False),   # names the watch itself
    # model words and specs after the noun
    "apple watch series 9": D("watch", False, False),
    "airpods pro case": D("headphones", False, True),
    "headphones wireless bluetooth": D("headphones", False, False),
    "buy a laptop": D("laptop", False, False),
    "e-reader for the beach": D("tablet", False, False),
    "phone holder for car": D("phone", False, True),
    # owner needs: a possessive before the device, or a care / use verb first
    "my laptop keeps overheating": None,
    "her first smartphone": None,
    "charge my phone in the car": None,
    "protect my ipad screen": None,
    "his and hers watches": None,
    # ... and the verb alone, no possessive (the 2026-10-27 review: every row above also
    # had one, so a reader without the verb rule passed this table)
    "stop laptop overheating": None,
    "cool down laptop": None,
    "fix cracked iphone screen": None,
    "charge apple watch": None,
    "store watches": None,
    "protect ipad screen": None,
    "keep phone charged while camping": None,
    "mount phone in car": None,
    "organize headphones": None,
    "power laptop in car": None,
    "connect phone to tv": None,
    # a lexical collision the pool gate neutralises (main._effective_device)
    "dishwasher tablets": D("tablet", False, False),
    # 2026-10-27 review, owner call: the family noun must END the head phrase, followed
    # only by model / spec / descriptor words; any other noun after it is a compound
    # for the device (a part or an add-on), so the reading fails closed -- None, the page
    # of before 2026-10-27 -- whether or not the accessory list knows the noun
    "watch cap": None,
    "watch face": None,
    "speaker wire": None,
    "headphone pads": None,
    "headphone amp": None,
    "headphone jack": None,
    "laptop desk": None,
    "laptop ram": None,
    "camera microphone": None,
    "camera flash": None,
    "phone ring": None,
    "phone screen": None,
    # ... descriptors after the noun keep the device (the keyword-query word order)
    "headphones noise cancelling": D("headphones", False, False),
    "watch automatic mens": D("watch", False, False),
    "iphone unlocked black": D("phone", False, False),
    "laptop gaming": D("laptop", False, False),
    "speaker waterproof": D("speaker", False, False),
    "camera digital": D("camera", False, False),
    "kindle paperwhite 16gb": D("tablet", False, False),
    "gopro hero 11 black": D("camera", False, False),
    # coordinations: a device ask only when every part is the same device
    "headphones or earbuds": D("headphones", False, False),
    "smart watch and fitness tracker": D("watch", False, False),
    "iphone and apple watch charger": None,
    "phone or tablet stand": None,
    "headphones and earbuds case": None,
    "laptop and mouse": None,
    "iphone and ipad": None,
    "black and white camera": None,
    "gift for someone who loves watches and cameras": None,
    "watch for men and women": D("watch", False, False),   # the coordination is after "for"
    # compounds are right-headed: the LAST family noun names the product
    "phone camera": D("camera", False, False),
    "laptop speaker": D("speaker", False, False),
    "watch phone": D("phone", False, False),
    "iphone camera lens": D("camera", False, True),
    # 2026-10-30 review: a word BEFORE the noun that makes it another device -- the ask's
    # twin of the title rules (_OTHER_DEVICE_RE['tablet'], _LANDLINE_RE; catalog_taxonomy
    # files VR headsets under gaming). Read as tablet / phone / headphones, one stray core
    # row passed the gate and inverted the page ("writing tablet for kids": 10 Android kids
    # tablets seated over the LCD writing tablets; "landline phone": a Samsung mobile first)
    "drawing tablet": None,
    "graphics tablet for beginners": None,
    "writing tablet for kids": None,
    "lcd writing tablet": None,
    "wacom tablet": None,
    "landline phone": None,
    "cordless phone": None,
    "corded phone with caller id": None,
    "vr headset": None,
    "virtual reality headset": None,
    "gift for a drawing tablet lover": None,
    # ... only a headSET is vetoed, only a word before the noun, and not "e-paper" (an
    # e-reader) or "3d" (gaming headsets sell "3d surround" audio)
    "vr headphones": D("headphones", False, False),
    "3d surround gaming headset": D("headphones", False, False),
    "android tablet": D("tablet", False, False),
    "tablet for drawing": D("tablet", False, False),
    "e-paper tablet": D("tablet", False, False),
    "flip phone": D("phone", False, False),
}


def test_reader_table():
    for query, want in READER.items():
        got = dr.requested_device(query)
        assert got == want, (query, got, want)
        # case, curly apostrophes and spacing do not matter
        assert dr.requested_device("  " + query.upper().replace("'", "’") + " ") == want, query


def test_families_are_the_owner_list():
    """D2: a closed list; NOT power bank (a power bank is itself a charger) and NOT
    tracker / AirTag (0 AirTags in the catalog: a supply gap the banner reports)."""
    assert dr.FAMILIES == ("watch", "laptop", "headphones", "phone", "tablet", "camera",
                           "speaker")
    assert set(dr.FAMILY_RE) == set(dr.FAMILIES)
    for query in ("power bank", "portable charger", "airtag", "tile tracker", "key finder"):
        assert dr.requested_device(query) is None, query
    for blank in (None, "", "   ", 3, float("nan")):
        assert dr.requested_device(blank) is None


def test_intents_that_name_the_device():
    """A gift's add-on intents keep reserved seats; its device intents do not."""
    core = ["vintage mechanical wristwatch", "vintage style men's analog dress watch",
            "men's wristwatch", "pocket watch"]
    addon = ["leather watch winder", "vintage watch repair kit", "vintage watch display case",
             "watch cleaning kit", "wooden watch display box", "watch band", "coin display case"]
    for term in core:
        assert dr.names_device(term, "watch"), term
    for term in addon:
        assert not dr.names_device(term, "watch"), term
    assert not dr.names_device("lightweight laptop", "watch")
    assert dr.names_device("lightweight laptop", "laptop")


# ------------------------------------------------------------------ rows
# (vertical, title, family, expected role, expected addon rank or None = not checked)
ROWS = [
    ("Fashion", "Men's Automatic Mechanical Wrist Watch Leather Strap", "watch", "core", None),
    ("Fashion", "Kenneth Cole Mens Automatic Bracelet Watch", "watch", "core", None),
    ("Fashion", "Antique Mechanical Pocket Watch with Chain", "watch", "core", None),
    ("Electronics", "Apple Watch Series 7 GPS 41mm Smartwatch", "watch", "core", None),
    ("Electronics", "Garmin Forerunner 255 GPS Running Smartwatch", "watch", "core", None),
    ("Electronics", "Fitbit Charge 5 Fitness Tracker", "watch", "core", None),
    ("Fashion", "Watch Box for Men 12 Slots Leather Watch Case Organizer", "watch",
     "accessory", 0),
    ("Fashion", "Automatic Watch Winder for Single Watch", "watch", "accessory", 0),
    ("Fashion", "Watch Band Strap Quick Release Leather Replacement", "watch", "accessory", 1),
    ("Fashion", "Trail Loop Band for Apple Watch", "watch", "accessory", 1),
    ("Fashion", "SOONAN DIY Watch Repair Tool Kit", "watch", "accessory", 2),
    ("Electronics", "Screen Cleaner Kit, Laptop Computer Monitor Cleaner", "watch",
     "accessory", 2),
    ("Fashion", "Leather Bifold Wallet for Men", "watch", "unknown", None),
    ("Grocery", "Chocolate Pocket Watch Candy", "watch", "unknown", None),
    ("Electronics", "Lenovo IdeaPad 3 Laptop 15.6 inch Intel Core i5 8GB RAM", "laptop",
     "core", None),
    ("Electronics", "Lenovo 2023 Yoga 7i 16\" Touchscreen 2-in-1 Laptop", "laptop", "core", None),
    ("Electronics", "Laptop Sleeve 13 inch Compatible with MacBook Air", "laptop",
     "accessory", 0),
    ("Electronics", "Laptop Cooling Pad with 5 Fans", "laptop", "accessory", 1),
    ("Electronics", "Dell Inspiron Desktop Computer Intel Core i7", "laptop", "unknown", None),
    ("Electronics", "Sony WH-1000XM4 Wireless Noise Canceling Overhead Headphones",
     "headphones", "core", None),
    ("Electronics", "Hard Travel Case for Sony WH-1000XM4 Headphones", "headphones",
     "accessory", 0),
    ("Electronics", "Apple iPhone 12 Mini, 64GB, Black - Unlocked (Renewed)", "phone",
     "core", None),
    ("Electronics", "iPhone 12 Case Clear Shockproof", "phone", "accessory", 0),
    ("Electronics", "Fast Charger for iPhone 12 USB C", "phone", "accessory", 2),
    ("Electronics", "Kindle Paperwhite 8GB", "tablet", "core", None),
    ("Grocery", "Finish Dishwasher Tablets 90 Count", "tablet", "unknown", None),
    ("Electronics", "Canon EOS Rebel T7 DSLR Camera with 18-55mm Lens", "camera", "core", None),
    # a security camera reads optics ("night vision") / smart_home, a dash cam car
    ("Electronics", "Wireless Outdoor Security Camera 1080P Night Vision", "camera",
     "core", None),
    ("Electronics", "Dash Cam Front and Rear 4K", "camera", "core", None),
    ("Electronics", "JBL Flip 5 Waterproof Portable Bluetooth Speaker", "speaker", "core", None),
    ("Electronics", "Echo Dot (3rd Gen) - Smart speaker with Alexa", "speaker", "core", None),
    # 2026-10-27 review: real catalog titles that catalog_taxonomy files INSIDE the
    # device's category (item_role "device"), from the judged samples (build/NOTES_B1.md
    # "Review round"). Before the title rules each of these non-devices was core -- seated
    # first and sent to the reranker as `role=device` ("ipad": a case, a handle, a book).
    ("Electronics", "Tablet Pc Handle for iPad - Movie / Gamer Model - White / Pink / White",
     "tablet", "accessory", None),
    ("Electronics", "iPad in easy steps: Covers all models of iPad with iOS 12", "tablet",
     "accessory", None),
    ("Electronics", "BETA-7 Time-Release Beta-Alanine, 180 Tablets", "tablet", "accessory",
     None),
    ("Electronics", "Boogie Board E-Writer 8.5\" Paperless Memo Pad Tablet Blue with Neoprene "
     "Sleeve + Stylus", "tablet", "accessory", None),
    ("Electronics", "Universal Cell Phone Retro Handset", "phone", "accessory", None),
    ("Electronics", "DONG 3.5mm Retro Phone Handset for Cell Phone", "phone", "accessory", None),
    ("Electronics", "VTech 9111 900MHz Analog Phone (Blue)", "phone", "accessory", None),
    ("Electronics", "Katana Safety Arc: The Personal Security System That Attaches Directly "
     "to Your Smartphone", "phone", "accessory", None),
    ("Electronics", "Automobile Dodge VES Chrysler Ear Pads Headphones Foam", "headphones",
     "accessory", None),
    ("Electronics", "HP PAVILION DM4-2070US Laptop Screen 14 HP PAVILION DM4-2070US Laptop "
     "Screen WXGA HD 1366x768", "laptop", "accessory", None),
    ("Electronics", "Mystery Gaming Cooling Fan Stand Powerful Laptop Ultrabook Cooler Pad "
     "Computer Cooling Mat", "laptop", "accessory", None),
    ("Electronics", "Takom 2065 T30/34 US Heavy Tank 2-In-1 1:35 Scale - Plastic Tank Model Kit",
     "laptop", "unknown", None),
    ("Electronics", "Camarm Action Camera Waterproof Housing Hero 4 Black Design", "camera",
     "accessory", None),
    ("Electronics", "Gariz Genuine Leather XA-CFEM5BK Camera Cap Fixs for Olympus OM-D E-M5 "
     "EM5 42.5mm, Black", "camera", "accessory", None),
    ("Electronics", "GoPro Protective Housing (HERO8 Black) - Official GoPro Accessory",
     "camera", "accessory", None),
    ("Electronics", "KISSTAKER 57x20inch Speaker Fabric Cloth - Stereo Grill Mesh for Speaker "
     "Box Repair-Black-Recover Your Speaker in Minutes", "speaker", "accessory", None),
    ("Electronics", "1byone Belt Drive 3 Speed Mini Stereo Turntable with Built in Speakers",
     "speaker", "accessory", None),
    ("Electronics", "Goldwood Speaker Box Bass Air Ports 2 Speaker Port Tubes Black (PT-307-2)",
     "speaker", "accessory", None),
    ("Electronics", "KUNFINE Android Autoradio Car Navigation Stereo GPS Radio 9\" IPS Touch "
     "Screen Pad BT WiFi Headunit Tablet for Hyundai Elantra", "tablet", "accessory", None),
    ("Electronics", "iRULU Tablet 7 inch Android 8.1 Quad Core 1024x600 Dual Camera WiFi "
     "Bluetooth 8GB", "camera", "accessory", None),
    # an audio device with no speaker noun is no speaker (a microphone)
    ("Electronics", "Blue Yeti USB Microphone for Recording and Streaming", "speaker",
     "unknown", None),
    # ... and the devices those rules must keep: model names without the family noun, a
    # "Desktop Replacement" laptop, a 2-in-1 named by its specs, a metal-housing camera
    # and earphones, a camera named after a parenthesised "(Compatible for ...)", "Android
    # 10 Tablets", the ASUS MeMO Pad tablet line, an in-ear monitor
    ("Electronics", "Samsung Galaxy S10 Factory Unlocked Android Cell Phone, US Version, "
     "128GB", "phone", "core", None),
    ("Electronics", "Samsung Galaxy Tab A7 10.4 Wi-Fi 32GB Silver", "tablet", "core", None),
    ("Electronics", "Hoozo Android Tablet 10 Inch, Android 10 Tablets, ZPad R05 32GB ROM 128GB "
     "Expand, 6000mah Battery", "tablet", "core", None),
    ("Electronics", "ASUS MeMO Pad ME172V-A1-GR 7.0-Inch 16 GB Tablet (Grey)", "tablet", "core",
     None),
    ("Electronics", "ROG G703GX Desktop Replacement Gaming Laptop, GeForce RTX 2080, Intel Core "
     "i9-8950HK Processor", "laptop", "core", None),
    ("Electronics", "Dell XPS 13 9365 2-in-1 - 13.3\" FHD Touch - i7-7Y75 - 16GB Ram - 256GB "
     "SSD - Silver", "laptop", "core", None),
    ("Electronics", "COOLINT Glossy Black Solid Zinc Alloy Metal Housing Car Parking Backup "
     "Camera with Multifunction Switchers", "camera", "core", None),
    ("Electronics", "(Compatible for Hikvision) Outdoor PTZ PoE IP Camera 5MP Security Dome "
     "with 4X Optical Zoom", "camera", "core", None),
    ("Electronics", "Fujifilm Instax Mini 9 (Lime Green) with Instax Mini Film (20 Sheets)",
     "camera", "core", None),
    ("Electronics", "GREENPOINTSELECT Premium Genuine Full Metal Housing Dual Driver Deep Heavy "
     "Bass Handmade In-ear Noise-isolating Earphones", "headphones", "core", None),
    ("Electronics", "Linsoul THIEAUDIO Legacy 4 Hybrid Driver in-Ear Monitor with 2 Tuning "
     "Switch", "headphones", "core", None),
    ("Electronics", "Clarion CMG2510W 10-Inch 250 Watt Marine Subwoofer", "speaker", "core",
     None),
    # 2026-10-30 R2 (catalog_taxonomy): a cleaner is never the device it cleans. These real
    # titles were filed in the device's category with item_role "device" (the title rules
    # kept them non-core, but the rail counted them as devices and they ranked as other
    # add-ons); now tools, the care add-ons that rank last
    ("Electronics", "6 in 1 Keyboard Cleaner Kit Earphone Cleaner Kit with Brush Spong Pen "
     "Nibs for Electronic Cleaning Cellphone Hole Earbuds Monitor (White)", "headphones",
     "accessory", 2),
    ("Electronics", "Bluetooth Earbuds Cleaning Pen 3 in 1 Cleaner Kit Earphone Brush, "
     "Multi-Function Portable Soft Cleaning Brush Pen Tools for Airpods Pro", "headphones",
     "accessory", 2),
    ("Electronics", "Charm &Magic Professional Camera Cleaning Set with Air Blower Cleaner "
     "Lens Cleaning Pen Cleaning Cloths for Cleaning Your Camera Lens", "camera", "accessory",
     2),
    ("Electronics", "vershine Mini Computer Vacuum Keyboard Cleaner PC Laptop Brush Dust "
     "Cleaning Kit Computer Vacuums", "laptop", "accessory", 2),
    ("Electronics", "Vinyl Record Cleaner Carbon Fiber Brush Turntables LP Deep Cleaning Roller "
     "Professional and Effective Vinyl Cleaning Solution", "speaker", "accessory", 2),
    # ... and the watch reader's residue (B1's judged core samples): a strap's hole punch,
    # a slap bracelet and a multitool adapter were core watches; a slap WATCH still is
    ("Fashion", "NEW REVOLVING LEATHER PUNCH WATCH BELT HOLE PUNCH RED", "watch", "accessory",
     2),
    ("Fashion", "1Pc Women's Feather Slap Bracelet Watch Patting Circle Slap Ring Wrist "
     "Decoration Cuff Sleeves Slap Bangle Hair Accessories", "watch", "accessory", 1),
    ("Fashion", "BestTechTool watch adapter compatible with LEATHERMAN TREAD LT - BTT adapter "
     "(compatible with Garmin Fenix5X-Fenix 3 (see description), Black, TREAD LT)", "watch",
     "accessory", 2),
    ("Fashion", "Tocosy Slap on Watch Silicone Quartz Number Sport Wrist Band for Children "
     "Girl Lady Women Men Boy Unisex Slap Kids Watches", "watch", "core", None),
    # ... a smartwatch's protective "watch case" is protection, a watch box storage
    ("Electronics", "HoneyGirl Watch Case for Fitbit Versa 3, Scratch-Proof Cover Protective "
     "Bumper Shell", "watch", "accessory", 1),
    ("Electronics", "Watch Box 12 Slot Display Case", "watch", "accessory", 0),
    # 2026-10-30 F2 (catalog_taxonomy, the Phase C live run's residue): a band named right
    # before "Apple Watch" was a core watch and a kids-tablet case a core tablet; a camera
    # strap filed with the watch accessories is a camera accessory; the real kids tablet
    # that ends on its Kid-Proof Case stays core
    ("Fashion", "Goton Leather Band Apple Watch Series 4 3 2 1, Retro Leather Watch Strap "
     "Compatible Apple Watch Band 40mm 38mm Leather Men Women Stainless Steel Buckles", "watch",
     "accessory", 1),
    ("Electronics", "Armorex Dragon Touch Alldaymall NeuTab Chromo Inc 7\" Kids Tablet, Durable "
     "[Shockproof] [Impact Resistant] [Kids Safe] Rugged Silicone Cover (Hot Pink)", "tablet",
     "accessory", 0),
    ("Electronics", "HiGrace Kids Tablet 10 inch, Android 12 Quad core Tablet for Kids (Ages "
     "3-12), 5000mAh, 32GB ROM, Dual Camera, WiFi, Parental Control, Kid-Proof Case-Blue",
     "tablet", "core", None),
    ("Electronics", "Eorefo Camera Strap Leather Universal Shoulder Neck Belt Strap for "
     "Mirrorless and DSLR Camera,Flower Red.", "camera", "accessory", 1),
    # 2026-10-30 review: catalog_taxonomy files car and marine speakers under 'car' (the dash
    # cam's category); they were accessories or unknown on "car speakers" while the car
    # speakers filed audio were core. Now core -- but not a head unit or an enclosure that
    # does not say it comes loaded, filed there too, nor an audio-filed "1/2 DIN" equalizer
    ("Electronics", "Alpine SPJ-161C2 6\" Coaxial Car Speakers", "speaker", "core", None),
    ("Electronics", "Hifonics ZS46CX Zeus Coaxial Car Speakers (Black, Pair) – 4x6 Inch Coaxial "
     "Speakers, 200 Watt, 2-Way Car Audio, Passive Crossover, Sound System (Grills Not "
     "Included)", "speaker", "core", None),
    ("Electronics", "Dual Electronics DMS365 Two 6.5 inch 3 Way Marine Speakers with 120 Watts of "
     "Peak Power", "speaker", "core", None),
    ("Electronics", "Rockford Fosgate Punch P1-1X12 Single P1 12\" Loaded Subwoofer Enclosure "
     "Ported,Black", "speaker", "core", None),
    ("Electronics", "Rockville RWS12CA Slim 1200 Watt 12\" Powered Car Subwoofer Enclosure + Wire "
     "Kit", "speaker", "core", None),
    ("Electronics", "ORION HCCA Series SUBWOOFER CAR Audio CAR Stereo RMS Power WATTS 2000 "
     "(HCCA104 / 10' 4 OHMS)", "speaker", "core", None),
    ("Electronics", "JVC CS-J420X 4-Inch 210 Watts 2-Way Coaxial Car Stereo Speakers (Pair)",
     "speaker", "core", None),
    ("Electronics", "(2) New JVC CS-DR6930 6\"X9\" 3-Way 1000 Watt Car Stereo Speakers+(2) "
     "Enclosures", "speaker", "core", None),
    ("Electronics", "MTX Audio SLH12U Unloaded Subwoofer Enclosure", "speaker", "unknown", None),
    ("Electronics", "Bbox Dual Sealed 10 Inch Subwoofer Enclosure - Accu-Tuned Sealed Subwoofer "
     "Boxes & Enclosures - Subwoofer Box Improves Audio Quality, Sound & Bass - Fits 2011-2016 "
     "Ford F250/350/450 Crew Cab", "speaker", "unknown", None),
    ("Electronics", "Double Din Car Stereo with Wireless Apple CarPlay and Android Auto, 7-Inch "
     "FHD Touchscreen Car Audio Receiver with Backup Camera, Bluetooth, Car Radio with FM, "
     "USB/TF/AUX Port, Mirror Link, Subwoofer", "speaker", "unknown", None),
    ("Electronics", "Dual Electronics MCP103 Multimedia Detachable Single DIN Marine Stereo with "
     "Built-in USB & SD Card Ports, Two 6.5 inch Dual Cone High Performance Marine Speakers & "
     "Long Range Marine Antenna", "speaker", "accessory", None),
    ("Electronics", "Hifonics HFEQ7 7-Band 9 Volts 1/2 DIN Pre-Amp Car Audio Graphic Equalizer "
     "with Front 3.5mm Auxiliary Input, Rear RCA Auxiliary Input and High Level Speaker Inputs "
     "Black", "speaker", "accessory", None),
]


def _annotated(rows, lower=False):
    df = pd.DataFrame([{"bsns_vrtcl_name": v.lower() if lower else v,
                        "Product_title": t.lower() if lower else t, "store": "s"}
                       for v, t, *_ in rows])
    ct.annotate_catalog(df)
    return [{k: (None if isinstance(v, float) and v != v else v) for k, v in r.items()}
            for r in df.to_dict(orient="records")]


def test_core_role_and_addon_rank_from_the_load_time_columns():
    for lower in (False, True):
        for (vertical, title, family, want, rank), row in zip(ROWS, _annotated(ROWS, lower)):
            got = dr.ROLE_NAMES[dr.role(row, family)]
            assert got == want, (lower, title, family, got)
            assert dr.is_core(row, family) == (want == "core"), title
            if want == "core":
                assert dr.mentions(row, family), title
            if rank is not None:
                assert dr.addon_rank(row, family) == rank, (title, dr.addon_rank(row, family))


def test_core_needs_the_device_role():
    """item_role 'device' is required on its own (owner call, 2026-10-27 review): today
    every row of a device category carries it, so this guards a taxonomy change that
    files add-ons in a device category with their own role."""
    row = {"bsns_vrtcl_name": "Electronics", "item_category": "headphones",
           "Product_title": "Sony WH-1000XM4 Wireless Noise Canceling Overhead Headphones",
           "item_role": "device"}
    assert dr.is_core(row, "headphones")
    for other in ("accessory", "part", "consumable", None, float("nan")):
        assert not dr.is_core(dict(row, item_role=other), "headphones"), other
    # watches read watch_kind alone (a Fashion wristwatch has no item_role at all)
    assert dr.is_core({"bsns_vrtcl_name": "Fashion", "watch_kind": "watch",
                       "Product_title": "Men's Automatic Watch"}, "watch")


def test_rows_without_columns_are_unknown():
    """An un-annotated row (an older frame, a mock) or NaN / None columns: never core,
    never a crash; only the title can make it an accessory."""
    for family in dr.FAMILIES:
        for row in ({}, {"Product_title": None}, {"watch_kind": float("nan"),
                                                  "item_category": float("nan")}):
            assert not dr.is_core(row, family)
            assert dr.role(row, family) == dr.UNKNOWN, (family, row)
    assert dr.role({"Product_title": "Laptop Sleeve 13 inch"}, "laptop") == dr.ACCESSORY
    assert not dr.mentions({"Product_title": "Laptop Sleeve", "bsns_vrtcl_name": "grocery"},
                           "laptop")


def test_constraint_text():
    """Deterministic per family, never the shopper's words; the examples in parentheses
    (main._type_constraint's idiom), no braces; the gift text scoped "for it" and never
    "consumables" (it would cap a coffee lover's coffee)."""
    for family in dr.FAMILIES:
        plain = dr.constraint(D(family, False, False))
        gift = dr.constraint(D(family, True, False))
        for c in (plain, gift):
            assert c["type"] == "product_type" and set(c) == {"type", "value"}
            assert "(" in c["value"] and "{" not in c["value"] and "}" not in c["value"]
            assert "consumable" not in c["value"].lower()
        assert plain["value"].endswith("violate it") and "accessories FOR it (" in plain["value"]
        assert gift["value"].startswith("the gift is ")
        assert "storage and display pieces for it rank below every real one" in gift["value"]
        assert gift["value"].endswith("replacement items for it violate it")
        assert dr.constraint(D(family, False, False)) == plain   # deterministic
    assert dr.constraint(D("watch", True, False))["value"] == (
        "the gift is a watch itself (wristwatches, pocket watches, smartwatches); storage "
        "and display pieces for it rank below every real one; tools, repair kits, cleaning "
        "and polishing supplies, spare parts, cables and replacement items for it violate it")


# ------------------------------------------------------------------ the whole catalog
def _real_catalog():
    if os.getenv("CAS_SLOW_TESTS") != "1":
        _skip("set CAS_SLOW_TESTS=1 (and PRODUCTS_CSV) to read the real catalog")
    if not Path(cfg.PRODUCTS_CSV).exists():
        _skip(f"no catalog at {cfg.PRODUCTS_CSV}")
    wanted = ["bsns_vrtcl_name", "Product_title", "store", "specs", "dietary"]
    return pd.read_csv(cfg.PRODUCTS_CSV, low_memory=False, usecols=lambda c: c in wanted)


# Core rows per family on 2026-10-27's catalog (299,972 rows), measured after the review
# round's title rules (the category alone in brackets): watch 2,007 [2,007; 2,022 before
# catalog_taxonomy's 2026-10-30 watch_kind fix, 2,014 before its R2 residue fix: a hole
# punch, a slap bracelet, a multitool adapter; 2,011 before its F2 fix: four bands named
# right before a smartwatch's name], laptop 2,245 [2,538; 2,542 before R2 filed four desk
# vacuums and dusters as tools], headphones 3,577 [3,814; 3,816: two earbud cleaners],
# phone 49 [215: landlines, VoIP and satellite phones, mounts, cables], tablet 476 [601;
# 479 / 622 before F2 filed 21 tablet cases listed like their tablets as cases, three of
# them core], camera 3,259 [3,841; 3,260 / 3,844 before R2: an air blower was core, and
# two camera cleaning kits], speaker 2,153 [2,802; 2,032 / 2,613 before the 2026-10-30
# review took the car category's 124 speakers in and three "1/2 DIN" equalizers out]. Judged
# precision (build/NOTES_B1.md "Review round"): ~0.93-1.00 a family. The ranges keep a
# ~10-15% margin: a catalog_taxonomy or title-rule change that moves a count out of its
# range is a deliberate edit here too
CORE_RANGES = {"watch": (1800, 2300), "laptop": (2000, 2600), "headphones": (3200, 4100),
               "phone": (35, 70), "tablet": (420, 560), "camera": (2900, 3700),
               "speaker": (1800, 2350)}


def test_real_catalog_core_counts_slow():
    df = _real_catalog()
    started = time.perf_counter()
    ct.annotate_catalog(df)
    rows = df[["bsns_vrtcl_name", "Product_title", "watch_kind", "item_category",
               "item_role"]].to_dict(orient="records")
    print(f"\n  annotate_catalog {time.perf_counter() - started:.1f}s, {len(rows)} rows")
    for family in dr.FAMILIES:
        counts = {name: 0 for name in dr.ROLE_NAMES.values()}
        grocery = 0
        for r in rows:
            role = dr.role(r, family)
            counts[dr.ROLE_NAMES[role]] += 1
            if role == dr.CORE:
                assert dr.mentions(r, family), (family, r["Product_title"])
            if dr._grocery(r):
                grocery += role != dr.UNKNOWN or dr.mentions(r, family)
        print(f"  {family:10s} {counts}")
        lo, hi = CORE_RANGES[family]
        assert lo <= counts["core"] <= hi, (family, counts)
        assert grocery == 0, (family, grocery)
    # one source: an Electronics "watches" row is exactly a watch_kind "watch" row
    elec = [r for r in rows if r["bsns_vrtcl_name"] == "Electronics"]
    assert all((r["item_category"] == "watches") == dr.is_core(r, "watch") for r in elec)


# ------------------------------------------------------------------ plain runner
if __name__ == "__main__":
    _PLAIN_RUNNER = True
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
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
