"""Offline tests for app/product_type.py and its app/search.py hook.

No network, no LLM, no embedding model: the search-hook tests fake the embedder and
the retrieval. Runs under pytest (`python -m pytest tests/test_product_type.py`) or
plainly (`python tests/test_product_type.py`).

Pinned here: the hand-labelled title table (in original case AND lowercased, the
local backend's shape), the shopper-ask table the ranking acts on, the documented
vocabulary collisions main.py's pool gate must neutralise, that annotate() never
moves a row (catalog_index is a ROW POSITION), that search_text and the embedding
cache key are unchanged by it, and that `keep=` filters before the per-intent quota.
Since 2026-10-19 (contract C1) also: the columns annotate() adds beyond the type
(fine_subtype, shopper_gender, audience, color_family) with their exported label and
swatch maps, and the shopper-ask readers (requested_fit, requested_subtypes,
requested_audience, implies_women, is_wedding_guest_ask) with their collisions.
Since 2026-10-27: a watch or watch band is never jewelry, and both load_index branches
carry app/catalog_taxonomy.py's columns (tests/test_catalog_taxonomy.py pins them);
since 2026-10-30 the ask agrees ("apple watch bracelet", "bracelet for apple watch"
request accessory, the type of their rows).
CAS_SLOW_TESTS=1 adds the full-catalog coverage, distributions, timing,
needle-soundness check and the watch fix's moved-row count (set
PRODUCTS_CSV=data/products_enriched.csv).
"""

import contextlib
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
from app import catalog_taxonomy  # noqa: E402
from app import product_type as pt  # noqa: E402
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


# ------------------------------------------------------------------ the title table
# Real catalog titles (a few short synthetic ones mark an exception), labelled by
# hand. Each is also checked lowercased and uppercased.
TITLES = [
    # apparel
    ("Hanpceirs Women's Cap Sleeve 1950s Retro Vintage Cocktail Swing Dresses with Pocket Purple 2X", "apparel"),
    ("Corala Women Solid Long Sleeve Wide Leg Long Jumpsuits Romper Pants with Belt,Black,X-Large", "apparel"),
    ("Siam Secrets Textured White Cotton Mens Shirt Mandarin Collar Short Sleeve Kurta XL", "apparel"),
    ("Timeson Women Dress Shirts Short Sleeve V Neck Pleated Front Tunic Top for Juniors Large Green", "apparel"),
    ("Karrimor Womens 2 Pack Running Socks Turquoise/Fusch Ladies 4-8", "apparel"),
    ("NCAA Georgia Bulldogs T-Shirt - Chevron Pattern - Georgia Logo - How Bout Them Dawgs (Small)", "apparel"),
    ("Romwe Women's Elegant Square Neck Floral Applique Mesh Sleeve Bodysuit Jumpsuit White M", "apparel"),
    ("DANTILE Men's Capri Plaid Shorts Cotton Linen, Red Black, Tag Size 5XL = W40", "apparel"),
    ("Zeagoo Womens A-Line Sleeveless Pleated Little Cocktail Party Dress, Orange_1, Large", "apparel"),
    ("DRESSIS Womens Round Neck Sleeveless Racerback Maxi Tank Dress Khaki S", "apparel"),
    ("BIADANI Women Long Sleeve Classic Lightweight Front Pockets Cardigan Teal X-Large", "apparel"),
    ("Plus Size Palazzo Pant,13696PPUR3X,Purple,3X", "apparel"),
    ("Nike Women Dri-fit Four Pleated Tennis Skort (XLarge, Light Retro)", "apparel"),
    ("Sportrendy Men's Slim Fit Casual Button Down Shirt JZS071 Black L", "apparel"),
    ("Jed North Women's Athletic Gym Yoga Fitness Workout Leggings", "apparel"),
    ("Mens Real Leather Biker Brando Style Jacket Popular Zip Up Coat Johnny Red (LARGE)", "apparel"),
    ("COCOSHIP Red and White Polka dot Retro Vintage One Piece Swimwear Swimsuit 44(FBA)", "apparel"),
    ("Leonisa Full Coverage Underwire Bra for Women - Contouring Back Smoothing Bra Black", "apparel"),
    ("Gihuo Women's Winter Sherpa Lined Pullover Hooded Sweatshirt Hoodie (Grey, Medium)", "apparel"),
    ("Aiyana Gold Mermaid V-Neck Backless Long Bridesmaid Dresses Sequins Wedding Party Gown 18W", "apparel"),
    ("California Costumes Women's Malice In Wonderland Costume, Black/White, Medium", "apparel"),
    ("Columbia Women's Bryce Peak Pant", "apparel"),
    ("HUE Utopia Ladies Denim Legging (Midnight, Medium)", "apparel"),
    ("Men's Cotton Boxer Briefs Breathable Pouch Underwear 5 Pack Mixed color US L/Tag Asia XL", "apparel"),
    ("Levi's Men's 527 Slim Boot Cut Jeans", "apparel"),
    ("Women's High Waist Paper Bag Waist Trousers with Belt", "apparel"),
    ("Boys Ring Bearer Outfit Tuxedo with Bow Tie and Suspenders", "apparel"),
    ("Toddler Girls Floral Sundress with Headband Summer Outfit", "apparel"),
    ("Opaque Control Top Tights for Women", "apparel"),
    ("Hanes Comfort Soft Brief Panty - 5 Pack (46HUSB) 5/WhiteDenimBlueRoyalty", "apparel"),
    ("Women's Long-Sleeve Oxford Shirt", "apparel"),
    ("Mens Hiker Crew Socks Merino Wool", "apparel"),
    # 'bow tie' / 'handkerchief' as a neckline or hem; the garment is the product
    ("Bow Tie Neck Long Sleeve Blouse", "apparel"),
    ("Romwe Women's Casual Cap Sleeve Bow Tie Blouse Top Shirts Navy S", "apparel"),
    ("Handkerchief Hem Tunic Top", "apparel"),
    ("Zeagoo Women's High Low Solid Print Black Handkerchief Hemline Midi A Line Skirt,X-Large", "apparel"),
    # a belt joined to a garment, or named before it, is part of the garment
    ("Wrap Dress with Waist Belt", "apparel"),
    ("Nihsatin Vintage Polka Dot Dress 1950s Waist Belt Cocktail Party Swing Dress White", "apparel"),
    ("Women Maxi Long Tulle Skirt Floor Length High Waist Bowknot Belt for Bridal Wedding Tutu Skirts", "apparel"),
    # socks stay socks next to a shoe word that says what they are for
    ("PUMA Women's 8 Pair No Show Sneaker Socks - 5-9.5", "apparel"),
    ("Areke No Show Socks Women for Flats, Ultra Low Cut Cotton Hidden Non Slip Boat Liner Sock", "apparel"),
    ("Bestjybt Womens Short Boots Socks Crochet Knitted Boot Cuffs Leg Warmers Socks (Gray)", "apparel"),
    ("Women's Half Slip Skirt", "apparel"),
    # singular 'short' / 'jean' name the garment (192 such rows were unknown) ...
    ("Nike Women's Training Short Black/Black Heather Size Large", "apparel"),
    ("Gloria Vanderbilt Women's Avery Pull On Skimmer Short, Fairfax, 14", "apparel"),
    ("Levi's Women's Straight Leg Jean", "apparel"),
    # ... but not a slogan or a 'short sleeve'
    ("Life is Short Funny T-Shirt", "apparel"),
    # outerwear nouns the coarse rule lacked
    ("Arssm Mens Corduroy Shacket Contrast Color Long Sleeve Patchwork Button Down Shirt Shacket(Blue-M)", "apparel"),
    ("Men's Wool Blend Overcoat", "apparel"),
    ("Women's Hooded Puffer Coat", "apparel"),
    # a garment named before a face cover is the product
    ("Long Sleeve Shirt with Attached Face Cover", "apparel"),
    # footwear
    ("Converse Kids Chuck Taylor All Star Madison Low Top Sneaker (13, Dolphin/Barley Rose/White)", "footwear"),
    ("Western Rhinestone Bling Concho Flip Flops in 5 Colors S061 (X-Large, Black)", "footwear"),
    ("Padders Women's Ankle Boots", "footwear"),
    ("BININBOX Girls Bling Jelly Kids Sandals Bowknot Girls Dress Shoes, Gold, 9 Toddler", "footwear"),
    ("Kitulandy Girls Boys Slippers Warm Winter Cotton House Kids Cozy Cute Dinosaur Indoor Shoes", "footwear"),
    ("BareTraps Women's Bt Drennan Ankle Bootie", "footwear"),
    ("Clarks Men's Casual Pitney Limit Oxford, Navy, 10.5 M US", "footwear"),
    ("Rugged Outback Mens Brown Men's Alpine Waterproof Hiker 6.5 Regular", "footwear"),
    ("Timberland Garrison Trail Waterproof Mid Fabric Hiker Black/Brown 10 D (M)", "footwear"),
    ("Easy Premium Orthopedic Leather Loafer (8.5,Black)", "footwear"),
    ("Crocs Kids' Crocband R2-D2 Lights Clog, light grey, 9 M US Toddler", "footwear"),
    ("Steve Madden Cally Mule Cognac Suede 9 M", "footwear"),
    ("NINE WEST Emmala Pump Black Leather 7.5 M", "footwear"),
    ("Riley Cap Toe Lace-Up Oxford (41 M (B), Black)", "footwear"),
    ("Florsheim Men's Kenmoor Wingtip Brogues", "footwear"),
    ("Cole Haan Men's Plain Toe Derby", "footwear"),
    ("Style & Co Radleyy Women US 9.5 Yellow Open Toe Wedge Heel", "footwear"),
    ("Michael Kors Farrah Flat Bootie Luggage 8.5 M", "footwear"),
    ("Merrell Men's Moab 2 Waterproof Hikers", "footwear"),
    # water shoes sold as 'aqua socks', sock-knit sneakers, and slip-ons: not socks/slips
    ("RAOXYE Kids Water Shoes Quick Dry Barefoot Aqua Socks for Beach Swimming Pool", "footwear"),
    ("JABASIC Kids Walking Shoes Boys Girls Breathable Slip On Knit Sock Sneakers (Blk/Blk,3)", "footwear"),
    ("WAVE Girls Swim Shoe Aqua Socks (3, Pink)", "footwear"),
    ("Women's Ankle Rain Boots with Removable Boot Socks Short Garden Rainboots", "footwear"),
    ("VANS Stormy Bird Slip-On Unisex | Black/True White (6.5)", "footwear"),
    ("Converse Kids Girl's Chuck Taylor All Star Stretch Lace Slip-on (Little Kid/Big Kid)", "footwear"),
    ("Beast Fashion Sasha-02 Pointy Toe Slip On Shiny Patent Stiletto Heel Slip On Pump Black 6", "footwear"),
    ("Breckelle's Talia-01 Pointed Toe Slip On Ballet Flat", "footwear"),
    # a shoe named by model alone is footwear by its US width size or a known
    # running / sneaker model (616 such rows were unknown)
    ("NINE WEST Rainford Pink 9 M", "footwear"),
    ("Juicy Couture Calli White 8 B", "footwear"),
    ("Nike Air Max Plus (Girls Grade School) (5, Black/White/Vivid Pink)", "footwear"),
    ("Brooks Men's Adrenaline GTS 21", "footwear"),
    ("ASICS Women's Gel-Kayano 28", "footwear"),
    ("New Balance Men's 990v5", "footwear"),
    ("UGG Women's Classic Short II Boot, Chestnut, 7 B (M) US", "footwear"),   # short = shaft
    # 'dress' modifies the shoe, not a garment
    ("City Classified Comfort Pointed Toe Cross Strap Mid Stiletto Heel Dress Pump (6.5 B(M) US, Black Nubuck)", "footwear"),
    ("Girls Patent Mary Jane School Shoes", "footwear"),
    ("Women's Pointed Toe Slingback Kitten Heel", "footwear"),
    # jewelry
    ("Michael Kors Rose Goldtone Pink Mother Of Pearl & Clear Pave Stud Earrings", "jewelry"),
    ("Genuine Tiger Eye, Carnelian, Aventurine, Amethyst, and Rose Quartz Chip Stretch Bracelet", "jewelry"),
    ("SHEGRACE Chic 925 Sterling Silver Fresh Water Pearl Mermaid Pendant Necklace", "jewelry"),
    ("Masedy 4Pcs Waist Belt Belly Chains for Women Sexy Body Chain Bikini Beach Jewelry Set", "jewelry"),
    ("14K Yellow Gold 9-9.5mm Round White Freshwater Cultured Pearl Heart Design Lever-back Earrings", "jewelry"),
    ("Sterling Silver Pear Cut Genuine London Blue Topaz Halo Ring (1.1 CT.T.W)", "jewelry"),
    ("Besteel Stainless Steel Womens Mens Plain Wedding Band Ring Polished 4mm Size 9", "jewelry"),
    ("Boho Beach Barefoot Sandals Foot Jewelry Anklet for Women", "jewelry"),
    ("Gold Drop Earrings for Wedding Dress", "jewelry"),
    ("Sujak Military Items Silver Knuckles Replica Hat Lapel Pin 1 1/2 Inch IVANP452", "jewelry"),
    ("Sterling Silver Stud Earrings with Dust Cover", "jewelry"),   # a dust cover is packaging
    # bag
    ("Packing Cubes 8pcs my FL Backpack Organizers Set for Carry on Travel Bag Luggage Cube", "bag"),
    ("eBags Villa Cross Body 2.0 (Slate)", "bag"),
    ("Women's Leather Tote Handbag", "bag"),
    ("Michael Kors Jet Set Travel Large Crossbody", "bag"),
    # headwear
    ("Yonex Bandana Headband Color - White", "headwear"),
    ("Foetest Adjustable Cap Baseball Cap Outdoors Casual Hat Sports Cap Leather Hat Solid Color", "headwear"),
    ("Kentucky Derby Fascinator Hat", "headwear"),
    ("EAGLE CREST POW-MIA Logo with Text Cap, Black", "headwear"),
    ("Branded Bills Vintage Old Glory PVC Snapback (Flat Trucker, Multicam)", "headwear"),
    ("Diesel Power Plus Peterbilt Stocking Cap hat Beanie red Black pete White Embroidered Logo", "headwear"),
    ("Trapper Hat with Face Cover Winter Ushanka", "headwear"),   # the hat is named first
    # accessory
    ("Black HD Vision Aviator Sunglasses", "accessory"),
    ("UGG Womens Sheepskin And Leather Mixed Glove, Chestnut, Size Small", "accessory"),
    ("18 mm 2 Piece Military NATO Style Nylon Watch Band Strap Rainbow Quick Release Pins", "accessory"),
    ("Galleria Reverse Close Folding Umbrella, Beach Scene with art on both sides.", "accessory"),
    ("Swatch Women's Scintillating YLS450 White Rubber Swiss Quartz Dress Watch", "accessory"),
    ("American Flag Patter 4th of July or Memorial Day Infinity Scarf", "accessory"),
    ('Rounded Edge Heel Bar Buckle Casual Jean Suede Leather Belt 1 1/2" Wide (Blue, 34)', "accessory"),
    ("YLOKO 18pcs Shoe Charms for Croc Clog Sandals", "accessory"),
    ("Waxed Round Dress Shoelaces for Oxford Shoes", "accessory"),
    ("Jewelry Box Organizer for Women Earrings Rings Necklaces", "accessory"),
    ("Men's Bow Tie and Pocket Square Set", "accessory"),
    # a belt is the product when a garment word says what it is for
    ("Men's Classic Stitched Leather Dress Belt", "accessory"),
    ("Men's Genuine Leather Ratchet Dress Belt with Automatic Buckle", "accessory"),
    ("Women's Leather Belt for Jeans Pants", "accessory"),
    ("BABEYOND Metal Waistband Skinny Belt Elastic Waist Strap Stretchy Belt for Dress", "accessory"),
    ("CARDANRO Women's Genuine Leather Jeans Dress Belt with Metal Buckle for Business and Casual", "accessory"),
    ("Michael Kors Mens Silk Pindot Neck Tie Navy O/S", "accessory"),
    ("Brooks Brothers Mens Purple Bar Stripe Silk Tie", "accessory"),
    # face masks, neck gaiters and balaclavas are accessories, not headwear (2026-10-19:
    # 810 of 1,432 were typed headwear, so a hat query or a Hats tick showed masks)
    ("Skull Face Mask for Women Men Dust Wind UV Sun Protection Seamless Bandana Face Mask for Rave Festival", "accessory"),
    ("Men Mask Bandanas 3D Printed Balaclava Neck Scarf Face UV Neck Headwear Gaiter for Adult Teen", "accessory"),
    ("Genrics Face Cover Sports Balaclavas for University Football or Basketball Fans Soft Mouth Cover", "accessory"),
    ('(2-in-1) Pearl Beaded Mask Chain 28" Face mask holder, Beads chains, Eyeglass chains', "accessory"),
    # a belt a garment or leather word describes ("dress work belt", "top grain")
    ("Enimay Men's Black Genuine Leather Ratchet Belt Formal Dress Work Belt Smooth G Italy Large A20", "accessory"),
    ('Womens Belts for Jeans, Fashion Brown Waist Belt Cute Dress Belt for Women 105cm fit for waist:34"', "accessory"),
    ("Men's Top Grain Leather Belt with Prong Buckle", "accessory"),
    ("igvean Stocking Stuffers for Women Best Friend Christmas Gifts Friendship Gifts for Female Friends Keychains", "accessory"),
    # unknown: a size string with no width letter, D-ring hardware
    ("Zoe and Zac Hot Pink Smooth Girls' Fae String Tie Flat 1.5 Regular", "unknown"),
    ("YT018XXL Peace Love Yoga Tie Dye", "unknown"),
    ("Vivipow Colorful Skull Pattern Rectangle Non-Slip Rubber Mousepad Gaming Mouse Pad", "unknown"),
    ("Ridlee Black / 8.5", "unknown"),
    ('Amanaote Black 1" Inner Diameter D Ring D Rings Non Welded Pack of 10', "unknown"),
]


def _all_rules():
    """Every (pattern, needles) rule in the module, for the needle-soundness checks."""
    def is_rule(v):
        return (isinstance(v, tuple) and len(v) == 2 and isinstance(v[0], re.Pattern)
                and isinstance(v[1], tuple))
    rules = [v for v in vars(pt).values() if is_rule(v)]
    rules += [r for r, _t, _s in pt._OVERRIDES]
    rules += [r for _t, _s, r in pt._STRONG]
    rules += [r for _s, r in pt._APPAREL]
    rules += [r for _k, r in pt._FINE_APPAREL + pt._FINE_FOOTWEAR]
    rules += [heads for _word, heads in pt._TOP_MODIFIERS.values()]
    return rules


def _hidden_matches(texts):
    """(needles, text) pairs where a pattern matches but no needle is present: a
    needle that hides a real match would silently change a classification."""
    bad = []
    for pattern, needles in _all_rules():
        for t in texts:
            if pattern.search(t) and not any(n in t for n in needles):
                bad.append((needles[:3], t))
    return bad


def test_title_table_is_big_enough_and_covers_every_type():
    assert len(TITLES) >= 70, len(TITLES)
    assert {ty for _t, ty in TITLES} == set(pt.FASHION_TYPES) | {pt.UNKNOWN}


def test_classify_title_table_any_case():
    wrong = []
    for title, want in TITLES:
        for variant in (title, title.lower(), title.upper()):
            got = pt.classify_title(variant)[0]
            if got != want:
                wrong.append((variant, want, got))
    assert not wrong, wrong[:5]


def test_classify_title_exceptions():
    cases = {
        "Women's Cap Sleeve Maxi Dress": "apparel",            # cap sleeve is not a hat
        "Cole Haan Cortland Cap Toe Ballet II": "unknown",     # nor is a cap toe
        "Levi's 527 Boot Cut Jeans": "apparel",                # boot cut is not a boot
        "Paper Bag Waist Shorts": "apparel",                   # paper bag waist is not a bag
        "Ring Bearer Suit for Boys": "apparel",                # ring bearer is not a ring
        "Girls Sundress with Headband": "apparel",             # garment set beats headband
        "Floral Dress with Matching Earrings": "jewelry",      # ...but not an earring set
        "Bikini Harness Necklace Waist Belt Belly Body Chain 2 PCS": "jewelry",  # nor a body chain
        "Barefoot Sandals for Beach Wedding": "jewelry",       # barefoot sandals are jewellery
        "Women's Sheer Tights": "apparel",                     # tights are apparel
        "Hiking Boot Socks": "apparel",                        # socks beat boots
        "Shoe Charms for Crocs": "accessory",                  # shoe charms are accessories
        "Men's Boxer Briefs with Pouch": "apparel",            # underwear pouch is not a bag
        "Leather Coin Pouch": "bag",                           # a real pouch is
        "Halo Ring Engagement Set": "jewelry",                 # 'o ring' lookbehind is word-bounded
        "Rubber O-Ring Kit": "unknown",                        # ...an O-ring still is not jewellery
        "Earrings for Wedding Dress": "jewelry",               # a strong noun beats the garment
    }
    wrong = {t: (want, pt.classify_title(t)[0]) for t, want in cases.items()
             if pt.classify_title(t)[0] != want}
    assert not wrong, wrong


def test_classify_title_2026_10_19_exceptions():
    """The vetoes of the 2026-10-19 rules: each line is a false positive the catalog
    diff showed (old -> new type over every Fashion title)."""
    cases = {
        # a singular 'short' after a word is the garment, except as a length, a hair or
        # wig style, a boot shaft or a phrase
        "Men's Mesh Pocket Short": "apparel",
        "Short Sleeve Henley": "apparel",                     # the henley, not a short
        "Wig Short Blonde Bob": "unknown",
        "Women's Short Kurta Cotton": "apparel",              # a kurta, typed by its noun
        "Mitchell & Ness Pistons Blank Front Short Hook Snapback": "headwear",
        "Bearpaw Women's Elle Short Winter Boot": "footwear",
        # a singular 'jean' is the garment, not a jean belt or a designer's name
        "Casual Jean Suede Leather Belt": "accessory",
        "Belt Jean Women's": "accessory",
        "Jean Paul Gaultier Perfume Bottle Pendant": "jewelry",
        # 'dress' modifies the shoe, belt, watch or suit after it
        "Men's Dress Belt Reversible": "accessory",
        "Women's Pointed Toe Dress Pump": "footwear",
        "Men's Slim Fit Dress Suit": "apparel",
        # 'stocking' is no sock in a stocking cap or a stocking stuffer
        "Toddler Fleece Lined Beanie Stocking Cap": "headwear",
        "Stocking Stuffer Crystal Tennis Bracelet": "jewelry",
        # a face cover after a hat or a garment is a feature of it; a mask chain is no jewellery
        "Long Sleeve Fishing Shirt with Face Cover": "apparel",
        "Face Mask Lanyard Chain for Women": "accessory",
        "Washable Cotton Face Mask 3 Pack": "accessory",
        "Neck Gaiter Face Cover UV Protection": "accessory",
        # a top-grain belt or watch band is leather, not a top
        "Top Grain Leather Watch Band 42mm": "accessory",
        # US width sizes and kids' size ranges name a shoe; a bare number does not
        "Steve Madden Troopa Black Leather 8.5 M US": "footwear",
        "Stride Rite Kids Baby Jack (Little Kid/Big Kid)": "footwear",
        "Ridlee Black 8.5": "unknown",
    }
    wrong = {t: (want, pt.classify_title(t)[0]) for t, want in cases.items()
             if pt.classify_title(t)[0] != want}
    assert not wrong, wrong


def test_watches_and_watch_bands_are_never_jewelry():
    """2026-10-27: a title the watch reader (catalog_taxonomy.watch_kind) calls a watch or
    a watch band types accessory even when it says "bracelet" (500 Fashion rows moved,
    30 of 30 sampled were watches or bands); real jewellery and the watch reader's other
    kinds keep their reading. Real catalog titles, some shortened."""
    cases = {
        # watches sold "with bracelet" / as a "bracelet watch"
        "Kenneth Cole New York Men's Stainless Steel Case Stainless Steel Bracelet Silver "
        "Casual Watch": ("accessory", "accessory"),
        "Seiko neo Classic Mens Analog Japanese Quartz Watch with Stainless Steel Bracelet":
            ("accessory", "accessory"),
        "Narmi Rhinestone Embellished Cuff Bangle Watch - Silver": ("accessory", "accessory"),
        "Unisex Steampunk Antique Vintage Rib Quartz Pocket Watch Fashion Pendant + Chain "
        "Necklace": ("accessory", "accessory"),
        # bands for a watch
        "Compatible with Apple Watch (38/40 mm) Series 5, 4, 3, 2, 1 // Leather Replacement "
        "Bracelet Strap Wristband": ("accessory", "accessory"),
        "bayite Metal Bands Compatible Fitbit Versa, Replacement Bracelet Rhinestone Band":
            ("accessory", "accessory"),
        # untouched: jewellery with no watch, and the reader's other kinds
        "Gold Tennis Bracelet for Women": ("jewelry", "jewelry"),
        "Women's Gold Tone Celtic Cross and Watch Pendant Necklace": ("jewelry", "jewelry"),
        "Mystic Metals Body Jewelry Embedded Steampunk Watch Parts Plugs - 7/16 Inch":
            ("jewelry", "jewelry"),
        "Kalawen Watch Winder for Rolex": ("accessory", "storage/care"),
        # known residuals, pinned as read: jewellery made of a watch movement reads as a
        # watch (jewelry is right); a band whose last noun is its clasp reads as a watch
        # part, so it stays jewelry (accessory is right)
        "Glazed Black Cherry Steampunk Oval Watch Movements Ring Mechanical Parts":
            ("accessory", "accessory"),
        "Brushed Solid Stainless Steel Link Watch Band Strap 20mm Replacement Bracelet "
        "Butterfly Clasp": ("jewelry", "jewelry"),
    }
    wrong = {t: (want, pt.classify_title(t)) for t, want in cases.items()
             if pt.classify_title(t) != want}
    assert not wrong, wrong
    for title in cases:   # the local backend's lowercase titles read the same
        assert pt.classify_title(title.lower()) == cases[title], title


def test_singular_footwear_nouns():
    shoes = ["Penny Loafer", "Wooden Clog", "Wingtip Brogue", "Suede Ankle Bootie",
             "Lace-Up Oxford", "Plain Toe Derby", "Trail Hiker", "Waterproof Hikers",
             "Leather Mule", "Pointed Toe Pump", "Block Heel"]
    assert all(pt.classify_title(s)[0] == "footwear" for s in shoes), \
        [(s, pt.classify_title(s)) for s in shoes if pt.classify_title(s)[0] != "footwear"]
    # the weak ones never beat a garment or a stronger noun, and a garment part vetoes them
    not_shoes = {
        "Men's Oxford Button Down Shirt": "apparel",
        "Kentucky Derby Church Hat": "headwear",
        "Mule Deer Hunting T-Shirt": "apparel",
        "Hiker Crew Sock": "apparel",
        "Hiker Backpack 30L": "bag",
        "Kentucky Derby Garden Flag": "unknown",
        "UltraClub Ladies' Classic Wrinkle-Free Long-Sleeve Oxford": "unknown",
        "Pump Replacement Sleeve": "unknown",
        "Clearly Adjustable Heel Lifts": "unknown",
    }
    wrong = {t: (w, pt.classify_title(t)[0]) for t, w in not_shoes.items()
             if pt.classify_title(t)[0] != w}
    assert not wrong, wrong


def test_classify_title_odd_inputs():
    for blank in (None, float("nan"), pd.NA, "", "   "):
        assert pt.classify_title(blank) == ("unknown", ""), blank
    assert pt.classify_title("Women\u2019s Ankle Boots") == pt.classify_title("women's ankle boots")
    assert pt.slot_type(12345) == "unknown"


def test_classify_only_returns_known_names():
    seen = {pt.classify_title(t) for t, _ in TITLES}
    assert {ty for ty, _ in seen} <= set(pt.FASHION_TYPES) | {pt.UNKNOWN}
    assert not {"other-accessory", "unclassified"} & {x for pair in seen for x in pair}


def test_needles_never_hide_a_match_on_the_tables():
    texts = [pt._normalise(t) for t, _ in TITLES]
    texts += [pt._normalise_query(q) for q in REQUESTED] + [pt._normalise_query(q) for q in NONE]
    texts += [pt._normalise_query(q) for q in COLLISIONS]
    texts += [pt._normalise_query(q) for q in RESIDUALS]
    texts += [pt._normalise(t) for t in FINE] + [pt._normalise(t) for t, *_ in GENDERS]
    assert not _hidden_matches(texts), _hidden_matches(texts)[:5]


def test_is_fashion():
    for yes in ("Fashion", "fashion", " FASHION ", "Fashion\n", {"bsns_vrtcl_name": "fashion"}):
        assert pt.is_fashion(yes), yes
    for no in ("Electronics", "grocery", "", None, float("nan"), np.nan, pd.NA, 0,
               "Fashion Accessories", {"bsns_vrtcl_name": None}, {}):
        assert not pt.is_fashion(no), no


# ------------------------------------------------------------------ the shopper's ask
APPAREL = frozenset({"apparel"})
ACCESSORY = frozenset({"accessory"})
REQUESTED = {
    "cloths for a beach wedding in summer": APPAREL,
    "clothes for a beach wedding": APPAREL,
    "what should I wear to a beach wedding": APPAREL,
    "something to wear with my black boots": APPAREL,
    "what to wear with white sneakers": APPAREL,
    "clothes to go with my new hiking boots": APPAREL,
    "shirt to match my watch": APPAREL,
    "dress to go with gold sandals": APPAREL,
    "socks for hiking boots": APPAREL,
    "what shoes to wear with a red dress": frozenset({"footwear"}),
    "earrings to wear to a wedding": frozenset({"jewelry"}),
    "earrings for a wedding dress": frozenset({"jewelry"}),
    "bag to match a wedding guest dress": frozenset({"bag"}),
    "dress shoes for men": frozenset({"footwear"}),
    "dress and sandals for a beach wedding": frozenset({"apparel", "footwear"}),
    "gold drop earrings for a wedding": frozenset({"jewelry"}),
    "waterproof hiking boots": frozenset({"footwear"}),
    # beyond the spec's table
    "something for her to wear to a black-tie gala": APPAREL,   # head 'something' -> whole query
    "summer beach wedding outfit dress": APPAREL,               # the garment, not 'outfit'
    "cozy clothes for a lazy sunday at home": APPAREL,
    "how to dress for a beach wedding": APPAREL,
    "cloths and shoes for a beach wedding": frozenset({"apparel", "footwear"}),
    "hat and gloves": frozenset({"headwear", "accessory"}),
    "shirt, shorts + sandals": frozenset({"apparel", "footwear"}),
    "boho summer dress": APPAREL,
    "wedding guest dress": APPAREL,
    "summer clothes": APPAREL,
    "for men: dress shoes": frozenset({"footwear"}),            # a leading connector is no cut
    # a belt or tie a garment word describes is the accessory, not the garment
    "men's dress belt": ACCESSORY,
    "men's black dress belt": ACCESSORY,
    "genuine leather belt for men": ACCESSORY,
    "belt for jeans": ACCESSORY,
    "tie for a suit": ACCESSORY,
    "neck tie to match a blue shirt": ACCESSORY,
    "vans slip on": frozenset({"footwear"}),
    # 'accessories' is the whole accessory family; a narrower noun still wins
    "accessories for a black dress": pt.ACCESSORY_FAMILY,
    "accessories for a wedding dress": pt.ACCESSORY_FAMILY,
    "accessories to match my navy suit": pt.ACCESSORY_FAMILY,
    "dress accessories": pt.ACCESSORY_FAMILY,
    "summer clothes and accessories": pt.ACCESSORY_FAMILY | APPAREL,
    "wedding guest dress and accessories": pt.ACCESSORY_FAMILY | APPAREL,
    "clothes, shoes and accessories for a cruise":
        pt.ACCESSORY_FAMILY | APPAREL | frozenset({"footwear"}),
    "hair accessories": frozenset({"headwear"}),
    # The head noun of a query is its RIGHTMOST type noun: a jewellery, watch, boot or
    # sandal word in front of the garment is the setting (these came back as that
    # type, and the page showed no clothes)
    "watch party clothes": APPAREL,
    "boot camp clothes": APPAREL,
    "sandal weather dresses": APPAREL,
    "engagement ring photoshoot dress": APPAREL,
    "dress shoes": frozenset({"footwear"}),
    "boot cut jeans": APPAREL,
    "cap sleeve top": APPAREL,
    "shoe charms": ACCESSORY,
    "apple watch band": ACCESSORY,
    "men's leather dress belt": ACCESSORY,
    "bow tie": ACCESSORY,
    "tie dye shirt": APPAREL,
    "hair clips": frozenset({"headwear"}),
    "wedding dress earrings": frozenset({"jewelry"}),
    "d ring belt": ACCESSORY,                          # 'ring belt' read without its 'd'
    "kentucky derby hat": frozenset({"headwear"}),
    "shirt bow tie set": APPAREL,                      # a garment set, as in titles
    "watch party accessories": pt.ACCESSORY_FAMILY,    # was {accessory}: 'watch' modifies
    # ...and a wear-verb ask reads only what precedes it; the occasion after it is no ask
    "what to wear to a ring ceremony": APPAREL,
    "what should I wear to a ring ceremony": APPAREL,
    "what to wear to an engagement ring ceremony": APPAREL,
    "what should I wear to a watch party": APPAREL,
    "what to wear to a watch party": APPAREL,
    "what shoes to wear": frozenset({"footwear"}),
    "what should my husband wear to a wedding": APPAREL,
    "wear to work shoes": frozenset({"footwear"}),     # the part ENDS on the product
    # Conjunctions whose every part is typed, or is only a modifier, keep the request
    "jeans and a t-shirt": APPAREL,
    "shirt and tie": frozenset({"apparel", "accessory"}),
    "black and white dress": APPAREL,
    "women's plus size dresses": APPAREL,              # ' plus size' is no conjunction
    "dress plus sandals": frozenset({"apparel", "footwear"}),
    "dress, size 8": APPAREL,
    "red dress, long sleeve": APPAREL,
    "gold and silver earrings": frozenset({"jewelry"}),
    "h&m dress": APPAREL,
    # The noun named LAST wins even right after a non-garment noun: the title rules
    # let the modifier win the pair ('sneaker tee' -> footwear, 'rings t-shirt' ->
    # jewelry, 'tie attire' -> accessory) and the page showed no clothes
    "black tie attire": APPAREL,
    "black tie attire for men": APPAREL,
    "black-tie attire": APPAREL,
    "white tie attire": APPAREL,
    "derby attire": APPAREL,
    "lord of the rings t-shirt": APPAREL,
    "lord of the rings hoodie": APPAREL,
    "lord of the rings shirt men": APPAREL,
    "lord of the rings necklace": frozenset({"jewelry"}),
    "sneaker tee": APPAREL,
    "sneaker shirts": APPAREL,
    "sneaker hoodie": APPAREL,
    "sneaker tees to match jordan 11": APPAREL,
    "cowboy boots dress": APPAREL,
    "clutch dress": APPAREL,
    "earring tee": APPAREL,
    "hat tee": APPAREL,
    "ring tee": APPAREL,
    "watch tee": APPAREL,
    "suit tie": ACCESSORY,                             # ...the same rule, the other way
    "shoe bag": frozenset({"bag"}),
    "earring jackets": frozenset({"jewelry"}),         # a compound the left noun heads
    "shoe wardrobe": frozenset({"footwear"}),
    "black tie": ACCESSORY,                            # a bare black tie is the tie
    # The occasion after a wear-verb QUESTION is never the product, even when it ends on
    # a type noun (only the compound 'wear to work shoes' keeps its last noun)
    "what to wear to a black tie": APPAREL,
    "what to wear to black tie": APPAREL,
    "what should i wear to a black tie": APPAREL,
    "what to wear to a white tie": APPAREL,
    "what do men wear to black tie": APPAREL,
    "what to wear to the derby": APPAREL,
    "what to wear to a derby": APPAREL,
    "what should I wear to the derby": APPAREL,
    "what to wear to oxford": APPAREL,
    "what to wear to a boxing ring": APPAREL,
    # ...and the occasion goes on across a conjunction; a part ending on its type noun
    # is a product again
    "what to wear to a wedding and a ring ceremony": APPAREL,
    "what to wear to church and brunch": APPAREL,
    "what to wear to a wedding & reception": APPAREL,
    "what to wear under a cap and gown": APPAREL,
    "earrings to wear to a wedding and a clutch": frozenset({"bag", "jewelry"}),
    "what to wear to a wedding and shoes": frozenset({"apparel", "footwear"}),
    # Adjective parts are no second product (the untyped-part rule read 'light',
    # 'breezy', 'boho', ... as products and dropped the clothing request)
    "light, breezy clothes for a beach wedding": APPAREL,
    "light and airy clothes for a beach wedding in summer": APPAREL,
    "airy, lightweight clothes for a humid beach wedding": APPAREL,
    "flowy, breathable dress for a beach wedding": APPAREL,
    "relaxed and elegant beach wedding guest dress": APPAREL,
    "semi-formal and breezy beach wedding attire": APPAREL,
    "bright and colorful clothes for a beach party": APPAREL,
    "beach wedding clothes, light and airy": APPAREL,
    "boho and chic dresses": APPAREL,
    "pastel and floral dresses": APPAREL,
    "stretchy and comfy jeans": APPAREL,
    "christmas and holiday sweaters": APPAREL,
    "red dress, midi, v-neck": APPAREL,
    "wedding guest dress, tea length": APPAREL,
    "dresses in neutral and earthy tones": APPAREL,     # the 'in' phrase goes on
    "shirts that are breathable and wrinkle free": APPAREL,
    "a dress that's light and flowy": APPAREL,
    "socks, 6 pairs": APPAREL,
    "rain and snow boots": frozenset({"footwear"}),     # ...nor a noun modifying the last
    "ankle and knee high boots": frozenset({"footwear"}),
    "mother and daughter matching dresses": APPAREL,
    "homecoming and prom dresses": APPAREL,
    "bridesmaid and flower girl dresses": APPAREL,
    "what to wear to a wedding and reception": APPAREL,
    "hat, wide brim": frozenset({"headwear"}),
    "dress, a-line": APPAREL,
    "bra, front closure": APPAREL,
    "earrings, hypoallergenic": frozenset({"jewelry"}),
    "boots, size 9, wide calf": frozenset({"footwear"}),
    "backpack, anti theft": frozenset({"bag"}),
    # a plain watch is still a Fashion accessory next to a garment
    "running shoes and a watch": frozenset({"footwear", "accessory"}),
    # 2026-10-30: a jewellery noun that names a watch or its band asks for what the rows
    # type accessory (their watch rule, test_watches_and_watch_bands_are_never_jewelry):
    # read as jewelry, every watch band ranked below every plain bracelet...
    "apple watch bracelet": ACCESSORY,
    "watch bracelet": ACCESSORY,
    "women's watch bracelet": ACCESSORY,
    "smart watch bracelet": ACCESSORY,
    "watch strap bracelet": ACCESSORY,
    "apple watch bracelet for women": ACCESSORY,
    "pocket watch necklace": ACCESSORY,
    "watch necklace": ACCESSORY,
    "watch bracelet to accessorize my outfit": ACCESSORY,
    "dress and watch bracelet": frozenset({"apparel", "accessory"}),
    # ...as does one FOR a watch, when the phrase after 'for' is the watch itself
    "bracelet for apple watch": ACCESSORY,
    "beaded bracelet for apple watch": ACCESSORY,
    "rose gold bracelet for apple watch 41mm": ACCESSORY,
    "bracelet for my husband's apple watch": ACCESSORY,
    "bracelet for galaxy watch 4 classic": ACCESSORY,
    "bracelet for my fitbit charge 5": ACCESSORY,
    # ...while jewellery for a person who likes watches, a charm, jewellery that is not a
    # band, a multi-item ask and plain jewellery keep jewelry
    "bracelet for my watch-loving dad": frozenset({"jewelry"}),
    "silver bracelet for my husband who collects watches": frozenset({"jewelry"}),
    "bracelet for a man who loves his apple watch": frozenset({"jewelry"}),
    "bracelet for watch lovers": frozenset({"jewelry"}),
    "necklace for my wife who loves watches": frozenset({"jewelry"}),
    "watch charm": frozenset({"jewelry"}),
    "bracelet with watch charm": frozenset({"jewelry"}),
    "earrings for apple watch": frozenset({"jewelry"}),
    "necklace for apple watch": frozenset({"jewelry"}),
    "earrings to match my watch": frozenset({"jewelry"}),
    "bracelet to match my apple watch": frozenset({"jewelry"}),   # 'to match' names no target
    "gold watch and bracelet set": frozenset({"accessory", "jewelry"}),
    "earrings and watch": frozenset({"accessory", "jewelry"}),
    "silver bracelet": frozenset({"jewelry"}),
    "charm bracelet": frozenset({"jewelry"}),
    "tennis bracelet": frozenset({"jewelry"}),
    "minimalist gold jewelry": frozenset({"jewelry"}),
    # A head that names nothing but an OUTFIT is a clothing ask (outfit bundles are off
    # since 2026-10-15, so these are flat pages whose garments must rank first); they
    # were None while an outfit bundle served them
    "beach wedding outfit": APPAREL,
    "linen outfit ideas": APPAREL,
    "breathable outfit for a humid day": APPAREL,
    "outfit for a job interview": APPAREL,
    "hiking outfit": APPAREL,
    "ring bearer outfit": APPAREL,                     # 'ring' is cancelled; the outfit is left
    "outfit for my daughter": APPAREL,
    "men's outfit for a wedding": APPAREL,
    "outfits for women": APPAREL,
    "outfit with jeans": APPAREL,
    "black and white outfit": APPAREL,                 # modifiers, no second product
    "casual and comfy outfit": APPAREL,
    # ...a type noun in the head still wins over the outfit around it
    "earrings to complete my outfit": frozenset({"jewelry"}),
    "earrings that match my outfit": frozenset({"jewelry"}),
    "shoes for my outfit": frozenset({"footwear"}),
    "bag to go with my outfit": frozenset({"bag"}),
    "accessories for my outfit": pt.ACCESSORY_FAMILY,
    # 'accessorize' asks for accessories FOR the garment it names
    "accessorize this dress": pt.ACCESSORY_FAMILY,
    "how to accessorize a black dress": pt.ACCESSORY_FAMILY,
    "accessorizing a wedding guest dress": pt.ACCESSORY_FAMILY,
    "how to accessorize a black dress with shoes": pt.ACCESSORY_FAMILY,
    # ...but a type named BEFORE the verb is the ask (review 2026-10-15: the boots
    # were ranked after every accessory and the reranker was told they violate it)
    "boots to accessorize my outfit": frozenset({"footwear"}),
    "boots that accessorize my outfit": frozenset({"footwear"}),
    "shoes to accessorize my dress": frozenset({"footwear"}),
    "ankle boots to accessorise a midi dress": frozenset({"footwear"}),
    "jewelry to accessorize a dress": frozenset({"jewelry"}),
    # A dress code, a style or an occasion role names clothing although no garment noun
    # does (2026-10-19: these were None, so the untyped window gave 10-13 of its 15
    # seats to one intent -- a bow tie, cargo pants). Was pinned None until 2026-10-19:
    # "black tie event", "black-tie event", "black tie optional", "white tie gala";
    # "black tie wear" was a residual (accessory).
    "black tie event": APPAREL,
    "black-tie event": APPAREL,
    "black tie optional": APPAREL,
    "white tie gala": APPAREL,
    "black tie wear": APPAREL,
    "black tie event men": APPAREL,
    "business casual for men": APPAREL,
    "business casual": APPAREL,
    "smart casual outfit ideas": APPAREL,
    "gift for my boyfriend who likes streetwear": APPAREL,
    "cocktail attire": APPAREL,
    "formal wear for men": APPAREL,
    "semi-formal attire for a wedding": APPAREL,
    "dress code for a gala": APPAREL,
    "bridesmaid": APPAREL,
    "mother of the bride": APPAREL,
    "groomsmen": APPAREL,
    "prom": APPAREL,
    "something warm to wear": APPAREL,
    "something warm for a ski trip": APPAREL,   # beyond the contract's 'to wear' (owner)
}
NONE = [
    "beach wedding", "gift for my wife", "cozy desk setup ideas", "wireless earbuds",
    "instant noodles", "never lose my keys again", "running gear for winter", "rain gear for hiking",
    "what do I need for a beach wedding", "first aid kit", "kit kat", "home office setup",
    "gaming setup under $500", "keep my coffee hot", "microfiber cloths",
    "cloths for cleaning", "", None,
    # The head names no type: a type noun AFTER the connector is the context, not the
    # ask (these came back as the context's type and the reranker was told the
    # earrings, clutches and neckties the shopper wanted violate the request)
    "something to go with my black dress", "something to go with my black boots",
    "laces for sneakers", "something to match my navy suit",
    # ...which also retires three former collisions
    "phone case with card holder", "microfiber cloths for cleaning glasses",
    "non-slip mouse pad",
    # A conjunction part that names a product no type covers drops the request: a
    # footwear or clothing request ranked the headlamp, hand warmers and phone pouch
    # (Electronics) below every Fashion row, out of the rerank window
    "hiking boots and a headlamp", "a headlamp and hiking boots",
    "winter coat and hand warmers", "rain jacket and a waterproof phone pouch",
    "waterproof phone pouch",                          # a device pouch is no Fashion bag
    "shoes and an outfit",                             # an outfit is more products
    # ...in either order (review 2026-10-15: these came back as the other part's type
    # alone, so every garment on the flat page was a type violator)
    "outfit and shoes for a wedding", "outfit and shoes", "wedding guest outfit and heels",
    "outfit plus sandals for the beach", "outfit and earrings for a party",
    "something for my outfit",                         # the outfit is after the connector
    # ...a bare noun right after the typed part, or a plural noun anywhere, is one too
    "hiking boots and headlamp", "hiking boots, headlamp and trekking poles",
    "dress and perfume", "a swimsuit and sunscreen", "jeans, ripped knees",
    # A non-Fashion product named with a Fashion noun, next to a Fashion item: the pool
    # is Fashion-majority, so the gate kept the request and buried the device rows
    "gym shorts and an apple watch band", "dress and apple watch band",
    "dress and a ring light", "hiking boots and a camera bag", "leggings and a laptop bag",
    "running shoes and a smartwatch", "suit and a smart watch",
    "running shorts and a smart ring", "yoga pants and a tea bag",
    "jacket and a fitbit band", "sleeping bag and hiking boots", "phone case and a dress",
    # a type noun its neighbour cancels is no noun
    "tie dye", "boot cut", "kentucky derby",
    # ...and the style / role readers need a style, not a word that looks like one, and
    # an ask for the role's clothes, not a gift or a prop for them
    "wear leveling", "ssd wear leveling", "something warm to drink",
    "bridesmaid proposal box", "groomsmen gifts", "gift for my bridesmaid",
    "prom night decorations",
]
# Known residuals, pinned so a change is seen (not the desired answers): a bare
# singular noun BEFORE the typed part, or between typed parts, reads as a modifier
# (the price of keeping "light, breezy clothes" a clothing ask); 'wear' as a noun is
# no clothing noun.
RESIDUALS = {
    # 'look' is no ask ("a gift that looks expensive"): the garment intents come from
    # the translator, with no type request behind them
    "complete my look": None,
    "office look for men": None,
    "headlamp and hiking boots": frozenset({"footwear"}),
    "sunscreen and a swimsuit": APPAREL,
    "dress, sunscreen and a hat": frozenset({"apparel", "headwear"}),
    # jewellery with a watch motif or a bling band? Read as the band, as such titles are
    # ("... Watch Jewelry Band" types accessory)
    "watch jewelry": ACCESSORY,
    "apple watch jewelry": ACCESSORY,
}
# Lexical collisions, documented: the function MAY return a Fashion type here, and
# main.py's pool gate (Fashion share of the retrieved pool < 0.5) neutralises it.
# Pinned so the gate tests in tests/test_ranking.py keep exercising a real request.
COLLISIONS = {
    "organic green tea bags": frozenset({"bag"}),
    "watch movies on my bedroom wall": frozenset({"accessory"}),
    "ring light": frozenset({"jewelry"}),
    "smart ring": frozenset({"jewelry"}),
    "apple watch": frozenset({"accessory"}),
    "laptop bag": frozenset({"bag"}),
    "camera bag": frozenset({"bag"}),
    "desk fan": frozenset({"accessory"}),
    "cable ties": frozenset({"accessory"}),
    "black cable ties": frozenset({"accessory"}),   # no dress code: 'black tie' needs one
    "black zip ties": frozenset({"accessory"}),
    "lens cap": frozenset({"headwear"}),
    "phone accessories": pt.ACCESSORY_FAMILY,
    "laptop accessories": pt.ACCESSORY_FAMILY,
}


def test_requested_types_table():
    wrong = {q: (sorted(w), sorted(pt.requested_types(q) or [])) for q, w in REQUESTED.items()
             if pt.requested_types(q) != w}
    assert not wrong, wrong


def test_requested_types_none():
    wrong = {q: sorted(pt.requested_types(q)) for q in NONE if pt.requested_types(q) is not None}
    assert not wrong, wrong


def test_requested_types_documented_collisions():
    wrong = {q: (sorted(w), pt.requested_types(q)) for q, w in COLLISIONS.items()
             if pt.requested_types(q) != w}
    assert not wrong, wrong
    assert all(r <= set(pt.FASHION_TYPES) for r in COLLISIONS.values())


def test_requested_types_known_residuals():
    wrong = {q: (sorted(w or ()), pt.requested_types(q)) for q, w in RESIDUALS.items()
             if pt.requested_types(q) != w}
    assert not wrong, wrong


def test_requested_types_shape_and_case():
    for q, want in REQUESTED.items():
        got = pt.requested_types(q)
        assert isinstance(got, frozenset)
        assert pt.requested_types(q.upper()) == want, q
        assert pt.requested_types("  " + q.replace(" ", "   ") + "  ") == want, q
    assert pt.requested_types("What\u2019s good to wear to a wedding") == APPAREL


def test_watch_jewelry_asks_rank_their_own_rows_first():
    """2026-10-30 review: the row watch rule (bands and watches are accessory) without the
    ask side read "apple watch bracelet" as jewelry, so main.py ranked a PANDORA bracelet
    (rr 45) above the Apple Watch bracelets the reranker scored 92. The ask and its rows
    must agree: its watch bands type_rank 0, plain bracelets 2 -- and the reverse for a
    plain bracelet ask. Real catalog titles."""
    bands = ["Solomo Bracelet Compatible for Apple Watch 38MM 40MM,[Jewel Series] Luxury Bling",
             "TOROTOP Bracelet Compatible for Apple Watch Band 40mm/38mm Women Girl",
             "Leather Replacement Bracelet Strap Wristband for Apple Watch 38mm",
             "Kenneth Cole New York Mens Stainless Steel Bracelet Casual Watch"]
    plain = ["PANDORA Heart Clasp Bracelet 6.7 inch Bundle",
             "Carolyn Pollack Sterling Silver Multi Gemstone Bracelet"]
    for ask in ("apple watch bracelet", "watch bracelet", "bracelet for apple watch",
                "women's watch bracelet"):
        req = pt.requested_types(ask)
        assert [pt.type_rank(pt.classify_title(t)[0], req) for t in bands] == [0] * 4, ask
        assert [pt.type_rank(pt.classify_title(t)[0], req) for t in plain] == [2, 2], ask
    req = pt.requested_types("silver bracelet")
    assert [pt.type_rank(pt.classify_title(t)[0], req) for t in plain] == [0, 0]
    necklace = ("DSD Seahorse Pocket Watch Floating Glass Lockets Necklace Style Retro Vintage "
                "Pocket Watch Necklace")
    assert pt.type_rank(pt.classify_title(necklace)[0],
                        pt.requested_types("pocket watch necklace")) == 0


def test_type_rank():
    req = frozenset({"apparel"})
    assert pt.type_rank("footwear", None) == 0
    assert pt.type_rank(None, None) == 0
    assert pt.type_rank("apparel", req) == 0
    assert pt.type_rank("unknown", req) == 1
    for other in ("footwear", "jewelry", "bag", "headwear", "accessory", None, float("nan"), ""):
        assert pt.type_rank(other, req) == 2, other
    assert pt.type_rank("footwear", frozenset({"apparel", "footwear"})) == 0
    assert pt.type_rank("jewelry", frozenset()) == 0  # an empty request is no request


def test_slot_type():
    assert pt.slot_type("beach wedding sandals") == "footwear"
    assert pt.slot_type("linen midi dress") == "apparel"
    assert pt.slot_type("gold drop earrings") == "jewelry"
    assert pt.slot_type("straw clutch") == "bag"
    assert pt.slot_type("wide brim sun hat") == "headwear"
    assert pt.slot_type("ring light") == "jewelry"   # a collision main.py's keep handles
    assert pt.slot_type("warm desk lamp") == "unknown"
    # a dress belt slot is an accessory (it was apparel, so it sorted before the shoes)
    assert pt.slot_type("men's brown leather dress belt") == "accessory"
    assert pt.slot_type("kids water shoes aqua socks") == "footwear"
    # 'accessories' sorts as an accessory and accepts the whole family
    assert pt.slot_type("delicate gold accessories") == "accessory"
    assert pt.slot_type("hair accessories") == "headwear"


def test_slot_accepts():
    assert pt.slot_accepts("delicate gold accessories") == pt.ACCESSORY_FAMILY
    assert pt.slot_accepts("beach wedding dress accessories") == pt.ACCESSORY_FAMILY
    assert pt.slot_accepts("hair accessories") == frozenset({"headwear"})
    assert pt.slot_accepts("flat leather sandals") == frozenset({"footwear"})
    assert pt.slot_accepts("warm desk lamp") == frozenset()
    assert pt.slot_accepts(None) == frozenset()
    assert pt.ACCESSORY_FAMILY < set(pt.FASHION_TYPES)
    # the family word is read in asks and slots only: titles keep their own rules
    assert pt.classify_title("Hair Accessories for Women") == ("headwear", "hair accessory")
    assert pt.classify_title("Fashion Accessories Gift Box")[0] == "unknown"


# ------------------------------------------------------------------ annotate
def _frame(index=None):
    """A Databricks-shaped frame: Title-case vertical, original-case titles, gender None."""
    rows = [
        ("Fashion", "Women's Clothing", "Zeagoo Womens Little Cocktail Party Dress", "women"),
        ("Fashion", "Women's Clothing", "Gold Drop Earrings for Wedding Dress", "women"),
        ("Fashion", "Men's Shoes", "Clarks Men's Casual Pitney Limit Oxford", "men"),
        ("Fashion", "Women's Clothing", "Ridlee Black / 8.5", "women"),
        ("Fashion", "Women's Clothing", None, "women"),
        ("Electronics", "Headphones", "Wireless Earbuds with Charging Case", None),
        ("Grocery", "Tea", "Organic Green Tea Bags, 100 Count", None),
        ("Electronics", "Camera", "Camera Bag for DSLR", None),
        (None, None, "Mystery Dress", None),
    ]
    df = pd.DataFrame(rows, columns=["bsns_vrtcl_name", "categ_lvl2_name", "Product_title", "gender"])
    for col in ("prod_description", "color", "material", "occasion"):
        df[col] = col
    if index is not None:
        df.index = index
    return df


EXPECTED = ["apparel", "jewelry", "footwear", "unknown", "unknown", None, None, None, None]
# What app/catalog_taxonomy.py adds to the same rows when app/search.py loads them (the
# frame has no store / dietary / specs: brand None, tags ())
TAXONOMY_EXPECTED = {
    "item_category": [None] * 5 + ["headphones", "tea", "bags", None],
    "item_role": [None] * 5 + ["device", "food", "accessory", None],
    "watch_kind": [None] * 9,
    "diet_tags": [()] * 9,
    "works_with": [()] * 9,
    "item_condition": [None] * 9,
    "brand": [None] * 9,
}
# Every column annotate() adds, in the order it adds them
_ADDED = ["product_type", "product_subtype", "fine_subtype", "shopper_gender", "audience",
          "color_family"]


def test_annotate_types_rows_and_leaves_others_none():
    df = pt.annotate(_frame())
    assert df[pt.TYPE_COLUMN].tolist() == EXPECTED
    assert df[pt.SUBTYPE_COLUMN].tolist()[5:] == [None] * 4
    assert df[pt.TYPE_COLUMN].dtype == object and df[pt.SUBTYPE_COLUMN].dtype == object
    for v in df[pt.TYPE_COLUMN].tolist() + df[pt.SUBTYPE_COLUMN].tolist():
        assert v is None or isinstance(v, str), repr(v)
    # row dicts read None, never NaN (main.py and the API see these)
    assert df.iloc[6].to_dict()[pt.TYPE_COLUMN] is None


def _lowercased(df):
    """The frame as the local backend served it until 2026-10-15 (_build_search_text
    lowercased the text columns IN PLACE, missing values ''); since then it lowercases
    copies, but a lowercase catalog must still type the same."""
    for col in ("bsns_vrtcl_name", "categ_lvl2_name", "Product_title"):
        df[col] = df[col].fillna("").astype(str).str.lower()
    return df


def test_annotate_after_local_lowercasing_is_identical():
    """annotate is case-insensitive, so a lowercase frame (the old local shape) cannot
    silently switch the feature off; a literal == "Fashion" test would have typed
    nothing on it."""
    title_case = pt.annotate(_frame())
    local = _lowercased(_frame())
    assert set(local["bsns_vrtcl_name"]) == {"fashion", "electronics", "grocery", ""}
    assert (local["bsns_vrtcl_name"] == "Fashion").sum() == 0  # the trap
    pt.annotate(local)
    for column in _ADDED:
        assert local[column].tolist() == title_case[column].tolist(), column
    assert pt.type_counts(local) == pt.type_counts(title_case)
    assert pt.annotation_counts(local) == pt.annotation_counts(title_case)


def test_annotate_never_moves_a_row():
    index = pd.Index([40, 3, 17, 9, 0, 21, 5, 8, 1])
    df = _frame(index=index)
    before = df.copy()
    out = pt.annotate(df)
    assert out is df
    assert len(df) == len(before) and df.index.equals(index)
    assert df[pt.TYPE_COLUMN].tolist() == EXPECTED  # aligned by position, not by label
    pd.testing.assert_frame_equal(df[before.columns], before)  # nothing else touched
    again = pt.annotate(df.copy())
    pd.testing.assert_frame_equal(again, df)  # idempotent


def test_annotate_degenerate_frames():
    assert _ADDED == [pt.TYPE_COLUMN, pt.SUBTYPE_COLUMN, pt.FINE_SUBTYPE_COLUMN,
                      pt.GENDER_COLUMN, pt.AUDIENCE_COLUMN, pt.COLOR_FAMILY_COLUMN]
    empty = pt.annotate(pd.DataFrame(columns=["bsns_vrtcl_name", "Product_title"]))
    assert list(empty.columns)[-6:] == _ADDED and len(empty) == 0
    no_vertical = pt.annotate(pd.DataFrame({"Product_title": ["Red Dress"]}))
    for column in _ADDED:
        assert no_vertical[column].tolist() == [None], column
    no_title = pt.annotate(pd.DataFrame({"bsns_vrtcl_name": ["Fashion"]}))
    assert no_title[pt.TYPE_COLUMN].tolist() == [None]
    assert no_title[pt.GENDER_COLUMN].tolist() == [None]
    # colour is read on any frame with a colour column, Fashion or not
    colour_only = pt.annotate(pd.DataFrame({"color": ["Navy Blue", None, "as shown"]}))
    assert colour_only[pt.COLOR_FAMILY_COLUMN].tolist() == ["navy", None, None]
    assert colour_only[pt.TYPE_COLUMN].tolist() == [None] * 3


def test_annotate_reads_pandas_string_dtype():
    """read_csv gives pandas' str dtype (NaN for missing); toPandas gives object/None."""
    df = _frame()
    df = df.astype({"bsns_vrtcl_name": "str", "Product_title": "str"})
    assert pt.annotate(df)[pt.TYPE_COLUMN].tolist() == EXPECTED


def _c1_frame():
    """Fashion rows with real tags and colours, plus other verticals with colours."""
    rows = [  # vertical, category, title, gender tag, colour
        ("Fashion", "Women's Clothing", "Women's High Waist Yoga Leggings", "women", "Black"),
        ("Fashion", "Unisex", "Ballet Flats Pointed Toe", "unisex", "Rose Gold"),
        ("Fashion", "Women's Clothing", "Girl's Sleeveless Skater Dress 12 Years", "women",
         "Navy Blue"),
        ("Fashion", "Kids", "Baby Girl Romper 6-12 Months", "kids", None),
        ("Fashion", "Kids", "OutTop Womens Winter Coats for Women Long Puffer Jacket", "kids",
         "Wine"),
        ("Fashion", "Men's Clothing", "Brooks Men's Adrenaline GTS 21", None, "grey"),
        ("Fashion", "Unisex", "Skull Face Mask for Women Men Dust Wind UV", "unisex", "multicolor"),
        ("Electronics", "Cables", "USB-C Cable 6ft", None, "Black1"),
        ("Grocery", "Tea", "Organic Green Tea Bags", None, float("nan")),
    ]
    return pd.DataFrame(rows, columns=["bsns_vrtcl_name", "categ_lvl2_name", "Product_title",
                                       "gender", "color"])


C1_EXPECTED = {
    "fine_subtype": ["leggings", "flats", "dresses", "jumpsuits_sets", "coats_jackets",
                     "athletic", None, None, None],
    "shopper_gender": ["women", "women", "kids", "kids", "women", "men", "unisex", None, None],
    "audience": ["adult", "adult", "kids", "baby", "adult", "adult", "adult", None, None],
    # on every vertical
    "color_family": ["black", "gold", "navy", None, "burgundy", "grey", "multicolor", "black",
                     None],
}


def test_annotate_adds_the_c1_columns():
    """Contract C1: fine_subtype / shopper_gender / audience are Fashion-only (None
    elsewhere), color_family is read on every vertical; all object dtype, str or None."""
    df = pt.annotate(_c1_frame())
    assert df[pt.TYPE_COLUMN].tolist()[:7] == ["apparel", "footwear", "apparel", "apparel",
                                               "apparel", "footwear", "accessory"]
    for column, want in C1_EXPECTED.items():
        assert df[column].tolist() == want, (column, df[column].tolist())
    for column in _ADDED:
        assert df[column].dtype == object, column
        assert all(v is None or isinstance(v, str) for v in df[column].tolist()), column
    assert df.iloc[7].to_dict()[pt.GENDER_COLUMN] is None          # never NaN in a row dict
    assert set(df[pt.GENDER_COLUMN].dropna()) <= set(pt.SHOPPER_GENDERS)
    assert set(df[pt.AUDIENCE_COLUMN].dropna()) <= set(pt.AUDIENCES)
    assert set(df[pt.FINE_SUBTYPE_COLUMN].dropna()) <= set(pt.FINE_SUBTYPE_LABELS)
    assert set(df[pt.COLOR_FAMILY_COLUMN].dropna()) <= set(pt.COLOR_FAMILIES)
    counts = pt.annotation_counts(df)
    assert counts[pt.GENDER_COLUMN] == {"women": 3, "kids": 2, "men": 1, "unisex": 1}
    assert counts[pt.COLOR_FAMILY_COLUMN]["none"] == 2     # every vertical counted
    assert sum(counts[pt.FINE_SUBTYPE_COLUMN].values()) == 7   # Fashion rows only


def test_annotate_c1_columns_idempotent_and_in_place():
    index = pd.Index([90, 4, 33, 7, 1, 50, 2, 8, 0])
    df = _c1_frame()
    df.index = index
    before = df.copy()
    out = pt.annotate(df)
    assert out is df and df.index.equals(index)
    pd.testing.assert_frame_equal(df[before.columns], before)       # no source column touched
    for column, want in C1_EXPECTED.items():
        assert df[column].tolist() == want, column                  # by position, not label
    pd.testing.assert_frame_equal(pt.annotate(df.copy()), df)       # idempotent
    # the category bucket stands in for a missing tag, and pandas' str dtype reads the same
    no_tags = _c1_frame().drop(columns=["gender"])
    pt.annotate(no_tags)
    assert no_tags[pt.GENDER_COLUMN].tolist()[:2] == ["women", "women"]
    as_str = _c1_frame().astype({"Product_title": "str", "gender": "str", "color": "str"})
    pt.annotate(as_str)
    assert as_str[pt.GENDER_COLUMN].tolist() == C1_EXPECTED["shopper_gender"]


# ------------------------------------------------------------------ fine subtype
FINE = {
    # order settles keyword-stuffed titles
    "Lehenga Choli for Women Wedding": "ethnic_wear",
    "Men's Cotton Kurta Pajama Set": "ethnic_wear",
    "Women's Flannel Pajama Pants": "sleep_lounge",
    "Women's Two Piece Lounge Set": "sleep_lounge",
    "Carters Baby Boys Fleece Snap Up Bear Print Sleep & Play 3 Months": "sleep_lounge",
    "Women's Bikini Panties 6 Pack": "lingerie",
    "Women's Triangle Bikini Swimsuit": "swim",
    "Women's Chiffon Beach Cover Up": "swim",
    "Men's Swim Trunks": "swim",
    "Women's Cable Knit Sweater Dress": "dresses",
    "Women's Denim Shirt Dress": "dresses",
    "Men's 3 Piece Slim Fit Suit": "suits_blazers",
    "Men's Slim Fit Blazer": "suits_blazers",
    "Women's Casual Romper": "jumpsuits_sets",
    "Gihuo Women's Winter Sherpa Lined Pullover Hooded Sweatshirt Hoodie (Grey, Medium)": "hoodies",
    "Women's Quilted Puffer Jacket": "coats_jackets",
    "Women's Plaid Shacket": "coats_jackets",
    "Under Armour Mens UA Gamer Fleece (XL, Midnight Navy)": "coats_jackets",
    "Women's Open Front Cardigan": "sweaters",
    "Women's Denim Jean Shorts": "shorts",
    "Nike Women's Training Short Black/Black Heather Size Large": "shorts",
    "Women's Denim Midi Skirt": "skirts",
    "Men's Slim Fit Jeans": "jeans",
    "Levi's Women's Straight Leg Jean": "jeans",
    "Women's High Waist Yoga Leggings": "leggings",
    "Fleece Lined Leggings": "leggings",          # 'fleece' describes the leggings
    "Men's Jogger Sweatpants": "pants",
    "Men's Chino Pants": "pants",
    "Women's Ribbed Tank Top": "tops_tees",
    "Men's Graphic T-Shirt": "tops_tees",
    "Men's Button Down Oxford Shirt": "blouses_shirts",
    "Women's Chiffon Blouse": "blouses_shirts",
    "Women's Wireless Bra": "lingerie",
    "Men's Boxer Briefs": "lingerie",
    "Men's Crew Socks 6 Pack": "socks_hosiery",
    "Women's Sheer Tights": "socks_hosiery",
    "Women's Nightgown": "sleep_lounge",
    # a 'suit' that is no tailoring (2026-10-19 review: ~90 sets read Suits & blazers)
    "COOFANDY Mens Hooded Jogging Suits Sets Tracksuits 2 Piece Full Zip Hooded Sweatsuits "
    "Sports Suits": "jumpsuits_sets",
    "Gold's Gym Adult Performance Sauna Suit, L/XL": "jumpsuits_sets",
    "Youth Pullover Hoodie Sweatpants Suit Hooded Tracksuit Sweatshirt Set": "jumpsuits_sets",
    "Womens 2 Pieces Romper Suits Fringe Trim Zipper Sweatshirt": "jumpsuits_sets",
    "Toddler Girl Warm Winter Ski Suit Bib Jacket Coat 5T": "jumpsuits_sets",
    "Lemorecn Women Wetsuits Neoprene 3/2mm Full Body Diving Suit": None,
    "Men's Slim Fit Athletic Sport Coat": "suits_blazers",   # the veto guards 'suit' only
    "Men's Suit Vest Slim Fit": "suits_blazers",
    "Women's Pant Suit 2 Piece Business": "suits_blazers",
    # footwear
    "Brooks Women's Ghost 14 Neutral Running Shoe": "athletic",
    "ASICS Women's Gel-Kayano 28": "athletic",
    "Converse Chuck Taylor All Star Low Top Sneaker": "sneakers",
    "Women's Ankle Boots": "boots",
    "UGG Women's Classic Short II Boot, Chestnut, 7 B (M) US": "boots",
    "Women's Flip Flops": "sandals",
    "Women's Wedge Sandals": "sandals",
    "Women's Stiletto Heel Pump": "heels",
    "Women's Pointed Toe Slingback Kitten Heel": "heels",
    "Women's Ballet Flat": "flats",
    "Men's Penny Loafer": "flats",
    "Men's Leather Oxford Shoes": "flats",
    "Girls Patent Mary Jane School Shoes": "flats",
    "Men's Memory Foam Slippers": "slippers",
    # 2026-10-19 review: a word that names a garment alone only describes the tee, tank
    # or top it sits next to; a sleeve length or what a top is worn with is no garment
    "Women's Cotton T-Shirts Short Sleeve Loose Comfy Basic Plain Tunic Tee": "tops_tees",
    "Fall/Winter Women's Fashion T-Shirt V-Neck Pullover Print Long Sleeve": "tops_tees",
    "Sunward Summer Casual Tank Tops Vest for Women": "tops_tees",
    "Women's Swing Vest Strap Cami Camisole Tank Top": "tops_tees",
    "Women's Halter Lace Up Back Vest Crop Top": "tops_tees",
    "Women Plus Size One Shoulder Short/Long Sleeve Casual T-Shirt Tops": "tops_tees",
    "Women's Short & Long Sleeves V-Neck Maternity Shirt": "tops_tees",
    "Women's Short Puff Sleeve Raglan T-Shirt Summer Tops": "tops_tees",
    "Women's Long Sleeve Tunic Tops for Leggings": "blouses_shirts",
    "Short Sleeve Tunics for Women to Wear with Leggings Flowy Shirt": "blouses_shirts",
    "Womens V Neck Linen Cotton Tops Summer Casual Long Sleeve Shirt Beach Pullover Tunic Blouse":
        "blouses_shirts",
    "Youtobin Women's New Winter Dress-Coats Slim Long Woolen Coat XL Blue": "coats_jackets",
    # ...but not when the rule names the garment too, or no top is named
    "Women's Quilted Puffer Vest with Tee": "coats_jackets",
    "Women's Open Front Plaid Vest Cowl Neck Tops": "coats_jackets",
    "Men's Cable Knit Pullover Sweater": "sweaters",
    "Women's Fleece Pullover Top": "sweaters",
    "Women's Tunic Tops": "blouses_shirts",
    "Women's Lace Tunic Blouse": "blouses_shirts",
    "Women's High Waist Leggings for Yoga": "leggings",
    "Women's Short Denim Overalls": "jumpsuits_sets",
    # no shop category: another type, a costume, a model-only shoe
    "Gold Hoop Earrings": None,
    "Women's Leather Tote Handbag": None,
    "JYZCOS Mens Toga Costume for Adult Halloween Cosplay Party Roman Fancy Dress": None,
    "NINE WEST Rainford Pink 9 M": None,
}


def test_fine_subtype_table():
    wrong = {t: (want, pt.fine_subtype(t)) for t, want in FINE.items()
             if pt.fine_subtype(t) != want}
    assert not wrong, wrong
    for title in FINE:   # case-insensitive, like the type
        assert pt.fine_subtype(title.lower()) == pt.fine_subtype(title.upper()) == FINE[title]


def test_fine_subtype_labels_are_the_contract():
    assert set(pt.APPAREL_SUBTYPES) == {
        "dresses", "tops_tees", "blouses_shirts", "sweaters", "hoodies", "coats_jackets",
        "suits_blazers", "jeans", "pants", "shorts", "leggings", "skirts", "jumpsuits_sets",
        "swim", "sleep_lounge", "lingerie", "socks_hosiery", "ethnic_wear"}
    assert set(pt.FOOTWEAR_SUBTYPES) == {
        "sneakers", "boots", "sandals", "heels", "flats", "slippers", "athletic"}
    assert set(pt.FINE_SUBTYPE_LABELS) == pt.APPAREL_SUBTYPES | pt.FOOTWEAR_SUBTYPES
    assert not pt.APPAREL_SUBTYPES & pt.FOOTWEAR_SUBTYPES
    labels = list(pt.FINE_SUBTYPE_LABELS.values())
    assert all(isinstance(v, str) and v and v[0].isupper() for v in labels)
    assert len(set(labels)) == len(labels)
    # every rule maps to a contract key of its own type
    assert {k for k, _r in pt._FINE_APPAREL} <= pt.APPAREL_SUBTYPES
    assert {k for k, _r in pt._FINE_FOOTWEAR} <= pt.FOOTWEAR_SUBTYPES


def test_fine_subtype_follows_the_type():
    for title, _want in TITLES:
        type_ = pt.classify_title(title)[0]
        fine = pt.fine_subtype(title, type_)
        if type_ == "apparel":
            assert fine is None or fine in pt.APPAREL_SUBTYPES, (title, fine)
        elif type_ == "footwear":
            assert fine is None or fine in pt.FOOTWEAR_SUBTYPES, (title, fine)
        else:
            assert fine is None, (title, fine)
    # the type passed in wins: a footwear row never gets a garment key
    assert pt.fine_subtype("Sock Sneakers", "footwear") == "sneakers"
    assert pt.fine_subtype("Women's Ankle Boots", "jewelry") is None
    for blank in (None, float("nan"), pd.NA, ""):
        assert pt.fine_subtype(blank) is None


# ------------------------------------------------------------------ colour family
COLORS = {
    "Black": "black", " BLACK ": "black", "Black1": "black", "Jet Black": "black",
    "Navy Blue": "navy", "navy": "navy",
    "Rose Gold": "gold", "rosegold": "gold",          # a modifier before another colour
    "Steel Blue": "blue", "Royal Blue": "blue", "Teal": "blue", "Denim": "blue",
    "Black/White": "black", "White-Black": "white",   # the first family named
    "Multicolor": "multicolor", "Multi-colored": "multicolor", "Assorted": "multicolor",
    "Floral": "multicolor", "Clear": "clear", "Transparent": "clear",
    "Khaki": "beige", "Tan": "beige", "Ivory": "beige", "Champagne": "beige",
    "Off White": "white", "Charcoal": "grey", "Heather Grey": "grey", "Gray": "grey",
    "Silver": "silver", "sliver": "silver", "Gold": "gold",
    "Burgundy": "burgundy", "Wine": "burgundy", "Wine Red": "red", "Maroon": "burgundy",
    "Olive": "green", "Mint Green": "green", "Coral": "orange", "Rust": "orange",
    "Fuchsia": "pink", "Hot Pink": "pink", "Rose": "pink", "Lavender": "purple",
    "Mustard": "yellow", "Chocolate": "brown", "Purple": "purple",
    # no family: a size, a pattern-less placeholder, a missing value
    "X-Large": None, "As Shown": None, "Light Wash": None, "": None, "   ": None,
}


def test_color_family_table():
    wrong = {c: (want, pt.color_family(c)) for c, want in COLORS.items()
             if pt.color_family(c) != want}
    assert not wrong, wrong
    for odd in (None, float("nan"), pd.NA, 12, ["black"]):
        assert pt.color_family(odd) is None, odd


def test_color_families_are_the_swatch_map():
    assert set(pt.COLOR_FAMILIES) == {
        "black", "blue", "white", "red", "grey", "silver", "pink", "green", "gold", "brown",
        "beige", "purple", "navy", "yellow", "orange", "burgundy", "multicolor", "clear"}
    assert all(re.fullmatch(r"#[0-9A-F]{6}", h) for h in pt.COLOR_FAMILIES.values())
    assert pt.COLOR_FAMILIES["black"] == "#111111" and pt.COLOR_FAMILIES["navy"] == "#1B2A4A"
    assert pt.SPECIAL_SWATCHES == {"multicolor", "clear"}
    assert set(pt.COLOR_LABELS) == set(pt.COLOR_FAMILIES)
    assert set(pt._COLOR_WORDS.values()) <= set(pt.COLOR_FAMILIES)
    assert pt._COLOR_MODIFIERS <= set(pt._COLOR_WORDS)


# ------------------------------------------------------------------ shopper gender, audience
GENDERS = [  # (title, stored tag, category) -> (shopper_gender, audience)
    # keyword stuffing ("for women girls", "teen girls") keeps an adult's tag
    ("Gold Hoop Earrings for Women Girls", "women", None, ("women", "adult")),
    ("Stud Earrings for Women Teen Girls Hypoallergenic", "women", None, ("women", "adult")),
    # a children's item filed under an adult department
    ("Girl's Sleeveless Skater Dress 12 Years", "women", None, ("kids", "kids")),
    ("Boys Graphic T-Shirt 8-10 Years", "men", None, ("kids", "kids")),
    ("Kid's Converse Chuck Taylor All Star", "men", None, ("kids", "kids")),
    # an adult garment filed under Kids
    ("OutTop Womens Winter Coats for Women Long Puffer Jacket", "kids", None, ("women", "adult")),
    # kids rows: toddler sizes are kids, months and newborns are babies
    ("Toddler Girls Floral Sundress 3T", "kids", None, ("kids", "kids")),
    ("Baby Girl Romper 6-12 Months", "kids", None, ("kids", "baby")),
    ("Newborn Baby Hat 0-3 Months", "kids", None, ("kids", "baby")),
    ("Baby Boy Sneakers Toddler Size 5", "kids", None, ("kids", "kids")),
    # plural possessives end on the apostrophe (2026-10-19 review: they never matched)
    ("Girls' Short Sleeve Ruffle Dress", "women", None, ("kids", "kids")),
    ("Fruit of the Loom Boys' Woven Boxer", "men", None, ("kids", "kids")),
    ("Boys' Running Shoes Lightweight Sneakers", "men", None, ("kids", "kids")),
    ("Stud Earrings for Women Girls'", "women", None, ("women", "adult")),
    # "baby girl's / boy's" is a baby's size, like "baby girls" (68 rows read kids)
    ("Baby Boy's Romper 6-12 Months", "kids", None, ("kids", "baby")),
    ("Duck Duck Goose Baby Girl's Short Sleeve Bodysuits 6-9 Months", "kids", None,
     ("kids", "baby")),
    ("Baby Girls' Romper 6-12 Months", "kids", None, ("kids", "baby")),
    ("Toddler Boys' Sneakers 3T", "kids", None, ("kids", "kids")),
    # the "unisex" fallback re-read from the title
    ("Juniors Ripped Jeans", "unisex", None, ("women", "adult")),
    ("Ballet Flats Pointed Toe", "unisex", None, ("women", "adult")),       # by convention
    ("Tennis Bracelet Cubic Zirconia", "unisex", None, ("women", "adult")),
    ("Silk Bow Tie Pre-Tied", "unisex", None, ("men", "adult")),
    ("necklace for women18 inch chain", "unisex", None, ("women", "adult")),
    ("Man Leather Wallet Bifold", "unisex", None, ("men", "adult")),
    ("Kids Rain Boots", "unisex", None, ("kids", "kids")),
    ("Women's and Men's Classic Clog", "unisex", None, ("unisex", "adult")),
    ("Leather Belt for Men and Women", "unisex", None, ("unisex", "adult")),
    ("Unisex Adult Crew Socks", "unisex", None, ("unisex", "adult")),
    ("Classic Aviator Sunglasses", "unisex", None, ("unisex", "adult")),
    # names and compounds are no gender word
    ("Spider-Man Graphic T-Shirt", "unisex", None, ("unisex", "adult")),
    ("Wonder Woman Logo T-Shirt", "unisex", None, ("unisex", "adult")),
    ("Snowman Christmas Sweater", "unisex", None, ("unisex", "adult")),
    ("Boyfriend Jeans Ripped", "unisex", None, ("unisex", "adult")),
    ("Baby Blue Silk Scarf", "unisex", None, ("unisex", "adult")),
    # 2026-10-19 review: a couple's item filed under Kids is an adult's, a weak kid word
    # notwithstanding; a strong kid cue or no pair cue keeps it kids
    ("Cute His and Hers Matching Couple T-Shirts - I Love This Guy and Girl (Left- XL/Right- M)"
     " Black", "kids", "Kids", ("unisex", "adult")),
    ("Matching Couples Hoodies Girl and Boy Valentines", "kids", "Kids", ("unisex", "adult")),
    ("Toddler Girl Couple Shirt", "kids", "Kids", ("kids", "kids")),
    ("Tuff Guys Boys Sleepwear Pajama Pant Set Size 7", "kids", "Kids", ("kids", "kids")),
    ("Girls Floral Party Dress", "kids", "Kids", ("kids", "kids")),
    # no tag: the category bucket, else the title
    ("Fleece Lined Leggings", None, "Women's Clothing", ("women", "adult")),
    ("Fleece Lined Leggings", None, "Men's Clothing", ("men", "adult")),
    ("Fleece Lined Leggings", None, "Girls", ("kids", "kids")),
    ("Men's Cargo Shorts", None, None, ("men", "adult")),
    ("Fleece Lined Leggings", None, None, ("unisex", "adult")),
    ("Fleece Lined Leggings", float("nan"), float("nan"), ("unisex", "adult")),
]


def test_shopper_gender_and_audience_table():
    wrong = [(t, g, c, want, pt.shopper_profile(t, g, c)) for t, g, c, want in GENDERS
             if pt.shopper_profile(t, g, c) != want]
    assert not wrong, wrong
    for title, tag, category, want in GENDERS:   # case-insensitive, like the type
        assert pt.shopper_profile(title.upper(), tag, category) == want, title
    assert pt.shopper_profile(None) == ("unisex", "adult")
    assert pt.shopper_profile("Kids Tee", "KIDS ") == ("kids", "kids")   # tag case and spaces


def test_annotate_matches_shopper_profile():
    titles = [t for t, _g, _c, _w in GENDERS]
    df = pd.DataFrame({"bsns_vrtcl_name": "Fashion", "Product_title": titles,
                       "gender": [g for _t, g, _c, _w in GENDERS],
                       "categ_lvl2_name": [c for _t, _g, c, _w in GENDERS]})
    pt.annotate(df)
    got = list(zip(df[pt.GENDER_COLUMN], df[pt.AUDIENCE_COLUMN]))
    assert got == [w for _t, _g, _c, w in GENDERS]


# ------------------------------------------------------------------ fit, style, audience asks
FITS = {
    "petite work pants": {"petite"},
    "petite women's black pants": {"petite"},
    "maternity dress": {"maternity"},
    "nursing tops": {"maternity"},
    "postpartum leggings": {"maternity"},
    "plus size jeans": {"plus"},
    "women's plus size dresses": {"plus"},
    "curvy jeans": {"plus"},
    "tall jeans for men": {"tall"},
    "big and tall polo shirts": {"tall"},
    "petite plus size dress": {"petite", "plus"},
    # collisions and non-fits: a phone model, a bin bag, a shaft height, small jewellery
    "iphone 14 plus": None, "tall kitchen trash bags": None, "tall boots": None,
    "knee tall socks": None, "petite earrings": None, "dress plus sandals": None,
    "summer dress": None, "": None, None: None,
}


def test_requested_fit_table():
    wrong = {q: (want, pt.requested_fit(q)) for q, want in FITS.items()
             if pt.requested_fit(q) != (frozenset(want) if want else None)}
    assert not wrong, wrong
    assert pt.requested_fit("PETITE   Work Pants") == frozenset({"petite"})
    assert all(f in pt.FITS for want in FITS.values() if want for f in want)


IMPLIES_WOMEN = {
    "petite work pants": True, "maternity dress": True, "petite women's black pants": True,
    "maternity jeans for men": False,     # a men's cue cancels it
    "petite suit for my husband": False,
    "plus size jeans": False,             # plus and tall are worn by everyone
    "tall jeans": False, "work pants": False, "": False, None: False,
}


def test_implies_women_table():
    wrong = {q: (want, pt.implies_women(q)) for q, want in IMPLIES_WOMEN.items()
             if pt.implies_women(q) is not want}
    assert not wrong, wrong


_FORMAL = {"suits_blazers", "blouses_shirts", "pants", "dresses", "jumpsuits_sets", "skirts"}
_MEN_FORMAL = {"suits_blazers", "blouses_shirts", "pants"}
SUBTYPE_ASKS = {
    "linen dress": {"dresses"},
    "sweater dress": {"dresses"},              # a noun right before another is a modifier
    "jean jacket": {"coats_jackets"},
    "boot socks": {"socks_hosiery"},
    "jeans not leggings": {"jeans"},           # a noun after a negation is excluded
    "jeans, not leggings": {"jeans"},
    "leggings instead of jeans": {"leggings"},
    "dress without sleeves": {"dresses"},
    "no show socks": {"socks_hosiery"},        # ...but a bare 'no' is a style
    "yoga pants": {"leggings", "pants"},
    "tank top": {"tops_tees"},
    "shirt": {"tops_tees", "blouses_shirts"},
    "button down shirt": {"blouses_shirts"},
    "running shoes": {"athletic"},
    "white sneakers": {"sneakers", "athletic"},
    "pajama pants": {"sleep_lounge"},
    "bikini top": {"swim"},
    "sports bra": {"lingerie"},
    "summer dresses and sandals": {"dresses", "sandals"},
    "lehenga for my wedding": {"ethnic_wear"},
    "petite work pants": {"pants"},
    # two-word garments read as their titles do (2026-10-19 review: the bare noun decided)
    "swim suit": {"swim"},
    "swimsuit for a beach wedding": {"swim"},
    "cover up for a beach wedding": {"swim"},
    "track suit for men": {"jumpsuits_sets"},
    "jogging suit": {"jumpsuits_sets"},
    "body suit": {"jumpsuits_sets"},
    "suit vest for men": {"suits_blazers"},
    "night gown": {"sleep_lounge"},
    "house dress": {"sleep_lounge"},
    "dressing gown": {"sleep_lounge"},
    "business suit": {"suits_blazers"},
    "wet suit": None, "rain suit": None, "space suit costume": None,
    # a black-tie / white-tie dress code names its garments (2026-10-19 review: the
    # jacket intent's windbreakers and hi-vis jackets held a third of the c11 window);
    # a named garment still wins, and a non-clothing ask or a bare tie names none
    "black-tie event": _FORMAL, "white tie gala": _FORMAL,
    "black tie optional wedding": _FORMAL, "what to wear to a black tie": _FORMAL,
    "something for her to wear to a black-tie gala": _FORMAL,
    # a men's ask leaves out gowns, skirts and jumpsuits
    "black tie event men": _MEN_FORMAL, "black tie attire for men": _MEN_FORMAL,
    "what do men wear to black tie": _MEN_FORMAL, "white tie gala for my husband": _MEN_FORMAL,
    "black tie event dress": {"dresses"},
    "black tie event shoes": None, "black tie": None, "black cable ties": None,
    "earrings for a black tie gala": None,
    # 'dress' modifying the noun after it is no dress, as in titles (2026-10-30 R2: "dress
    # watch for men" read the dresses subtype); a garment after it keeps its own subtype
    "dress watch for men": None, "men's dress watch": None, "dress watches for women": None,
    "vintage style men's analog dress watch": None, "dress belt": None,
    "men's black leather dress belt": None, "dress gloves": None, "dress uniform": None,
    "dress pump": None,
    "dress shirt": {"blouses_shirts"}, "dress pants": {"pants"}, "dress suit": {"suits_blazers"},
    "dress pumps": {"heels"}, "dress socks": {"socks_hosiery"}, "dress boots": {"boots"},
    "dress coat": {"coats_jackets"}, "dress watch and a dress": {"dresses"},
    "dress and shoes": {"dresses"}, "black dress": {"dresses"}, "shirt dress": {"dresses"},
    # no shop category, or the noun is the context after the connector
    "dress shoes for men": None,
    "earrings for a wedding dress": None,
    "what to wear to a wedding": None,
    "how to dress for an interview": None,
    "beach wedding outfit": None,
    "clothes": None, "": None, None: None,
}


def test_dress_code_subtypes_are_formal_garments():
    assert pt._FORMAL_SUBTYPES == frozenset(_FORMAL) <= pt.APPAREL_SUBTYPES
    assert pt._MEN_FORMAL_SUBTYPES == frozenset(_MEN_FORMAL) < pt._FORMAL_SUBTYPES
    assert not {"coats_jackets", "hoodies", "tops_tees", "jeans", "shorts"} & pt._FORMAL_SUBTYPES


def test_requested_subtypes_table():
    wrong = {q: (want, pt.requested_subtypes(q)) for q, want in SUBTYPE_ASKS.items()
             if pt.requested_subtypes(q) != (frozenset(want) if want else None)}
    assert not wrong, wrong
    assert all(k in pt.FINE_SUBTYPE_LABELS for want in SUBTYPE_ASKS.values() if want for k in want)


# Every noun the title rule says 'dress' can modify (_DRESS_MODIFIED), singular and plural
DRESS_MODIFIED_NOUNS = (
    "shirt shirts pant pants slacks trouser trousers suit suits tuxedo tuxedos blazer blazers "
    "waistcoat waistcoats vest vests coat coats jacket jackets socks shoe shoes boot boots "
    "booties pump pumps heel heels sandal sandals flat flats loafer loafers oxford oxfords "
    "sneaker sneakers belt belts watch watches glove gloves uniform uniforms").split()


def test_dress_modifying_a_noun_is_no_dress_in_titles_or_asks():
    """One list for both readings (2026-10-30 R2): the title rule's lookahead is built from
    _DRESS_MODIFIED, and the ask reader drops a 'dress' that modifies one of its nouns, so
    "dress watch for men" no longer asks for dresses."""
    assert pt._DRESS_MODIFIES == r"(?![\s-]*" + pt._DRESS_MODIFIED + r")"
    noun_re = re.compile(pt._DRESS_MODIFIED)
    wrong = {}
    for noun in DRESS_MODIFIED_NOUNS:
        assert noun_re.fullmatch(noun), noun          # the table mirrors the rule
        for ask in (f"dress {noun}", f"men's dress {noun}", f"dress-{noun} for women"):
            if "dresses" in (pt.requested_subtypes(ask) or ()):
                wrong[ask] = pt.requested_subtypes(ask)
        title = f"Men's Classic Dress {noun.title()}"
        if pt.fine_subtype(title) == "dresses":
            wrong[title] = "dresses"
    assert not wrong, wrong
    for ask in ("dress", "black dress for a wedding", "dress for men", "dresses with pockets"):
        assert pt.requested_subtypes(ask) == frozenset({"dresses"}), ask


# Garment names whose query reading must agree with the title reading of the same words:
# main._ask_misses demotes every row whose fine_subtype is not asked for, so a misread
# ask ("swim suit" -> suits_blazers) seats the wrong style ahead of every right row.
TWO_WORD_GARMENTS = [
    "swim suit", "swimsuit", "bathing suit", "track suit", "tracksuit", "sweat suit",
    "snow suit", "ski suit", "body suit", "jump suit", "play suit", "wet suit", "rain suit",
    "jogging suit", "jogger suit", "sport suit", "sports suit", "running suit", "workout suit",
    "training suit", "sauna suit", "gym suit", "space suit", "flight suit", "boiler suit",
    "romper suit", "suit vest", "tuxedo vest", "suit jacket", "sport coat", "business suit",
    "pant suit", "night gown", "nightgown", "night dress", "house dress", "dressing gown",
    "sleep shirt", "swim dress", "evening gown", "ball gown", "sweater dress", "jean jacket",
    "denim shorts", "yoga pants", "dress pants", "tank top", "bikini top", "cover up",
]


def test_two_word_garments_read_like_their_titles():
    wrong = {}
    for phrase in TWO_WORD_GARMENTS:
        asked = pt.requested_subtypes(phrase)
        for who in ("Women's", "Men's"):
            title = pt.fine_subtype(f"{who} {phrase.title()}")
            ok = asked is None if title is None else (asked is not None and title in asked)
            if not ok:
                wrong[f"{who} {phrase}"] = (asked, title)
    assert not wrong, wrong


AUDIENCE_ASKS = {
    "toddler girl party dress": {"kids", "baby"},
    "teen girl jeans": {"adult", "kids"},
    "7 year old boy shoes": {"kids"},
    "6 month old baby clothes": {"baby"},
    "2 year old birthday outfit": {"baby", "kids"},
    "18 year old gift": {"adult"},
    "gift for my wife": {"adult"},
    "women's running shoes": {"adult"},
    "petite work pants": {"adult"},
    "mother daughter matching dresses": {"adult", "kids"},
    "boys hoodie": {"kids"},
    "kids rain boots": {"kids"},
    "newborn hat": {"baby"},
    "baby formula": {"baby"},             # a collision main.py's pool gate neutralises
    # a child's occasion, school grade or wedding-party role, and "N yo" (2026-10-19
    # review: these read None, so main.py's default adult audience buried every kids' row)
    "first communion dress": {"kids"}, "holy communion veil": {"kids"},
    "ring bearer outfit": {"kids"}, "flower girl dress": {"kids"},
    "junior bridesmaid dress": {"kids"},
    "christening gown": {"kids", "baby"}, "baptism outfit": {"kids", "baby"},
    "school uniform for my 8 yo": {"kids"}, "8yo boy shoes": {"kids"},
    "sneakers for my 9 y/o": {"kids"}, "my 15 y.o. son": {"adult", "kids"},
    "back to school outfits for 3rd grade": {"kids"}, "first grade picture day outfit": {"kids"},
    "kindergarten graduation dress": {"kids"}, "preschool rain boots": {"kids"},
    "middle school dance dress": {"kids"},
    "bridesmaid dress": {"adult"},
    "kindergarten teacher shirt": None,  # the teacher is the shopper
    "yo-yo": None, "school uniform": None,
    # no audience stated
    "wonder woman t-shirt": None, "baby shower gift": None, "linen dress": None,
    "girls night out dress": None, "spider-man hoodie": None, "": None, None: None,
}


def test_requested_audience_table():
    wrong = {q: (want, pt.requested_audience(q)) for q, want in AUDIENCE_ASKS.items()
             if pt.requested_audience(q) != (frozenset(want) if want else None)}
    assert not wrong, wrong
    assert all(a in pt.AUDIENCES for want in AUDIENCE_ASKS.values() if want for a in want)


GUEST_ASKS = {
    # plainly a guest: the cap on bridal, bridesmaid and swimwear applies
    "wedding guest dress": True,
    "cloths for a beach wedding in summer": True,
    "beach wedding outfit": True,
    "summer beach wedding outfit dress": True,
    "what to wear to a wedding": True,
    "what to wear to a gala": True,
    "something for her to wear to a black-tie gala": True,
    "semi-formal attire for a wedding": True,
    "dress for my sister's wedding": True,
    "outfit for a friend's wedding reception": True,
    "dress for a party": True,
    "mother of the bride dress for a beach wedding": True,   # a parent of the couple is a guest
    "wedding guest": True,                    # names a guest and no other product
    "wedding guest dress and shoes": True,
    # a guest who asks for swimwear is still a guest: main._guest_constraint then drops
    # the swimwear clause (2026-10-19 review), bridal wear still violates it
    "swimsuit for a beach wedding": True,
    "wedding guest swimsuit": True,
    "bikini to wear to a beach wedding": True,
    "beach cover up for a wedding guest": True,
    "bathing suit to wear to a party": True,
    # the wedding party's own clothes: never capped (the 2026-10-15 narrowing)
    "wedding costume": False,
    "wedding dress": False,
    "bride costume": False,
    "lehenga for my wedding": False,
    "bridesmaid dress for a beach wedding": False,
    "groomsmen suits": False,
    "Wedding dresses for a destination wedding in rajasthan": False,
    # no clothing asked for, or a party where swimwear / costumes fit
    "earrings for a wedding": False,
    # ...also when a guest is named (2026-10-19 review: these were guests)
    "earrings for a wedding guest": False,
    "wedding guest clutch": False,
    "wedding guest shoes": False,
    "jewelry for a wedding guest": False,
    "sandals for a beach wedding": False,
    "pool party outfit": False,
    "costume party outfit": False,
    "how to dress for an interview": False,
    "mother of the bride dress": False,       # no wedding named: no reading
    "": False, None: False,
}


def test_is_wedding_guest_ask_table():
    wrong = {q: (want, pt.is_wedding_guest_ask(q)) for q, want in GUEST_ASKS.items()
             if pt.is_wedding_guest_ask(q) is not want}
    assert not wrong, wrong
    assert pt.is_wedding_guest_ask("WEDDING   GUEST DRESS") is True


def test_readers_on_the_stakeholder_and_routing_queries():
    """The readers on queries the type tables already pin: none raises, each returns
    its declared shape."""
    queries = list(REQUESTED) + [q for q in NONE if q] + list(COLLISIONS) + list(RESIDUALS)
    for q in queries:
        for reader in (pt.requested_fit, pt.requested_subtypes, pt.requested_audience):
            got = reader(q)
            assert got is None or (isinstance(got, frozenset) and got), (reader.__name__, q)
        assert isinstance(pt.implies_women(q), bool)
        assert isinstance(pt.is_wedding_guest_ask(q), bool)


# ------------------------------------------------------------------ the search hook
class _FakeEmbedder:
    model_name = "fake/model"


def test_search_text_and_cache_key_unchanged_by_annotate():
    plain, typed = _frame(), pt.annotate(_frame())
    fields_plain, fields_typed = S.effective_fields(plain.columns), S.effective_fields(typed.columns)
    assert fields_plain == fields_typed
    assert not set(_ADDED) & set(fields_typed), fields_typed
    text_plain = S._build_search_text(plain, fields_plain)
    text_typed = S._build_search_text(typed, fields_typed)
    assert text_plain.tolist() == text_typed.tolist()
    with patched(S, {"get_default_embedder": lambda: _FakeEmbedder(),
                     "_file_hash": lambda path: "abc123"}):
        assert S._cache_key(fields_plain) == S._cache_key(fields_typed)


_INDEX_STATE = ("_df", "_embeddings", "_loaded_backend", "_loaded_fields", "_embedded_fields")


@contextlib.contextmanager
def _isolated_index():
    """Run a load_index/search test without disturbing a real loaded index."""
    saved = {k: getattr(S, k) for k in _INDEX_STATE}
    old_backend = cfg.SEARCH_BACKEND
    try:
        yield
    finally:
        for k, v in saved.items():
            setattr(S, k, v)
        cfg.SEARCH_BACKEND = old_backend


def test_load_index_local_annotates_before_lowercasing():
    seen = {}
    real_annotate = pt.annotate

    def spy(df):
        seen["verticals"] = set(df["bsns_vrtcl_name"].dropna())
        return real_annotate(df)

    with tempfile.TemporaryDirectory() as tmp, _isolated_index():
        csv = Path(tmp) / "catalog.csv"
        _frame().to_csv(csv, index=False)
        cfg.SEARCH_BACKEND = "local"
        S._df = None
        fake_embed = lambda texts: np.ones((len(texts), 4), dtype=np.float32)  # noqa: E731
        with patched(S, {"PRODUCTS_CSV": csv, "CACHE_DIR": Path(tmp), "embed_text": fake_embed,
                         "get_default_embedder": lambda: _FakeEmbedder()}), \
                patched(pt, {"annotate": spy}):
            S.load_index()
            fields = S.effective_fields(pd.read_csv(csv).columns)
            cache = Path(tmp) / f"embeddings_{S._cache_key(fields)}.npy"
            assert cache.exists()  # the cache key ignores the new columns
        assert seen["verticals"] == {"Fashion", "Electronics", "Grocery"}  # original case
        df = S._df
        assert df[pt.TYPE_COLUMN].tolist() == EXPECTED
        # Casing-agnostic: the display columns keep their case since 2026-10-15
        # (app/search.py lowercases copies); the '' fill for a missing value stays
        assert set(df["bsns_vrtcl_name"].str.lower()) == {"fashion", "electronics", "grocery", ""}
        reference = _frame()
        assert df["search_text"].tolist() == S._build_search_text(reference, fields).tolist()
        assert S._embedded_fields == fields
        # the taxonomy columns too, and the LOCAL load keeps them in CACHE_DIR, keyed by
        # the CSV's md5 (taxonomy_<md5>_<rules digest>.npz), never next to the catalog
        for column, want in TAXONOMY_EXPECTED.items():
            assert df[column].tolist() == want, column
        assert not set(catalog_taxonomy.ADDED_COLUMNS) & set(fields)
        names = sorted(p.name for p in Path(tmp).iterdir())
        assert f"taxonomy_{S._file_hash(csv)}_{S._taxonomy_digest()}.npz" in names, names
        assert names == sorted(["catalog.csv", Path(cache).name,
                                f"taxonomy_{S._file_hash(csv)}_{S._taxonomy_digest()}.npz"])


def test_load_index_databricks_annotates_the_catalog():
    import app.vector_search as vs

    catalog = _frame()
    catalog["catalog_index"] = range(len(catalog))
    with tempfile.TemporaryDirectory() as tmp, _isolated_index(), \
            patched(vs, {"load_catalog": lambda: catalog}), \
            patched(S, {"CACHE_DIR": Path(tmp)}):
        cfg.SEARCH_BACKEND = "databricks"
        S._df = None
        S.load_index()
        assert S._df is catalog and len(S._df) == 9
        assert S._df[pt.TYPE_COLUMN].tolist() == EXPECTED
        assert S._df["catalog_index"].tolist() == list(range(9))
        assert S._embedded_fields == S.effective_fields(_frame().columns)
        # the 2026-10-27 taxonomy columns ride the same load (contract 3.1), derived on
        # every Databricks load: no csv hash there, so no .cache/ file is read or written
        for column, want in TAXONOMY_EXPECTED.items():
            assert S._df[column].tolist() == want, column
            assert S._df[column].dtype == object, column
        assert not list(Path(tmp).iterdir())
        assert not set(catalog_taxonomy.ADDED_COLUMNS) & set(S._embedded_fields)


def _search_fixture():
    df = pt.annotate(pd.DataFrame({
        "bsns_vrtcl_name": ["fashion"] * 6 + ["electronics"],
        "Product_title": ["boho barefoot sandals anklet", "beaded toe ring barefoot sandals",
                          "flat leather sandals", "strappy wedge sandals", "linen sundress",
                          "gold sandals", "sandals ring light"],
    }))
    hits = [[(0, 0.95), (1, 0.94), (2, 0.90), (3, 0.89), (5, 0.80), (6, 0.70)],
            [(4, 0.91), (2, 0.85)]]
    return df, hits


def test_search_products_keep_runs_before_the_quota():
    df, hits = _search_fixture()
    calls = []

    def keep(row, term):
        calls.append((row["catalog_index"], row["source_intent"], term, row["score"]))
        return row.get(pt.TYPE_COLUMN) == pt.slot_type(term)

    with _isolated_index(), patched(S, {"_raw_hits": lambda terms, k: hits}):
        cfg.SEARCH_BACKEND = "local"
        S._df, S._embeddings, S._loaded_backend = df, np.zeros((len(df), 4)), "local"
        terms = ["beach wedding sandals", "linen dress"]
        plain = S.search_products(terms, top_k=30, per_intent_quota=2)
        assert S.search_products(terms, 30, 2) == plain               # positional call
        assert S.search_products(terms, 30, 2, keep=None) == plain    # None is a no-op
        assert [r["catalog_index"] for r in plain] == [0, 1, 4, 2]    # jewellery fills the quota
        kept = S.search_products(terms, top_k=30, per_intent_quota=2, keep=keep)
        assert [r["catalog_index"] for r in kept] == [4, 2, 3]       # footwear fills it now
        assert {r[pt.TYPE_COLUMN] for r in kept} == {"footwear", "apparel"}
        assert calls[0] == (0, "beach wedding sandals", "beach wedding sandals", 0.95)
        # the quota stopped the scan: rows after the second kept sandal were never offered
        assert [c[0] for c in calls if c[2] == "beach wedding sandals"] == [0, 1, 2, 3]


# ------------------------------------------------------------------ slow: the real catalog
def test_real_catalog_coverage_timing_and_needles_slow():
    """Coverage and speed of annotate() on the configured catalog, a 200-title sample
    to eyeball, and that no needle ever hides a pattern match on any Fashion title."""
    if os.getenv("CAS_SLOW_TESTS") != "1":
        _skip("set CAS_SLOW_TESTS=1 (and PRODUCTS_CSV) to type the real catalog")
    if not Path(cfg.PRODUCTS_CSV).exists():
        _skip(f"no catalog at {cfg.PRODUCTS_CSV}")
    # every column annotate() reads, where the catalog has it (the raw products.csv has
    # no gender tag: shopper_gender then reads the category bucket and the title)
    wanted = ["bsns_vrtcl_name", "categ_lvl2_name", "Product_title", "gender", "color"]
    df = pd.read_csv(cfg.PRODUCTS_CSV, low_memory=False,
                     usecols=lambda c: c in wanted)
    started = time.perf_counter()
    pt.annotate(df)
    elapsed = time.perf_counter() - started
    fashion = df["bsns_vrtcl_name"].map(pt.is_fashion)
    types = df.loc[fashion, pt.TYPE_COLUMN]
    covered = types.notna().mean()
    unknown = (types == pt.UNKNOWN).mean()
    print(f"\n  annotate: {len(df):,} rows ({int(fashion.sum()):,} Fashion) in {elapsed:.1f}s; "
          f"typed {covered:.1%}, unknown {unknown:.2%}; {pt.type_counts(df)}")
    for column, counts in pt.annotation_counts(df).items():
        print(f"  {column}: {counts}")
    for type_ in ("apparel", "footwear"):
        rows = df[pt.TYPE_COLUMN] == type_
        share = df.loc[rows, pt.FINE_SUBTYPE_COLUMN].notna().mean()
        print(f"  fine_subtype coverage, {type_}: {share:.1%}")
    rng = random.Random(7)
    sample = rng.sample(list(df.loc[fashion, "Product_title"].fillna("").index), 200)
    for i in sample:
        print(f"    {df.at[i, pt.TYPE_COLUMN]:9s} {str(df.at[i, pt.FINE_SUBTYPE_COLUMN]):14s} "
              f"{df.at[i, pt.GENDER_COLUMN]:6s} {df.at[i, pt.AUDIENCE_COLUMN]:5s} "
              f"| {str(df.at[i, 'Product_title'])[:90]}")
    assert covered >= 0.90, covered
    assert unknown <= 0.08, unknown
    assert not df.loc[~fashion, pt.TYPE_COLUMN].notna().any()
    assert elapsed < 30, f"annotate took {elapsed:.1f}s (target < 15s on the dev laptop)"
    # the C1 columns: Fashion rows always carry a gender and an audience, apparel and
    # footwear mostly a shop category; values only from the exported maps
    for column, allowed in ((pt.GENDER_COLUMN, pt.SHOPPER_GENDERS),
                            (pt.AUDIENCE_COLUMN, pt.AUDIENCES)):
        assert df.loc[fashion, column].notna().all(), column
        assert set(df.loc[fashion, column]) <= set(allowed), column
        assert not df.loc[~fashion, column].notna().any(), column
    assert set(df[pt.FINE_SUBTYPE_COLUMN].dropna()) <= set(pt.FINE_SUBTYPE_LABELS)
    apparel = df[pt.TYPE_COLUMN] == "apparel"
    footwear = df[pt.TYPE_COLUMN] == "footwear"
    assert df.loc[apparel, pt.FINE_SUBTYPE_COLUMN].notna().mean() >= 0.95
    assert df.loc[footwear, pt.FINE_SUBTYPE_COLUMN].notna().mean() >= 0.80
    assert df.loc[apparel, pt.FINE_SUBTYPE_COLUMN].dropna().isin(pt.APPAREL_SUBTYPES).all()
    assert df.loc[footwear, pt.FINE_SUBTYPE_COLUMN].dropna().isin(pt.FOOTWEAR_SUBTYPES).all()
    if "color" in df.columns:
        assert set(df[pt.COLOR_FAMILY_COLUMN].dropna()) <= set(pt.COLOR_FAMILIES)
        has_colour = df["color"].notna() & (df["color"].astype(str).str.strip() != "")
        on_fashion = df.loc[fashion & has_colour, pt.COLOR_FAMILY_COLUMN].notna().mean()
        print(f"  color_family set on {on_fashion:.1%} of Fashion rows with a colour")
        assert on_fashion >= 0.95, on_fashion

    # the local backend's shape: lowercased, then annotated -> the same columns
    lowered = df[[c for c in wanted if c in df.columns]].copy()
    for col in ("bsns_vrtcl_name", "categ_lvl2_name", "Product_title"):
        if col in lowered.columns:
            lowered[col] = lowered[col].fillna("").astype(str).str.lower()
    pt.annotate(lowered)
    for column in _ADDED:
        assert lowered[column].tolist() == df[column].tolist(), column

    texts = sorted({pt._normalise(t) for t in df.loc[fashion, "Product_title"]})
    hidden = _hidden_matches(texts)
    assert not hidden, hidden[:5]

    # 2026-10-27: rows typed accessory only because the watch reader calls them a watch
    # or a watch band (jewelry without it): 500 rows of 2026-10-27's catalog
    accessories = df.loc[fashion & (df[pt.TYPE_COLUMN] == "accessory"), "Product_title"]
    with patched(pt, {"_watch_or_band": lambda t: False}):
        moved = sum(pt.classify_title(t)[0] == "jewelry" for t in accessories)
    print(f"  watch fix: {moved} Fashion rows accessory, not jewelry")
    assert 450 <= moved <= 550, moved


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
