"""Offline tests for app/product_type.py and its app/search.py hook.

No network, no LLM, no embedding model: the search-hook tests fake the embedder and
the retrieval. Runs under pytest (`python -m pytest tests/test_product_type.py`) or
plainly (`python tests/test_product_type.py`).

Pinned here: the hand-labelled title table (in original case AND lowercased, the
local backend's shape), the shopper-ask table the ranking acts on, the documented
vocabulary collisions main.py's pool gate must neutralise, that annotate() never
moves a row (catalog_index is a ROW POSITION), that search_text and the embedding
cache key are unchanged by it, and that `keep=` filters before the per-intent quota.
CAS_SLOW_TESTS=1 adds the full-catalog coverage, timing and needle-soundness check.
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
    # unknown: a shoe named by model alone, a size string, D-ring hardware
    ("Nike Air Max Plus (Girls Grade School) (5, Black/White/Vivid Pink)", "unknown"),
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
}
NONE = [
    "beach wedding", "gift for my wife", "cozy desk setup ideas", "wireless earbuds",
    "instant noodles", "never lose my keys again", "beach wedding outfit",
    "linen outfit ideas", "breathable outfit for a humid day", "outfit for a job interview",
    "hiking outfit", "running gear for winter", "rain gear for hiking",
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
    "ring bearer outfit", "tie dye", "boot cut", "kentucky derby",
    # ...nor is the tie of a dress code
    "black tie event", "black-tie event", "black tie optional", "white tie gala",
]
# Known residuals, pinned so a change is seen (not the desired answers): a bare
# singular noun BEFORE the typed part, or between typed parts, reads as a modifier
# (the price of keeping "light, breezy clothes" a clothing ask); 'wear' as a noun is
# no clothing noun.
RESIDUALS = {
    "headlamp and hiking boots": frozenset({"footwear"}),
    "sunscreen and a swimsuit": APPAREL,
    "dress, sunscreen and a hat": frozenset({"apparel", "headwear"}),
    "black tie wear": ACCESSORY,
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
    wrong = {q: (sorted(w), pt.requested_types(q)) for q, w in RESIDUALS.items()
             if pt.requested_types(q) != w}
    assert not wrong, wrong


def test_requested_types_shape_and_case():
    for q, want in REQUESTED.items():
        got = pt.requested_types(q)
        assert isinstance(got, frozenset)
        assert pt.requested_types(q.upper()) == want, q
        assert pt.requested_types("  " + q.replace(" ", "   ") + "  ") == want, q
    assert pt.requested_types("What\u2019s good to wear to a wedding") == APPAREL


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


def test_annotate_types_rows_and_leaves_others_none():
    df = pt.annotate(_frame())
    assert df[pt.TYPE_COLUMN].tolist() == EXPECTED
    assert df[pt.SUBTYPE_COLUMN].tolist()[5:] == [None] * 4
    assert df[pt.TYPE_COLUMN].dtype == object and df[pt.SUBTYPE_COLUMN].dtype == object
    for v in df[pt.TYPE_COLUMN].tolist() + df[pt.SUBTYPE_COLUMN].tolist():
        assert v is None or isinstance(v, str), repr(v)
    # row dicts read None, never NaN (main.py and the API see these)
    assert df.iloc[6].to_dict()[pt.TYPE_COLUMN] is None


def test_annotate_after_local_lowercasing_is_identical():
    """The local backend's _build_search_text lowercases the vertical and titles IN
    PLACE. load_index annotates BEFORE it, but because annotate is case-insensitive
    the order cannot silently switch the feature off; a literal == "Fashion" test
    would have typed nothing on the lowercased frame."""
    title_case = pt.annotate(_frame())
    local = _frame()
    S._build_search_text(local, S.effective_fields(local.columns))
    assert set(local["bsns_vrtcl_name"]) == {"fashion", "electronics", "grocery", ""}
    assert (local["bsns_vrtcl_name"] == "Fashion").sum() == 0  # the trap
    pt.annotate(local)
    assert local[pt.TYPE_COLUMN].tolist() == title_case[pt.TYPE_COLUMN].tolist()
    assert local[pt.SUBTYPE_COLUMN].tolist() == title_case[pt.SUBTYPE_COLUMN].tolist()
    assert pt.type_counts(local) == pt.type_counts(title_case)


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
    empty = pt.annotate(pd.DataFrame(columns=["bsns_vrtcl_name", "Product_title"]))
    assert list(empty.columns)[-2:] == [pt.TYPE_COLUMN, pt.SUBTYPE_COLUMN] and len(empty) == 0
    no_vertical = pt.annotate(pd.DataFrame({"Product_title": ["Red Dress"]}))
    assert no_vertical[pt.TYPE_COLUMN].tolist() == [None]
    no_title = pt.annotate(pd.DataFrame({"bsns_vrtcl_name": ["Fashion"]}))
    assert no_title[pt.TYPE_COLUMN].tolist() == [None]


def test_annotate_reads_pandas_string_dtype():
    """read_csv gives pandas' str dtype (NaN for missing); toPandas gives object/None."""
    df = _frame()
    df = df.astype({"bsns_vrtcl_name": "str", "Product_title": "str"})
    assert pt.annotate(df)[pt.TYPE_COLUMN].tolist() == EXPECTED


# ------------------------------------------------------------------ the search hook
class _FakeEmbedder:
    model_name = "fake/model"


def test_search_text_and_cache_key_unchanged_by_annotate():
    plain, typed = _frame(), pt.annotate(_frame())
    fields_plain, fields_typed = S.effective_fields(plain.columns), S.effective_fields(typed.columns)
    assert fields_plain == fields_typed
    assert pt.TYPE_COLUMN not in fields_typed and pt.SUBTYPE_COLUMN not in fields_typed
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
        assert set(df["bsns_vrtcl_name"]) == {"fashion", "electronics", "grocery", ""}
        reference = _frame()
        assert df["search_text"].tolist() == S._build_search_text(reference, fields).tolist()
        assert S._embedded_fields == fields


def test_load_index_databricks_annotates_the_catalog():
    import app.vector_search as vs

    catalog = _frame()
    catalog["catalog_index"] = range(len(catalog))
    with _isolated_index(), patched(vs, {"load_catalog": lambda: catalog}):
        cfg.SEARCH_BACKEND = "databricks"
        S._df = None
        S.load_index()
        assert S._df is catalog and len(S._df) == 9
        assert S._df[pt.TYPE_COLUMN].tolist() == EXPECTED
        assert S._df["catalog_index"].tolist() == list(range(9))
        assert S._embedded_fields == S.effective_fields(_frame().columns)


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
    df = pd.read_csv(cfg.PRODUCTS_CSV, usecols=["bsns_vrtcl_name", "Product_title"],
                     low_memory=False)
    started = time.perf_counter()
    pt.annotate(df)
    elapsed = time.perf_counter() - started
    fashion = df["bsns_vrtcl_name"].map(pt.is_fashion)
    types = df.loc[fashion, pt.TYPE_COLUMN]
    covered = types.notna().mean()
    unknown = (types == pt.UNKNOWN).mean()
    print(f"\n  annotate: {len(df):,} rows ({int(fashion.sum()):,} Fashion) in {elapsed:.1f}s; "
          f"typed {covered:.1%}, unknown {unknown:.2%}; {pt.type_counts(df)}")
    rng = random.Random(7)
    sample = rng.sample(list(df.loc[fashion, "Product_title"].fillna("").index), 200)
    for i in sample:
        print(f"    {df.at[i, pt.TYPE_COLUMN]:9s} | {str(df.at[i, 'Product_title'])[:100]}")
    assert covered >= 0.90, covered
    assert unknown <= 0.08, unknown
    assert not df.loc[~fashion, pt.TYPE_COLUMN].notna().any()
    assert elapsed < 30, f"annotate took {elapsed:.1f}s (target < 15s on the dev laptop)"

    # the local backend's shape: lowercased, then annotated -> the same types
    lowered = df[["bsns_vrtcl_name", "Product_title"]].copy()
    S._build_search_text(lowered, ("bsns_vrtcl_name", "Product_title"))
    pt.annotate(lowered)
    assert lowered[pt.TYPE_COLUMN].tolist() == df[pt.TYPE_COLUMN].tolist()

    texts = sorted({pt._normalise(t) for t in df.loc[fashion, "Product_title"]})
    hidden = _hidden_matches(texts)
    assert not hidden, hidden[:5]


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
