"""Offline tests for app/compact_text.py and scripts/compact_search_text.py (2026-11-20):
the compact embedding text of rows the local embedder would cut. No model, no catalog: a
whitespace word count stands in for the tokenizer (a wordpiece tokenizer counts a
space-joined text as the sum of its pieces, which is all compact_text relies on; checked
on gte-small by the 2026-11-20 review). The API's load path (SEARCH_TEXT_COMPACT in
search.load_index) is also in tests/test_search_text.py.

Runs under pytest (`python -m pytest tests/test_compact_text.py`) or plainly
(`python tests/test_compact_text.py`).
"""

import contextlib
import json
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import app.config as cfg  # noqa: E402
from app import compact_text as ct  # noqa: E402
from app import search as S  # noqa: E402


def words(text):
    return len(text.split())


def _row(**cols):
    base = {"bsns_vrtcl_name": "Fashion", "categ_lvl2_name": "Women's Clothing",
            "Product_title": "Floral Midi Dress", "prod_description": "", "color": "",
            "material": "", "occasion": "", "gender": "", "use_case": "", "dietary": "",
            "specs": "", "gen_description": ""}
    base.update(cols)
    return base


LONG_DESC = ("Made from soft breathable fabric that drapes well. " * 6
             + "FLATTERING FIT: elastic waist and a tie belt. NO PRESERVATIVES. "
             + "If you have any questions please contact us. "
             + "Machine wash cold, tumble dry low.")


# ------------------------------------------------------------------ the title
def test_title_parts_that_only_repeat_go_whole():
    t = ("botko foods grain free crackers 3 pack | gluten free crackers | vegan snack | "
         "dairy free, no added sugar | 3 pack | grain free crackers")
    out = ct.dedup_title(t)
    assert out == ("botko foods grain free crackers 3 pack | gluten free crackers | vegan snack | "
                   "dairy free, no added sugar")
    assert "gluten free" in out and "dairy free" in out           # phrases never broken


def test_title_numbers_keep_their_units_and_digits():
    for t in ("backup camera harness, multicolored, 4.00 inch x 3.00 inch x 1.15 inch",
              # a comma between digits is a number, never a part boundary (review C1)
              "160w led lamp 30 degree beam,30,000 hours lamp life",
              "kids shoes size 4,5,6,7,8,9,10 | 9 carat",
              "case for iphone 13,12,11/13,12,11 pro max",
              "chili extract,2,305 servings, 250,000-1,000,000 scovilles",
              "imac 17,1 memory 1.35v",
              "gopro session, hero5, 4, 3+, 3, 2,1 | 10 in 1 kit"):
        assert ct.dedup_title(t) == t, t
    # a numbered part goes only when the same phrase already stands in the title
    assert ct.dedup_title("usb cable 6 ft | 2 pack | fast charge | 2 pack") == \
        "usb cable 6 ft | 2 pack | fast charge"


def test_title_keeps_negated_and_single_letter_parts():
    """Review Q30 / Q31: 'not braided' says the opposite of 'braided'; 'xbox series s'
    and 'wii u' are other devices than 'xbox series x' and 'wii'."""
    for t in ("silicone braided solo loop band (silicone material, not braided)",
              "controller for xbox series x, xbox series s, wii, wii u",
              "galaxy tab, galaxy tab a, galaxy tab s",
              "phone charger, no battery, battery pack"):
        assert ct.dedup_title(t) == t, t


def test_title_with_nothing_repeated_is_unchanged():
    for t in ("women's floral midi dress, summer casual", "anker powercore 10000 portable charger",
              "a - b", "", "x | y"):
        assert ct.dedup_title(t) == t


# ------------------------------------------------------------------ words
def test_plurals_fold_and_nothing_else():
    pairs = [("dresses", "dress"), ("shoes", "shoe"), ("sizes", "size"),
             ("batteries", "battery"), ("cookies", "cookie"), ("boxes", "box"),
             ("watches", "watch"), ("cases", "case"), ("glasses", "glass")]
    for plural, singular in pairs:
        assert ct._stem(plural) == ct._stem(singular), (plural, singular)
    for a, b in (("capes", "cap"), ("notes", "not"), ("kites", "kit")):
        assert ct._stem(a) != ct._stem(b), (a, b)


def test_said_keys_units_negation_accents():
    assert ct.said_keys("512GB SSD 3.70 ml 12 fl oz 65W") == \
        ct.said_keys("512 gb ssd 3.7 milliliters 12 fluid ounces 65 watts")
    assert ct.said_keys("no added sugar") == ["no_added", "sugar"]
    assert "no_braided" in ct.said_keys("not braided")
    assert ct.said_keys("Café Mazel") == ct.said_keys("cafe mazel")
    assert ct.said_keys("Women's") == ["women"]
    assert ct.said_keys("Number of USB Ports 4") == ["number", "usb", "port", "4"]


def test_blank_reads_every_missing_value_as_empty():
    for value in (None, float("nan"), np.float32("nan"), np.float64("nan"), pd.NA):
        assert ct._blank(value) == "", value
    assert ct._blank(0) == "0" and ct._blank("x") == "x"


# ------------------------------------------------------------------ description segments
def test_segments_split_bullets_and_headers_but_not_inside_a_header():
    desc = ("Great taste. MINERALS AND ELECTROLYTES FOR TASTE: hydration matters "
            "【SOFT FABRIC】 cotton blend • machine wash; tumble dry ✅ gift ready")
    segs = ct.segments(desc)
    assert segs == ["Great taste.", "MINERALS AND ELECTROLYTES FOR TASTE: hydration matters",
                    "SOFT FABRIC cotton blend", "machine wash", "tumble dry", "gift ready"]
    assert ct.segments("") == [] and ct.segments(None) == [] and ct.segments(float("nan")) == []


def test_segments_keep_numbers_and_abbreviations_whole():
    assert ct.segments("Dimensions: 10 in. x 5 in. x 2 in. Great gift.") == \
        ["Dimensions: 10 in. x 5 in. x 2 in.", "Great gift."]
    assert ct.segments("Works from -40°C to 85°C. - Durable shell") == \
        ["Works from -40°C to 85°C.", "Durable shell"]
    assert ct.segments('HP x360 13. 3" touch screen. Fast.') == \
        ['HP x360 13. 3" touch screen.', "Fast."]


def test_boilerplate():
    for s in ("If you have any questions, please contact us.", "Buy with confidence!",
              "All brand names and logos are registered trademarks of their owners.",
              "Please check the size chart before ordering", "100% satisfaction guaranteed"):
        assert ct.is_boilerplate(s) and ct.strip_boilerplate(s) == "", s
    for s in ("100% cotton, machine wash", "No preservatives.", "Contact grill for steaks",
              "Fits iPhone 12 and 13"):
        assert not ct.is_boilerplate(s) and ct.strip_boilerplate(s) == s, s
    # a long segment loses only its boilerplate clause (review C2)
    long = ("Interface: SATA, Capacity: 2000GB, RPM: 5400RPM, Cache: 8MB, Thickness 9.5mm, "
            "fits Dell Latitude D531 and D620 laptops, tested before shipping, 30 Days Money "
            "Back Guarantee, 2TB 2.5 inch laptop hard drive for Dell")
    out = ct.strip_boilerplate(long)
    assert "money back" not in out.lower() and "Capacity: 2000GB" in out and "D620" in out


def test_long_boilerplate_clause_loses_only_the_words_around_the_phrase():
    """Review round 2 (C2): Amazon bullets are joined with spaces, so a long run with no
    ', ' is one clause; dropping it whole lost the specs around a trademark line."""
    run = ("Mini battery charger kit with smart charging LED indicator automatically turns to "
           "trickle charge after fully charged 3 Year Limited Warranty - Synergy Digital is a "
           "Registered Trademark Mini Battery charger fits Canon NB-11L batteries perfectly")
    out = ct.strip_boilerplate(run)
    assert "smart charging LED indicator" in out and "3 Year Limited Warranty" in out
    assert "Registered Trademark" not in out and "Digital is a" not in out
    # 'click the' / 'feel free to' are product facts unless they ask to order or write
    for fact in ("Simply click the one-touch button to pick up, hang up, and pause music.",
                 "All the material we used is non-toxic, you just feel free to use or enjoy it."):
        assert ct.strip_boilerplate(fact) == fact, fact
    for noise in ("Feel free to contact us anytime.", "Click the Add to Cart button now.",
                  "Shipping information: international airmail, 10 - 20 days"):
        assert ct.strip_boilerplate(noise) == "", noise


def test_bounds_stay_and_are_spaced():
    """≤ / ≥ are gte-small tokens and change a figure's meaning; glued ('cri≥95') they
    split the words around them into pieces, so they are spaced. Decoration still goes."""
    assert ct._clean("CRI≥95, ≤5V ★Soft") == "cri ≥ 95, ≤ 5v soft"
    assert ct.segments("Input ≤ 5V. Output ≥ 2A") == ["Input ≤ 5V.", "Output ≥ 2A"]
    # a key's ':' before a bound goes; a ':' between two figures stays (review round 3)
    assert ct._clean("Insertion Loss: ≤0.3dB, SWR ≤ 2: ≥ 1300 kHz") == \
        "insertion loss ≤ 0.3db, swr ≤ 2: ≥ 1300 khz"


def test_boilerplate_round_3():
    """Store and personalisation instructions are boilerplate; a fact in front of a short
    boilerplate phrase is kept; the cut window stops before a negator it would split."""
    for noise in ("Click on the name of our shop above the title to see more designs.",
                  "Please click the customize now button to add your text.",
                  "Feel free to talk with us within 7 days.", "Ask a question to the seller.",
                  "Shop with confidence, we stand behind our products."):
        assert ct.strip_boilerplate(noise) == "", noise
    assert ct.strip_boilerplate("65% polyester, 35% cotton, 30 days money back guarantee") == \
        "65% polyester, 35% cotton"
    assert ct.strip_boilerplate("Soft and cozy, 30 days money back guarantee") == ""
    # round 4: a size table or a tolerance note before the phrase is no fact; one stated
    # size is; a full stop with no space hides a sentence break; a seller who asks you to
    # check compatibility states the compatibility
    assert ct.strip_boilerplate(
        "Bust:(S)33.1-35.4 inches, (M)35.4-37.4 inches. Please check the size chart") == ""
    assert ct.strip_boilerplate("sizes range from 1 to 3 inches, detailed size chart is "
                                "included") == "sizes range from 1 to 3 inches"
    assert ct.strip_boilerplate("There may be 1-3cm error, if you have any questions "
                                "contact us") == ""
    assert ct.strip_boilerplate(
        'Click The "Customize Now" Button,Two Styles Of Texas State Flag Bandana Design For '
        "You To Choose.Material: Soft,") == "Material: Soft,"
    assert ct.strip_boilerplate(
        "COMPATIBILITY: Works with most corded desk phones- contact seller to verify "
        "compatibility- NOT COMPATIBLE with computers.").startswith("COMPATIBILITY")
    assert ct.strip_boilerplate("In case of any question or problem, please contact the "
                                "seller.") == ""
    # one clause of 30+ words; the 8th word after the phrase is 'not'
    run = ("Fast charging cable for phones and tablets with braided nylon jacket customer "
           "service team will gladly help you right now not backwards compatible with older "
           "USB 1.1 ports so check your device first")
    out = ct.strip_boilerplate(run)
    assert "customer service" not in out and "now not backwards compatible" not in out
    assert "not backwards compatible with older USB 1.1 ports" in out, out


def test_said_keys_fold_thousands_and_keep_frequencies():
    assert ct.said_keys("20,000 mAh") == ct.said_keys("20000 mah") == ["20000mah"]
    assert ct.said_keys("3,800 watts") == ct.said_keys("3800 W")
    assert ct.said_keys("10,000 mAh") != ct.said_keys("20,000 mAh")
    assert ct.said_keys("usb 3.0,100w") == ["usb", "3", "0", "100w"]   # no fold in a list
    assert ct.said_keys("13,12,11") == ["13", "12", "11"]
    assert ct.said_keys("144 Hz") == ["144hz", "144"] == ct.said_keys("144hz")
    assert ct.said_keys("3.5 GHz") == ct.said_keys("3.5ghz")
    # a decimal frequency keeps one fused bare key, never '2' and '4' (review rounds 3-4)
    assert ct.said_keys("2.4 GHz") == ["2p4ghz", "2p4"]
    row = _row(bsns_vrtcl_name="Electronics", categ_lvl2_name="Electronics",
               Product_title="Mini PC 4.1GHz", specs="Number of USB 2.0 Ports: 4")
    assert ct.compact_text(row, words, budget=50).endswith("number of usb 2.0 ports 4")


def test_spec_items():
    items = dict(ct._spec_items(
        "Unit Count: 1.00 Count; Other display features: Wireless; Number of USB 2.0 Ports: 4; "
        "Batteries Required: No; Compatible Devices: iPhone 12; Item Form: Powder; "
        "Age Range (Description): Adult; Unit Count: 12.00 Count; Voltage: 12 Volts"))
    assert list(items) == ["Number of USB 2.0 Ports 4", "compatible with iPhone 12", "Powder",
                           "Adult", "12.00 Count", "12 Volts"]


def test_spec_items_round_2():
    """Review round 2: a yes flag's key is the fact; a colour that is no colour, a compat
    warning and 'attribute not applicable' go; role keys keep their key."""
    items = [t for t, _ in ct._spec_items(
        "Is Waterproof: Yes; Is Cordless?: Yes; Assembly Required: Yes; Brand: Yes; "
        "Warranty Description: Yes; Is Dishwasher Safe: No; Color: 14; Color: Default; "
        "Colour: See Picture; Color: Navy; Color Rendering Index (CRI): 90; "
        "Compatible Devices: Not for iPhone 12; Compatible Devices: Galaxy S21; "
        "Allergen Information: Dairy; Input Voltage: 240 Volts; Included Components: DVD; "
        "Style: Attribute Not Applicable")]
    assert items == ["Waterproof", "Cordless", "Assembly Required", "Navy",
                     "Color Rendering Index 90", "compatible with Galaxy S21",
                     "Allergen Information Dairy", "Input Voltage 240 Volts",
                     "Included Components DVD"]
    row = _row(Product_title="Silk Scarf", color="Default")
    assert ct.compact_text(row, words, budget=50) == "fashion women's clothing silk scarf"
    for value in ("Random", "No Color", "Same as Picture Shown", "As Pictured", "Color 17",
                  "14", "#31-31", "黑色"):
        assert ct.is_placeholder_colour(value), value
    for value in ("Multicolor", "Colorful", "Rose Gold", "Colorblock", "Mixed", "Clear"):
        assert not ct.is_placeholder_colour(value), value
    # 'Included Components' naming nothing (round 3)
    junk = [t for t, _ in ct._spec_items(
        "Included Components: Not Null; Included Components: 1; Included Components: Owner's "
        "Manual; Included Components: User Manual, Warranty Card; Included Components: User "
        "Manual, Battery")]
    assert junk == ["Included Components User Manual, Battery"]
    # a longer number is a part number, not a count (round 4)
    assert [t for t, _ in ct._spec_items("Included Components: 100967")] == \
        ["Included Components 100967"]


def test_a_unit_count_already_said_goes():
    row = _row(bsns_vrtcl_name="Grocery", categ_lvl2_name="Grocery",
               Product_title="Green Tea Bags, 18 Pack", specs="Unit Count: 18.00 Count")
    assert "count" not in ct.compact_text(row, words, budget=50)
    row["Product_title"] = "Green Tea Bags"
    assert ct.compact_text(row, words, budget=50).endswith("18.00 count")
    row["Product_title"] = "Green Tea Bags 18ct"
    assert "count" not in ct.compact_text(row, words, budget=50)


def test_a_frequency_said_in_the_title_is_not_repeated():
    row = _row(bsns_vrtcl_name="Electronics", categ_lvl2_name="Electronics",
               Product_title="27 inch Gaming Monitor 144Hz", specs="Refresh Rate: 144 Hz")
    assert ct.compact_text(row, words, budget=50) == "electronics 27 inch gaming monitor 144hz"


# ------------------------------------------------------------------ compact_text
def test_every_tag_fits_and_the_text_stays_in_budget():
    row = _row(color="Navy Blue", material="Rayon", occasion="Wedding", gender="women",
               use_case="Beach", specs="Brand: Floral Co; Sleeve Type: Short Sleeve; "
               "Neckline: V-Neck", prod_description=LONG_DESC)
    out = ct.compact_text(row, words, budget=40)
    assert words(out) <= 40
    for value in ("navy blue", "rayon", "wedding", "beach", "short sleeve", "v-neck"):
        assert value in out, value
    assert out.startswith("fashion women's clothing floral midi dress")
    assert "contact us" not in out                                   # boilerplate
    assert out.count("soft breathable fabric") == 1                  # the repeated sentence once
    assert "no preservatives" in out                                 # a one-word fact kept


def test_tags_and_specs_already_in_the_title_are_not_repeated():
    row = _row(Product_title="Women's Black Leather Jacket", color="black", material="Leather",
               gender="women", specs="Brand: Leather Jacket; Color: Black; Closure Type: Zipper")
    out = ct.compact_text(row, words, budget=200)
    assert out == "fashion women's clothing women's black leather jacket zipper"


def test_a_comma_list_tag_adds_only_its_new_parts():
    row = _row(bsns_vrtcl_name="Grocery", categ_lvl2_name="Grocery",
               Product_title="Keto Crackers", dietary="vegan, keto")
    assert ct.compact_text(row, words, budget=50) == "grocery keto crackers vegan"
    row = _row(Product_title="Silk Scarf", color="14")               # a number is no colour
    assert ct.compact_text(row, words, budget=50) == "fashion women's clothing silk scarf"


def test_department_said_once():
    row = _row(bsns_vrtcl_name="Grocery", categ_lvl2_name="Grocery", Product_title="Oat Milk")
    assert ct.compact_text(row, words, budget=50) == "grocery oat milk"


def test_symbols_never_glue_to_words():
    row = _row(Product_title="💖Soft Hoodie🖤", prod_description="🍫British fruitcake ✅ gift tin")
    out = ct.compact_text(row, words, budget=50)
    assert out == "fashion women's clothing soft hoodie british fruitcake gift tin"


def test_repeated_runs_and_keyword_stuffing_go():
    stuffed = "pullover sweatshirts for women casual crewneck " * 8 + "fleece lined warm."
    row = _row(Product_title="Pullover", prod_description=stuffed)
    out = ct.compact_text(row, words, budget=60)
    assert out.count("crewneck") == 1 and "fleece lined warm." in out


def test_the_last_segment_that_does_not_fit_is_clipped_at_a_word():
    row = _row(prod_description="Alpha beta gamma delta epsilon zeta eta theta iota kappa.")
    out = ct.compact_text(row, words, budget=12)
    assert words(out) == 12
    assert out.endswith("alpha beta gamma delta epsilon zeta")       # whole words only
    none = ct.compact_text(row, words, budget=12, clip_last=False)
    assert "alpha" not in none


def test_a_clip_never_ends_dangling():
    """Review round 2: no trailing stop word or negator, no number without the unit the
    source gives it, no word without the 'free' that follows it. Only words go."""
    for clipped, source, want in (
            ("cholesterol free, gluten", "cholesterol free, gluten free and tasty",
             "cholesterol free,"),
            ("net weight 9.3", "net weight 9.3 oz per bag", "net weight"),
            ("runs for 30", "runs for 30 hours on one charge", "runs"),
            ("perfect for the", "perfect for the beach", "perfect"),
            ("contains no", "contains no sugar", "contains"),
            ("ice cream", "ice cream freezer", "ice cream"),
            ("pack of 30", "pack of 30 bags", "pack of 30"),
            ("adapter 10", "adapter 10 in 1 hub", "adapter"),   # cut from 'in 1', 10 says nothing
            ("screen 13", "screen 13 in laptops", "screen"),
            ("size 10 x", "size 10 x 5 cm", "size 10")):
        assert ct._trim_clip(clipped, source) == want, (clipped, want)
    row = _row(prod_description="Soft cotton blend that is gluten free and light.")
    out = ct.compact_text(row, words, budget=12)                      # room for 6 words
    assert out.endswith("floral midi dress soft cotton blend"), out    # not '... is gluten'


def test_a_list_number_is_no_sentence_end():
    assert not ct._is_sentence("Detailed Information: 1.")
    assert ct._is_sentence("Works with covid19.") and ct._is_sentence("Strong grip!")


def test_spare_room_readmits_the_new_words_of_a_templated_sentence():
    """Review round 2: the 80% 'mostly said' test dropped the one variant word of a
    templated sentence; with room left, its repeat-free rest comes back."""
    template = "This is the same hot sauce served at our family restaurant, perfect to add {} "\
               "flavor to any meal."
    row = _row(bsns_vrtcl_name="Grocery", categ_lvl2_name="Grocery", Product_title="Hot Sauce",
               prod_description=template.format("smoky chipotle") + " "
               + template.format("zesty jalapeno"))
    out = ct.compact_text(row, words, budget=60)
    assert "smoky chipotle" in out and "zesty jalapeno" in out
    assert out.count("same hot sauce") == 1
    tight = ct.compact_text(row, words, budget=words(out) - 3)
    assert "zesty" not in tight                                       # only with room left
    # never a scrap: a rest with no content word or a bare header (review round 3)
    title = "Classic Hot Sauce Made With Red Peppers"
    row = _row(Product_title=title, prod_description=title + " 1")   # mostly said; rest '1'
    assert ct.compact_text(row, words, budget=60) == \
        "fashion women's clothing classic hot sauce made with red peppers"


def test_after_a_miss_only_whole_sentences_are_added():
    desc = ("ULTRA WIDE COMPATIBILITY works with every phone sold this decade and most "
            "tablets too. Safety tested. Strong grip! wide compat")
    row = _row(Product_title="Phone Mount", prod_description=desc)
    out = ct.compact_text(row, words, budget=13, clip_last=False)
    assert "safety tested." in out and "strong grip!" in out
    assert not out.endswith("wide compat")                           # a scrap, not a sentence


def test_a_title_over_budget_alone_is_clipped():
    row = _row(Product_title=" ".join(f"w{i}" for i in range(50)), color="red")
    out = ct.compact_text(row, words, budget=20)
    assert words(out) <= 20 and out.startswith("fashion women's clothing w0 w1")


def test_extractive_never_a_word_the_listing_lacks():
    row = _row(color="Navy", material="Cotton", use_case="Gift", specs="Pattern: Floral",
               prod_description=LONG_DESC)
    out = ct.compact_text(row, words, budget=45)
    source = " ".join(str(v) for v in row.values()).lower()
    source_words = set(re.findall(r"[a-z0-9]+", source)) | {"compatible", "with"}
    assert set(re.findall(r"[a-z0-9]+", out)) <= source_words


def test_nan_and_missing_fields_read_as_empty():
    row = {"bsns_vrtcl_name": "Electronics", "Product_title": "USB Cable",
           "color": float("nan"), "prod_description": None, "specs": pd.NA}
    assert ct.compact_text(row, words, budget=20) == "electronics usb cable"


# ------------------------------------------------------------------ sidecar bookkeeping
def test_sidecar_names_follow_the_catalog_and_the_version():
    side, meta = ct.sidecar_paths(Path("data") / "products_enriched.csv")
    assert side == Path("data") / f"products_enriched.compact_v{ct.COMPACT_VERSION}.csv"
    assert meta == Path("data") / f"products_enriched.compact_v{ct.COMPACT_VERSION}.meta.json"
    long_name = "embeddings_f62483021a64_thenlper_gte-small_" + "-".join(["field"] * 25) + ".npy"
    cache = ct.compact_cache_path(Path(".cache") / long_name)
    assert len(cache.name) <= 40 and cache.name.endswith(f"_compact{ct.COMPACT_VERSION}.npy")
    assert ct.stamp_path(cache).suffix == ".json"


def test_sidecar_matches_names_the_first_difference():
    meta = {"version": ct.COMPACT_VERSION, "csv_hash": "h", "n_rows": 10, "model": "m",
            "max_seq_length": 128, "fields": ["a", "b"]}
    ok = ("h", 10, "m", 128, ("a", "b"))
    assert ct.sidecar_matches(meta, *ok) is None
    assert ct.sidecar_matches(meta, "other", 10, "m", 128, ("a", "b")) == "csv_hash differs"
    assert ct.sidecar_matches(meta, "h", 11, "m", 128, ("a", "b")) == "n_rows differs"
    assert ct.sidecar_matches(meta, "h", 10, "m2", 128, ("a", "b")) == "model differs"
    assert ct.sidecar_matches(meta, "h", 10, "m", 256, ("a", "b")) == "max_seq_length differs"
    assert ct.sidecar_matches(meta, "h", 10, "m", 128, ("a",)) == "fields differs"
    assert ct.sidecar_matches(dict(meta, version="0"), *ok) == "version differs"
    assert ct.changed_rows([5, 130, 126, 127], 126) == [1, 3]


def test_check_rows():
    meta = {"n_changed": 2}
    asins = np.array(["A", "B", "C"])
    assert ct.check_rows(meta, [0, 2], ["A", "C"], asins, 3) is None
    assert ct.check_rows(meta, [0], ["A"], asins, 3) == "row count differs"
    assert ct.check_rows(meta, [0, 3], ["A", "D"], asins, 3) == "rows out of range"
    assert ct.check_rows(meta, [0, 1], ["A", "C"], asins, 3) == "parent_asin differs"


# ------------------------------------------------------------------ the one-time script
class _FakeTokenizer:
    """One token per whitespace word, so budgets are word counts."""

    def __call__(self, texts, add_special_tokens=False, truncation=False, max_length=None):
        single = isinstance(texts, str)
        ids = [t.split() for t in ([texts] if single else texts)]
        if truncation:
            ids = [i[:max_length] for i in ids]
        return {"input_ids": ids[0] if single else ids}

    def decode(self, ids):
        return " ".join(ids)


class _FakeST:
    tokenizer = _FakeTokenizer()


class _FakeEmbedder:
    model_name = "fake/model"
    is_openai = False
    _st_model = _FakeST()

    def max_seq_length(self):
        return 14                      # budget 12 words


@contextlib.contextmanager
def _patched(obj, attrs):
    old = {k: getattr(obj, k) for k in attrs}
    for k, v in attrs.items():
        setattr(obj, k, v)
    try:
        yield
    finally:
        for k, v in old.items():
            setattr(obj, k, v)


def _catalog():
    long_desc = "Soft breathable cotton fabric. Elastic waist with pockets. Machine wash cold."
    return pd.DataFrame({
        "bsns_vrtcl_name": ["Fashion"] * 4,
        "categ_lvl2_name": ["Women's Clothing"] * 4,
        "Product_title": ["Short Tee", "Linen Wide Leg Pants", "Wool Coat", "Silk Scarf"],
        "prod_description": ["", long_desc, long_desc, ""],
        "color": ["red", "beige", "camel", "blue"],
        "material": ["cotton", "linen", "wool", "silk"],
        "parent_asin": ["A1", "A2", "A3", "A4"],
    })


def test_build_encode_load_end_to_end():
    """build writes the sidecar for the rows over budget only; encode re-encodes only
    them, keeps every other vector, writes short-named files and a stamp; the API serves
    the compact text only while the stamp names this sidecar; a rebuilt sidecar is
    never served old vectors, and resume never reuses another sidecar's parts."""
    from scripts import compact_search_text as cst
    calls = []

    def fake_embed(texts):
        calls.append(list(texts))
        return np.full((len(texts), 4), 2.0, dtype=np.float32)

    saved = {k: getattr(S, k) for k in ("_df", "_embeddings", "_loaded_backend", "_loaded_fields",
                                        "_embedded_fields", "_csv_hash_of", "_search_text_mode",
                                        "_loaded_compact")}
    old_backend, old_knob = cfg.SEARCH_BACKEND, cfg.SEARCH_TEXT_COMPACT
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            csv = tmp / "catalog.csv"
            _catalog().to_csv(csv, index=False)
            fake = _FakeEmbedder()
            env = dict(PRODUCTS_CSV=csv, CACHE_DIR=tmp, get_default_embedder=lambda: fake)
            with _patched(S, env), _patched(cst, dict(env, embed_text=fake_embed)), \
                    _patched(S, {"embed_text": lambda t: np.ones((len(t), 4), dtype=np.float32)}):
                cfg.SEARCH_BACKEND, cfg.SEARCH_TEXT_COMPACT = "local", False
                S._df = None
                S.load_index()                                    # the default cache (ones)
                default_text = S._df["search_text"].tolist()

                cst.build(dry_run=False)
                side, meta_path = ct.sidecar_paths(csv)
                rows = pd.read_csv(side, keep_default_na=False, dtype={"parent_asin": str})
                assert rows["catalog_index"].tolist() == [1, 2]   # the two long rows only
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                assert meta["n_changed"] == 2 and meta["sidecar_md5"] == ct.file_md5(side)
                assert all(words(t) <= 12 for t in rows["embed_text"])

                cst.encode()
                assert [len(c) for c in calls] == [2]             # changed rows only
                default_cache = next(tmp.glob("embeddings_*_fake_model_*.npy"))
                cache = ct.compact_cache_path(default_cache)
                vectors = np.load(cache)
                assert (vectors[[1, 2]] == 2.0).all() and (vectors[[0, 3]] == 1.0).all()
                created = [p.name for p in tmp.iterdir()
                           if p.name not in (default_cache.name, "catalog.csv")]
                assert all(len(n) <= 45 for n in created), created   # Windows path limit
                # a run stopped inside its own cleanup left parts and a staging file:
                # the stamp-matching run removes them too (review round 3)
                left = tmp / f"_enc_{'f' * 12}_{cst.CHUNK}"
                left.mkdir()
                np.save(left / "p00000.npy", np.zeros((2, 4), dtype=np.float32))
                np.save(tmp / f"_enc_{'f' * 12}.w.npy", np.zeros((4, 4), dtype=np.float32))
                cst.encode()
                assert len(calls) == 1                            # stamp matches: no work
                assert not list(tmp.glob("_enc_*"))

                cfg.SEARCH_TEXT_COMPACT = True                     # in-process flip reloads
                S.load_index()
                assert S.search_text_mode() == f"compact_v{ct.COMPACT_VERSION}"
                text = S._df["search_text"].tolist()
                assert text[1:3] == rows["embed_text"].tolist() and text[0] == default_text[0]
                assert (S._embeddings[[1, 2]] == 2.0).all()
                # a flip reaches every reader, not only load_index (review round 2, C11)
                cfg.SEARCH_TEXT_COMPACT = False
                assert S.get_dataframe()["search_text"].tolist() == default_text
                assert S.search_text_mode() == "default"
                cfg.SEARCH_TEXT_COMPACT = True
                assert S.get_dataframe()["search_text"].tolist()[1:3] == rows["embed_text"].tolist()

                # a build whose sidecar cannot be replaced keeps the old meta (review C10)
                before = meta_path.read_bytes()

                def held_open(path, write):
                    raise SystemExit("held open")
                with _patched(cst, {"_write_atomic": held_open}):
                    try:
                        cst.build(dry_run=False)
                        raise AssertionError("build should have stopped")
                    except SystemExit:
                        pass
                assert meta_path.read_bytes() == before

                # a rebuilt sidecar (other text, same rows): old vectors are refused
                rows.loc[0, "embed_text"] = "changed text"
                rows.to_csv(side, index=False)
                meta["sidecar_md5"] = ct.file_md5(side)
                meta_path.write_text(json.dumps(meta), encoding="utf-8")
                S._df = None
                S.load_index()
                assert S.search_text_mode() == "default" and S._df["search_text"].tolist() == default_text

                # its encode starts a new parts folder; other runs' leftovers are deleted
                stale = tmp / f"_enc_{'0' * 12}_{cst.CHUNK}"
                stale.mkdir()
                np.save(stale / "p00000.npy", np.zeros((2, 4), dtype=np.float32))
                np.save(tmp / f"_enc_{'0' * 12}.w.npy", np.zeros((4, 4), dtype=np.float32))
                cst.encode()
                assert len(calls) == 2
                assert not list(tmp.glob("_enc_*"))
                S._df = None
                S.load_index()
                assert S.search_text_mode() != "default"
                assert S._df["search_text"].tolist()[1] == "changed text"
    finally:
        for k, v in saved.items():
            setattr(S, k, v)
        cfg.SEARCH_BACKEND, cfg.SEARCH_TEXT_COMPACT = old_backend, old_knob


def test_encode_never_resumes_another_models_parts():
    """Review round 3 (C9 behaviour): model A stops mid-encode; model B has the same
    tokenizer, so build writes a byte-identical sidecar. B must encode every chunk itself:
    a parts folder named by the sidecar alone would hand it A's vectors."""
    from scripts import compact_search_text as cst
    calls = []
    current = [_FakeEmbedder()]

    class _OtherModel(_FakeEmbedder):
        model_name = "fake/other"

    def embed_a(texts):
        calls.append(("a", len(texts)))
        if len(calls) > 1:
            raise RuntimeError("stopped")                  # the run dies after chunk 1
        return np.full((len(texts), 4), 2.0, dtype=np.float32)

    def embed_b(texts):
        calls.append(("b", len(texts)))
        return np.full((len(texts), 4), 3.0, dtype=np.float32)

    saved = {k: getattr(S, k) for k in ("_df", "_embeddings", "_loaded_backend", "_loaded_fields",
                                        "_embedded_fields", "_csv_hash_of", "_search_text_mode",
                                        "_loaded_compact")}
    old_backend, old_knob = cfg.SEARCH_BACKEND, cfg.SEARCH_TEXT_COMPACT
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            csv = tmp / "catalog.csv"
            _catalog().to_csv(csv, index=False)
            env = dict(PRODUCTS_CSV=csv, CACHE_DIR=tmp, get_default_embedder=lambda: current[0])
            with _patched(S, env), _patched(cst, dict(env, CHUNK=1, embed_text=embed_a)), \
                    _patched(S, {"embed_text": lambda t: np.ones((len(t), 4), dtype=np.float32)}):
                cfg.SEARCH_BACKEND, cfg.SEARCH_TEXT_COMPACT = "local", False
                S._df = None
                S.load_index()                                    # A's default cache
                cst.build(dry_run=False)
                side, _meta = ct.sidecar_paths(csv)
                md5_a = ct.file_md5(side)
                try:
                    cst.encode()
                    raise AssertionError("encode should have stopped")
                except RuntimeError:
                    pass
                assert calls == [("a", 1), ("a", 1)]
                assert len(list(tmp.glob("_enc_*"))) == 1         # A's parts, one chunk done

                current[0] = _OtherModel()
                S._df = None
                S.load_index()                                    # B's default cache
                cst.build(dry_run=False)
                assert ct.file_md5(side) == md5_a                 # byte-identical sidecar
                calls.clear()
                with _patched(cst, {"embed_text": embed_b}):
                    cst.encode()
                assert calls == [("b", 1), ("b", 1)], calls       # nothing resumed from A
                default_b = next(tmp.glob("embeddings_*_fake_other_*.npy"))
                vectors = np.load(ct.compact_cache_path(default_b))
                assert (vectors[[1, 2]] == 3.0).all() and (vectors[[0, 3]] == 1.0).all()
                assert not list(tmp.glob("_enc_*"))
    finally:
        for k, v in saved.items():
            setattr(S, k, v)
        cfg.SEARCH_BACKEND, cfg.SEARCH_TEXT_COMPACT = old_backend, old_knob


def test_encode_runs_are_keyed_by_model_and_replace_waits_for_a_reader():
    """Review round 2 (C9): another model with the same tokenizer writes a byte-identical
    sidecar, so the parts folder is keyed by the cache name (model, catalog, fields) and
    the length too. (C10): on Windows a file another process holds open cannot be
    replaced for a moment; the replace retries."""
    from scripts import compact_search_text as cst
    a = Path("embeddings_x_thenlper_gte-small_y_compact1.npy")
    b = Path("embeddings_x_BAAI_bge-small-en-v1.5_y_compact1.npy")
    assert cst._run_key("abc", a, 128) != cst._run_key("abc", b, 128)
    assert cst._run_key("abc", a, 128) != cst._run_key("abc", a, 256)
    assert cst._run_key("abc", a, 128) == cst._run_key("abc", a, 128)

    import os
    tries = []

    def flaky(src, dst):
        tries.append(1)
        if len(tries) < 3:
            raise PermissionError(13, "in use")
    with _patched(os, {"replace": flaky}):
        cst._replace(Path("a"), Path("b"), wait=0)
        assert len(tries) == 3
        tries.clear()
        try:
            cst._replace(Path("a"), Path("b"), tries=2, wait=0)
            raise AssertionError("should give up")
        except SystemExit as e:
            assert "another process has it open" in str(e)


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as e:  # noqa: BLE001 -- report every failure, then exit non-zero
            failed += 1
            print(f"FAIL  {name}: {type(e).__name__}: {str(e)[:500]}")
    print(f"\n{len(tests) - failed} passed, {failed} failed, 0 skipped")
    sys.exit(1 if failed else 0)
