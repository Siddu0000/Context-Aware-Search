"""Shared Streamlit rendering + fetch helpers, namespaced per search surface."""

import hashlib
import html
import math
import os
import re

import requests
import streamlit as st

import ui_cart

BACKEND = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
PRODUCT_URL = f"{BACKEND}/product"
FEEDBACK_URL = f"{BACKEND}/feedback"

_SURFACE_KEY = "_active_surface"


def set_surface(name: str) -> None:
    """Declare which surface is rendering; call once at the top of every page."""
    st.session_state[_SURFACE_KEY] = name


def surface() -> str:
    return st.session_state.get(_SURFACE_KEY, "ai")


def ns(key: str) -> str:
    """Namespaced key so two pages never fight over `view` / `detail_index` / `query`."""
    return f"{surface()}__{key}"


def sget(key: str, default=None):
    return st.session_state.get(ns(key), default)


def sset(key: str, value) -> None:
    st.session_state[ns(key)] = value


def sinit(defaults: dict) -> None:
    for k, v in defaults.items():
        st.session_state.setdefault(ns(k), v)


# Backend groups every bundle into one card per component; only wording differs
BUNDLE_UI = {
    "recipe": {
        "heading": "🧺 Shopping list",
        "blurb": "One card per ingredient — use the Option tabs to switch brands.",
        "slot_label": "🧺",
        "slot_word": "brands",
    },
    "outfit": {
        "heading": "👔 Complete the outfit",
        "blurb": "One card per piece — use the Option tabs to switch styles.",
        "slot_label": "👕",
        "slot_word": "options",
    },
    "setup": {
        "heading": "🖥️ Complete the setup",
        "blurb": "One card per component — use the Option tabs to switch models.",
        "slot_label": "🔌",
        "slot_word": "models",
    },
}


def bundle_ui(bundle_type):
    """Presentation strings for a bundle_type, or None for ordinary results."""
    return BUNDLE_UI.get(bundle_type or "")


# ---- display maps (UI constants, one place) -------------------------------------------
# The API's facets carry their own labels; these cover the product page (whose row has
# only the keys) and an API that sends a bare key. Internal values (other-apparel,
# weak-*-word) have no label and are never shown.
TYPE_LABELS = {
    "apparel": "Clothing", "footwear": "Shoes", "jewelry": "Jewelry",
    "bag": "Handbags & bags", "headwear": "Hats & hair accessories",
    "accessory": "Accessories",
}
GENDER_LABELS = {"women": "Women", "men": "Men", "kids": "Kids",
                 "unisex": "Unisex / unspecified"}
# = app/product_type.FINE_SUBTYPE_LABELS (2026-10-19), copied: the UI talks to the API
# over HTTP and imports nothing from app/ (tests/test_ui.py keeps the two equal)
FINE_SUBTYPE_LABELS = {
    "dresses": "Dresses", "tops_tees": "Tops & tees", "blouses_shirts": "Blouses & shirts",
    "sweaters": "Sweaters & cardigans", "hoodies": "Hoodies & sweatshirts",
    "coats_jackets": "Coats, jackets & vests", "suits_blazers": "Suits & blazers",
    "jeans": "Jeans", "pants": "Pants & trousers", "shorts": "Shorts", "leggings": "Leggings",
    "skirts": "Skirts", "jumpsuits_sets": "Jumpsuits, rompers & sets",
    "swim": "Swimwear & cover-ups", "sleep_lounge": "Sleep & lounge",
    "lingerie": "Bras, lingerie & underwear", "socks_hosiery": "Socks & hosiery",
    "ethnic_wear": "Ethnic & traditional wear", "sneakers": "Sneakers",
    "athletic": "Athletic shoes", "boots": "Boots", "sandals": "Sandals",
    "heels": "Heels & pumps", "flats": "Flats, loafers & oxfords", "slippers": "Slippers",
}
# The stored coarse subtype (app/product_type.py), for rows without a fine one
SUBTYPE_LABELS = {
    "dress": "Dresses", "top": "Tops", "bottom": "Bottoms", "outerwear": "Coats & jackets",
    "intimates/sleep/hosiery": "Lingerie, sleep & hosiery",
    "swim/coverup": "Swimwear & cover-ups", "jumpsuit/romper/set": "Jumpsuits, rompers & sets",
    "suit/tailoring": "Suits & tailoring",
}
# Colour family -> swatch. PRODUCT colours (data), not the page skin: a reskin leaves
# them alone. CSS reads them from this fixed map only, never from catalog text.
COLOR_HEX = {
    "black": "#111111", "blue": "#1F5FBF", "white": "#FFFFFF", "red": "#C40000",
    "grey": "#8C8C8C", "silver": "#C0C0C0", "pink": "#F4A7B9", "green": "#2E7D32",
    "gold": "#D4AF37", "brown": "#6D4C41", "beige": "#E8DCC4", "purple": "#6A1B9A",
    "navy": "#1B2A4A", "yellow": "#F7D600", "orange": "#F57C00", "burgundy": "#800020",
    "multicolor": "conic-gradient(#C40000, #F7D600, #2E7D32, #1F5FBF, #6A1B9A, #C40000)",
    "clear": "repeating-conic-gradient(#D5D9D9 0 25%, #FFFFFF 0 50%) 50% / 8px 8px",
}
COLOR_LABELS = {k: k.capitalize() for k in COLOR_HEX}
# The search scope: the catalog's three verticals (a client catalog changes this)
SCOPES = ("All", "Fashion", "Electronics", "Grocery")


def _label(mapping: dict, key, fallback=None) -> str:
    key = _text(key)
    return mapping.get(key) or mapping.get(key.casefold()) or fallback or key


# ---- the skin --------------------------------------------------------------------------
# EVERY page colour lives here. 2026-10-19: Sai asked for a large-marketplace look (the
# Oct 15 shell was deliberately neutral). 2026-10-26, Sai: "looks too much Amazon, use
# eBay color grading, don't change anything else": the PALETTE is now eBay-style, the
# layout, sizes and positions are unchanged. Values come from eBay's open-source design
# tokens (eBay Skin, MIT -- archived, continued as eBay/evo-web, MIT:
# https://github.com/eBay/evo-web/blob/36905785fec740849b8a9a68d03324a66fef23ca/packages/skin/dist/tokens/evo-light.css),
# "derived" = eBay's 4% hover layer over the base. Never eBay's (or any marketplace's)
# name, logo, four-colour wordmark, membership marks or proprietary font: the wordmark
# stays ours, in one colour. A client reskin -- or an older look -- is an edit of this
# one dict, plus the same values in .streamlit/config.toml and
# run_app.STREAMLIT_THEME_FLAGS (Streamlit's own widgets read the theme;
# tests/test_ui.py keeps the three in step).
SKIN = {
    "belt": "#FFFFFF",            # header rows: wordmark, nav, search band
    "belt-2": "#F7F7F7",          # sub-nav strip, footer, chat panel header
    "belt-fg": "#191919",         # text and outlines on belt, belt-2, backtotop, footer-2
    "belt-muted": "#707070",
    "search-btn": "#0968F6",      # eBay's accent blue (primary button)
    "search-btn-hover": "#0964EC",  # derived
    "search-btn-fg": "#FFFFFF",   # the magnifier on the search button
    "scope-bg": "#F7F7F7",
    "scope-border": "#8F8F8F",
    "search-border": "#363636",   # resting edge of the search control (white on white)
    "focus": "#005FCC",           # ring around the whole search control
    "cart-count": "#D50B0B",
    "page": "#FFFFFF",
    "page-grey": "#F7F7F7",       # cart page, behind white panels
    "text": "#191919",
    "text-2": "#707070",          # secondary: "Sponsored", "est.", captions
    "link": "#0968F6",
    "link-hover": "#707070",
    "query": "#191919",           # the query in the results bar (bold)
    "border": "#8F8F8F",
    "divider": "#E5E5E5",
    "img-bg": "#F7F7F7",          # white product shots blend in (mix-blend-mode)
    "star": "#191919",
    "star-off": "#C7C7C7",
    "cart-btn": "#FFFFFF",        # Add to cart: eBay's outlined (secondary) button
    "cart-btn-border": "#0968F6",
    "cart-btn-hover": "#F7F7F7",
    "cart-btn-active": "#F5F9FF",
    "cart-btn-fg": "#0968F6",
    "buy-btn": "#0968F6",         # Buy Now: eBay's filled (primary) button
    "buy-btn-border": "#0968F6",
    "buy-btn-hover": "#0964EC",   # derived
    "buy-btn-fg": "#FFFFFF",
    "pill-bg": "#F7F7F7",         # sort pill, quantity select, disabled checkout
    "pill-hover": "#F5F5F5",
    "disabled-text": "#8F8F8F",   # pager Previous / Next when inactive (eBay's border-medium:
                                  # eBay's disabled #C7C7C7, 1.69:1, all but vanished)
    "backtotop": "#E5E5E5",
    "backtotop-hover": "#DCDCDC",  # derived
    "footer-fg": "#707070",
    "footer-2": "#FFFFFF",
    "swatch-border": "#8F8F8F",
    "bubble-assistant": "#F7F7F7",
    "bubble-user": "#D4E5FE",
    "avatar-user": "#0968F6",     # the Assistant's user avatar (else Streamlit's red); the
                                  # bot avatar is the theme's orangeColor (= star)
    "tab-active": "#191919",
    "notice-bg": "#F7F7F7",       # the weak-match banner
    "shadow-bar": "rgba(0,0,0,.15)",
    "shadow-pill": "rgba(0,0,0,0)",  # eBay's buttons and selects carry no drop shadow
}

# Layout and type in px on top of Streamlit 1.58's DOM. Selectors on data-testid are
# Streamlit internals (pinned to 1.58 in requirements.txt): a rename only loses styling,
# never function. st-key-* classes are the documented result of `key=`, so they are
# preferred. Only var(--cas-*) colours below: the values are SKIN's.
_SHOP_CSS = """<style>
.stApp{__SKIN__;--cas-gutter:24px;
  font-family:Arial,"Helvetica Neue",Helvetica,sans-serif;background:var(--cas-page)}
@media (max-width:1023px){.stApp{--cas-gutter:20px}}
@media (max-width:639px){.stApp{--cas-gutter:16px}}
.cas-sr,.st-key-cas_searchrow [data-testid="stBaseButton-primaryFormSubmit"] p{
  position:absolute !important;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0);
  white-space:nowrap}
/* row 1: Streamlit's own fixed 60px header holds the wordmark and the page links */
header[data-testid="stHeader"]{background:var(--cas-belt)}
header[data-testid="stHeader"] a,header[data-testid="stHeader"] a span,
header[data-testid="stHeader"] a p,header[data-testid="stHeader"] button{color:var(--cas-belt-fg) !important}
/* on a narrow screen the page links fold into the sidebar: its opener must show */
header[data-testid="stHeader"] [data-testid="stExpandSidebarButton"] *{color:var(--cas-belt-fg) !important}
[data-testid="stTopNavLink"]{border:1px solid transparent;border-radius:2px}
[data-testid="stTopNavLink"]:hover{border-color:var(--cas-belt-fg)}
[data-testid="stTopNavLink"] p,[data-testid="stTopNavLink"] span{font-weight:700;font-size:14px}
[data-testid="stTopNavLink"][aria-current="page"]{border-color:var(--cas-belt-fg)}
/* the page starts right under the header; gutters 24/20/16px (phone: no side scroll) */
[data-testid="stMainBlockContainer"]{padding:60px var(--cas-gutter) 0 !important;max-width:none}
/* row 2: the search band, full-bleed (the negative margin is exactly the gutter, so no
   horizontal overflow). Sticky on desktop only: on a phone it wraps to two rows */
.st-key-cas_masthead{background:var(--cas-belt);margin:0 calc(-1*var(--cas-gutter));
  width:calc(100% + 2*var(--cas-gutter)) !important;max-width:none;
  padding:8px var(--cas-gutter);gap:12px;flex-wrap:nowrap}
@media (min-width:640px){
  [data-testid="stLayoutWrapper"]:has(> .st-key-cas_masthead){position:sticky;top:60px;z-index:99}
  /* a hairline (a shadow, so no layout effect) where the white band meets white content */
  .st-key-cas_masthead{box-shadow:0 1px 0 var(--cas-divider)}
}
@media (max-width:639px){
  .st-key-cas_masthead{flex-wrap:wrap !important}
  .st-key-cas_masthead > :first-child{flex:1 1 100% !important}
  /* on a phone the band scrolls away and the fixed header is the edge content slides under */
  header[data-testid="stHeader"]{box-shadow:0 1px 0 var(--cas-divider)}
}
.cas-masthead-title{color:var(--cas-belt-fg);font-size:18px;font-weight:700;line-height:40px}
.st-key-cas_masthead [data-testid="stForm"]{padding:0;border:0}
/* one joined control: scope | text | button, with a ring around all three on focus */
.st-key-cas_searchrow{flex-wrap:nowrap !important;gap:0 !important;border-radius:4px;
  background:var(--cas-page);align-items:stretch !important;box-shadow:0 0 0 2px var(--cas-search-border)}
.st-key-cas_searchrow:focus-within{box-shadow:0 0 0 3px var(--cas-focus)}
.st-key-cas_searchrow [data-testid="stTextInputRootElement"],
.st-key-cas_searchrow [data-baseweb="input"],.st-key-cas_searchrow [data-baseweb="base-input"]{
  border:0 !important;border-radius:0 !important;background:var(--cas-page) !important;
  min-height:40px;height:40px}
.st-key-cas_searchrow input{font-size:15px;color:var(--cas-text)}
.st-key-cas_searchrow [data-testid="InputInstructions"]{display:none}
.st-key-cas_searchrow [data-testid="stSelectbox"] [data-baseweb="select"] > div{
  background:var(--cas-scope-bg);border:0;border-right:1px solid var(--cas-scope-border);
  border-radius:4px 0 0 4px;min-height:40px;height:40px;font-size:12px;align-items:center}
@media (max-width:479px){.st-key-cas_searchrow > :has([data-testid="stSelectbox"]){display:none}}
.st-key-cas_searchrow [data-testid="stBaseButton-primaryFormSubmit"]{background:var(--cas-search-btn);
  border:0;border-radius:0 4px 4px 0;height:40px;min-height:40px;width:48px;min-width:48px;
  color:var(--cas-search-btn-fg);padding:0}
.st-key-cas_searchrow [data-testid="stBaseButton-primaryFormSubmit"]:hover{background:var(--cas-search-btn-hover)}
.st-key-cas_searchrow [data-testid="stBaseButton-primaryFormSubmit"] [data-testid="stIconMaterial"]{
  color:var(--cas-search-btn-fg);font-size:24px;margin:0}
/* Settings and Cart: belt-fg text on the belt, a 1px outline on hover */
.st-key-cas_masthead [data-testid="stPopoverButton"],
.st-key-cas_masthead [data-testid="stBaseButton-secondary"]{background:transparent;
  border:1px solid transparent;color:var(--cas-belt-fg);box-shadow:none}
.st-key-cas_masthead [data-testid="stPopoverButton"]:hover,
.st-key-cas_masthead [data-testid="stBaseButton-secondary"]:hover{border-color:var(--cas-belt-fg);
  color:var(--cas-belt-fg)}
.st-key-cas_masthead [data-testid="stPopoverButton"] *,
.st-key-cas_masthead [data-testid="stBaseButton-secondary"] *{color:var(--cas-belt-fg)}
.st-key-cas_masthead :is([data-testid^="stBaseButton"],[data-testid="stPopoverButton"]) p{
  white-space:nowrap;font-weight:700}
[class*="_topbar_cart"] strong{color:var(--cas-cart-count) !important;font-size:16px}
/* row 3: shortcuts; the strip scrolls sideways on a phone, never the page */
[data-testid="stLayoutWrapper"]:has(> .st-key-cas_subnav),
[data-testid="stLayoutWrapper"]:has(> .st-key-cas_resultsbar){margin-top:-16px}
.st-key-cas_subnav{background:var(--cas-belt-2);margin:0 calc(-1*var(--cas-gutter));
  width:calc(100% + 2*var(--cas-gutter)) !important;max-width:none;
  padding:3px var(--cas-gutter);gap:2px !important;flex-wrap:nowrap !important;overflow-x:auto}
.st-key-cas_subnav button{background:transparent !important;border:1px solid transparent !important;
  border-radius:2px;min-height:32px;padding:2px 9px;box-shadow:none}
.st-key-cas_subnav button:hover{border-color:var(--cas-belt-fg) !important}
.st-key-cas_subnav button[data-testid="stBaseButton-secondary"]{border-color:var(--cas-belt-fg) !important}
.st-key-cas_subnav button p{color:var(--cas-belt-fg) !important;font-size:14px;white-space:nowrap}
/* results bar: "1–12 of N results for “q”" + Sort, white, full-bleed, a soft shadow */
.st-key-cas_resultsbar{background:var(--cas-page);margin:0 calc(-1*var(--cas-gutter)) 4px;
  width:calc(100% + 2*var(--cas-gutter)) !important;max-width:none;
  padding:6px var(--cas-gutter);box-shadow:0 2px 5px 0 var(--cas-shadow-bar);flex-wrap:wrap}
.st-key-cas_resultsbar [data-testid="stSelectbox"] [data-baseweb="select"] > div{
  background:var(--cas-pill-bg);border:1px solid var(--cas-border);border-radius:8px;
  box-shadow:0 2px 5px 0 var(--cas-shadow-pill);min-height:32px;font-size:13px}
.cas-results-head{font-size:14px;line-height:20px;color:var(--cas-text)}
.cas-results-head .q{color:var(--cas-query);font-weight:700}
.cas-results-head .sub{color:var(--cas-text-2)}
.cas-h2{font-size:21px;line-height:27px;font-weight:700;color:var(--cas-text);margin:0}
.cas-notice{border:1px solid var(--cas-border);border-left:4px solid var(--cas-link);
  background:var(--cas-notice-bg);border-radius:8px;padding:10px 14px;font-size:14px;color:var(--cas-text)}
.st-key-cas_intents [data-testid="stText"]{font-size:13px;padding:2px 10px;border:1px solid var(--cas-border);
  border-radius:8px;background:var(--cas-page);color:var(--cas-text)}
/* left rail: links (tertiary buttons) and checkboxes, no box on desktop; on a phone the
   expander is real and starts collapsed */
.st-key-cas_filters details{border:none}
@media (min-width:768px){
  .st-key-cas_filters > [data-testid="stExpander"] > details > summary{display:none}
  .st-key-cas_filters > [data-testid="stExpander"] > details > [data-testid="stExpanderDetails"]{
    border:0;padding:0}
}
.st-key-cas_rail [data-testid="stVerticalBlock"]{gap:2px}
.cas-rail-h{font-size:14px;line-height:20px;font-weight:700;color:var(--cas-text);margin:12px 0 2px}
.cas-rail-h .hint{color:var(--cas-text-2);font-weight:400;cursor:help}
/* a section's fixed note under its heading (Dietary's match-all rule), as quiet as a hint */
.cas-rail-note{font-size:12px;line-height:16px;color:var(--cas-text-2);margin:0 0 2px}
.st-key-cas_rail [data-testid="stBaseButton-tertiary"]{min-height:0;padding:0;
  justify-content:flex-start;text-align:left}
.st-key-cas_rail [data-testid="stBaseButton-tertiary"] p{font-size:14px;line-height:22px;
  color:var(--cas-text);text-align:left}
.st-key-cas_rail [data-testid="stBaseButton-tertiary"]:hover p{color:var(--cas-link-hover)}
[class*="rail_l2"]{padding-left:12px}
.st-key-cas_rail [data-testid="stCheckbox"] p{font-size:14px;line-height:22px}
.st-key-cas_rail [data-testid="stCaptionContainer"] p{font-size:12px;line-height:16px}
[class*="__f_color_"] [data-testid="stCheckbox"] p::before{content:"";display:inline-block;
  width:14px;height:14px;border:1px solid var(--cas-swatch-border);border-radius:3px;
  margin-right:6px;vertical-align:-2px;background:var(--sw,transparent)}
.st-key-cas_rail [data-testid="stExpander"] summary p{font-size:13px;color:var(--cas-link)}
/* product cards: no box, image on a light square, brand, a 2-line title link */
[class*="st-key-card_"]{border-color:transparent !important;background:var(--cas-page);
  padding:0 !important;gap:2px !important}
[class*="st-key-card_"]{scroll-margin-top:72px}
@media (min-width:640px){[class*="st-key-card_"]{scroll-margin-top:132px}}
/* the product page's suggestions: "Related items" lands here, below the sticky band
   (the cards' margins: 60px header + 56px band + 16px) */
.st-key-cas_related{scroll-margin-top:72px}
@media (min-width:640px){.st-key-cas_related{scroll-margin-top:132px}}
/* "Related items": a neutral outlined pill under Add to cart, in the same place on every
   card, so it reads as a button without competing with the blue Add to cart */
[class*="st-key-related_"]{margin-top:4px}
[class*="st-key-related_"] button{background:var(--cas-page) !important;
  border:1px solid var(--cas-border) !important;border-radius:100px;min-height:32px;
  padding:4px 14px;box-shadow:none}
[class*="st-key-related_"] button:hover{background:var(--cas-pill-hover) !important;
  border-color:var(--cas-text) !important}
[class*="st-key-related_"] button *{color:var(--cas-text) !important;font-size:13px}
[class*="st-key-related_"] button:focus-visible{outline:2px solid var(--cas-link)}
.cas-img{position:relative;width:100%;aspect-ratio:1/1;background:var(--cas-img-bg);overflow:hidden}
.cas-img img{position:absolute;inset:10px;width:calc(100% - 20px);height:calc(100% - 20px);
  object-fit:contain;mix-blend-mode:multiply}
.cas-img.empty{display:flex;align-items:center;justify-content:center}
.cas-img.empty::after{content:"No image";font-size:12px;color:var(--cas-text-2)}
.cas-spons{font-size:12px;line-height:16px;color:var(--cas-text-2);min-height:16px;margin-top:6px}
.cas-brand{font-size:14px;line-height:20px;font-weight:700;color:var(--cas-text);
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;min-height:20px}
.cas-title{font-size:15px;line-height:21px;color:var(--cas-text);display:-webkit-box;
  -webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden;min-height:42px;overflow-wrap:anywhere}
:is([class*="st-key-grid_"],[class*="st-key-d_cross_"],[class*="st-key-d_up_"]) button{
  padding:0;min-height:42px;justify-content:flex-start;align-items:flex-start;text-align:left;width:100%}
:is([class*="st-key-grid_"],[class*="st-key-d_cross_"],[class*="st-key-d_up_"]) button p{
  display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden;
  text-align:left;color:var(--cas-text);font-size:15px;line-height:21px;overflow-wrap:anywhere}
:is([class*="st-key-grid_"],[class*="st-key-d_cross_"],[class*="st-key-d_up_"]) button:hover p{color:var(--cas-link-hover)}
:is([class*="st-key-grid_"],[class*="st-key-d_cross_"],[class*="st-key-d_up_"]) button:focus-visible{
  outline:2px solid var(--cas-link)}
[class*="st-key-d_cross_"] button p,[class*="st-key-d_up_"] button p{font-size:13px;line-height:19px}
.cas-rating{display:flex;align-items:center;gap:5px;font-size:14px;min-height:22px;color:var(--cas-text)}
.cas-stars{font-size:15px;letter-spacing:1px;background:linear-gradient(90deg,var(--cas-star) var(--pct),var(--cas-star-off) var(--pct));
  -webkit-background-clip:text;background-clip:text;color:transparent}
.cas-rcount{color:var(--cas-link)}
.cas-price{min-height:30px;margin-top:2px;color:var(--cas-text);line-height:1.25}
.cas-price .sym,.cas-price .frac{font-size:13px;position:relative;top:-.55em}
.cas-price .whole{font-size:21px}
.cas-price.big .whole{font-size:28px}
.cas-price.estonly{font-size:15px;color:var(--cas-text-2);padding-top:5px}
.cas-price.estonly.big{font-size:18px}
.cas-price .est{font-size:11px;border:1px solid var(--cas-border);border-radius:4px;padding:0 4px;
  margin-left:6px;vertical-align:1px;color:var(--cas-text-2)}
.cas-price.none{font-size:13px;color:var(--cas-text-2);padding-top:6px}
.cas-meta{font-size:12px;line-height:16px;color:var(--cas-text-2)}
.cas-extras{min-height:22px;display:flex;flex-wrap:wrap;align-items:center;gap:4px 8px;font-size:12px;
  color:var(--cas-text-2)}
.cas-badge{display:inline-block;font-size:12px;line-height:16px;color:var(--cas-link);
  border:1px solid var(--cas-link);border-radius:4px;padding:1px 6px;background:var(--cas-page)}
.cas-vsw{display:inline-block;width:12px;height:12px;border-radius:50%;
  border:1px solid var(--cas-swatch-border);background:var(--cas-page)}
.cas-variants{display:inline-flex;align-items:center;gap:3px}
.cas-card.compact .cas-title{font-size:13px;line-height:19px;min-height:38px}
.cas-card.compact .cas-price .whole{font-size:18px}
/* Add to cart = an outlined pill, Buy Now = a filled one (colours by key class, so the
   pair stays distinct whatever the theme's primary colour is) */
[class*="st-key-cart_"]:not([class*="_topbar_cart"]) button{background:var(--cas-cart-btn) !important;
  border:1px solid var(--cas-cart-btn-border) !important;border-radius:100px;min-height:32px;
  padding:4px 14px;box-shadow:0 2px 5px 0 var(--cas-shadow-pill)}
[class*="st-key-cart_"]:not([class*="_topbar_cart"]) button:hover{background:var(--cas-cart-btn-hover) !important}
[class*="st-key-cart_"]:not([class*="_topbar_cart"]) button:active{background:var(--cas-cart-btn-active) !important}
[class*="st-key-cart_"]:not([class*="_topbar_cart"]) button *{color:var(--cas-cart-btn-fg) !important;font-size:13px}
[class*="st-key-buynow_"] button *{color:var(--cas-buy-btn-fg) !important;font-size:13px}
[class*="st-key-buynow_"] button{background:var(--cas-buy-btn) !important;
  border:1px solid var(--cas-buy-btn-border) !important;border-radius:100px;min-height:32px}
[class*="st-key-buynow_"] button:hover{background:var(--cas-buy-btn-hover) !important}
/* the grid: 4 per row; 2 per row on a phone */
@media (max-width:639px){
  [class*="st-key-cas_grid"] [data-testid="stHorizontalBlock"]{flex-wrap:wrap !important;gap:8px}
  [class*="st-key-cas_grid"] [data-testid="stColumn"]{min-width:calc(50% - 4px) !important;
    max-width:calc(50% - 4px);flex:1 1 calc(50% - 4px) !important}
}
/* pager: one joined strip, the current page bold with a dark border */
[class*="__pager"]{gap:0 !important;margin-top:20px}
[class*="__pager"] button{border-radius:0;border:1px solid var(--cas-border);margin-left:-1px;
  background:var(--cas-page);min-height:38px;padding:6px 14px;box-shadow:none}
[class*="__pager"] > :first-child button{border-radius:8px 0 0 8px}
[class*="__pager"] > :last-child button{border-radius:0 8px 8px 0}
[class*="__pager"] > *{flex:0 0 auto !important;width:auto !important}
[class*="__pager"] button p,[class*="__pager"] button span{color:var(--cas-text);font-size:14px;white-space:nowrap}
[class*="__pager"] button:hover{background:var(--cas-pill-hover)}
[class*="__pager"] button:disabled p,[class*="__pager"] button:disabled span{color:var(--cas-disabled-text)}
[class*="__pager"] button[data-testid="stBaseButton-primary"]:disabled{background:var(--cas-page);
  border-color:var(--cas-text);position:relative;z-index:1}
[class*="__pager"] button[data-testid="stBaseButton-primary"]:disabled p{color:var(--cas-text);font-weight:700}
/* product page */
.cas-crumbs{font-size:12px;color:var(--cas-text-2)}
.cas-detail-img{position:relative;width:100%;aspect-ratio:1/1;background:var(--cas-page);overflow:hidden}
.cas-detail-img img{position:absolute;inset:0;width:100%;height:100%;object-fit:contain}
.cas-detail-img.empty{display:flex;align-items:center;justify-content:center;background:var(--cas-img-bg)}
.cas-detail-img.empty::after{content:"No image";color:var(--cas-text-2)}
.cas-detail .cas-spons{margin-top:0}
.cas-detail-title{font-size:24px;line-height:32px;font-weight:400;color:var(--cas-text);margin:0 0 4px;
  overflow-wrap:anywhere}
.cas-detail-brand{font-size:14px;color:var(--cas-link);margin-bottom:2px}
.cas-detail hr,.cas-hr{border:0;border-top:1px solid var(--cas-divider);margin:10px 0}
.cas-h3{font-size:16px;font-weight:700;color:var(--cas-text);margin:12px 0 6px}
.cas-attrs{border-collapse:collapse;font-size:14px;color:var(--cas-text)}
.cas-attrs th{text-align:left;font-weight:700;padding:3px 24px 3px 0;vertical-align:top;white-space:nowrap;border:0}
.cas-attrs td{padding:3px 0;border:0;overflow-wrap:anywhere}
.cas-attrs tr{border:0}
.cas-about{font-size:14px;line-height:20px;color:var(--cas-text)}
.cas-about ul{margin:0;padding-left:18px}
.st-key-cas_buybox{border-radius:8px !important;padding:14px !important;border-color:var(--cas-border) !important}
.st-key-cas_buybox [data-testid="stSelectbox"] [data-baseweb="select"] > div{background:var(--cas-pill-bg);
  border-radius:8px;box-shadow:0 2px 5px 0 var(--cas-shadow-pill);min-height:32px;font-size:13px}
.st-key-cas_buybox [class*="st-key-cart_"] button,.st-key-cas_buybox [class*="st-key-buynow_"] button{width:100%}
.cas-incart{font-size:13px;color:var(--cas-text)}
[class*="st-key-cas_carousel_"]{flex-wrap:nowrap !important;overflow-x:auto;scroll-snap-type:x mandatory;
  padding-bottom:8px;gap:16px !important}
[class*="st-key-cas_carousel_"] > *{flex:0 0 200px !important;scroll-snap-align:start}
/* cart page: white panels on grey */
.stApp:has(.st-key-cas_cartpage){background:var(--cas-page-grey) !important}
.st-key-cas_cartlist,.st-key-cas_cartsub{background:var(--cas-page);border-radius:8px;padding:16px 20px;
  box-shadow:0 0 0 1px var(--cas-divider)}
.cas-cart-headrow{display:flex;justify-content:space-between;align-items:flex-end;
  border-bottom:1px solid var(--cas-divider);padding-bottom:6px}
.cas-cart-headrow .p{font-size:14px;color:var(--cas-text-2)}
.cas-cart-row{display:flex;gap:16px;align-items:flex-start}
.cas-cart-row .cas-img{width:120px;min-width:120px}
.cas-cart-main{flex:1;min-width:0}
.cas-cart-row .cas-title{-webkit-line-clamp:2;min-height:0;font-size:16px;line-height:22px}
.cas-cart-price{font-size:15px;font-weight:700;color:var(--cas-text);text-align:right;white-space:nowrap}
.cas-cart-price .est{font-size:11px;font-weight:400;border:1px solid var(--cas-border);border-radius:4px;
  padding:0 4px;margin-left:6px;color:var(--cas-text-2)}
.cas-subtotal{font-size:18px;line-height:24px;color:var(--cas-text)}
.cas-subtotal b{font-weight:700}
[class*="st-key-card_cart_"]{border-bottom:1px solid var(--cas-divider) !important;padding-bottom:12px !important}
[class*="st-key-cas_qty_"]{border:2px solid var(--cas-border);border-radius:100px;width:fit-content !important;
  gap:0 !important;align-items:center;padding:0 2px;flex-wrap:nowrap !important}
[class*="st-key-cas_qty_"] button{border:0 !important;background:transparent !important;min-height:28px;
  padding:0 10px;box-shadow:none}
.cas-qty-n{font-size:14px;font-weight:700;padding:0 6px;color:var(--cas-text)}
.st-key-cas_cartsub [data-testid="stBaseButton-secondary"]:disabled{background:var(--cas-pill-bg);
  color:var(--cas-text-2);border-color:var(--cas-border)}
/* Shopping Assistant: bubbles and a panel header */
[data-testid="stChatMessage"]{background:var(--cas-bubble-assistant);border-radius:8px;padding:8px 12px}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]){background:var(--cas-bubble-user)}
[data-testid="stChatMessage"] [data-testid="stChatMessageAvatarUser"]{background-color:var(--cas-avatar-user)}
[data-testid="stChatMessage"] p{font-size:14px}
.st-key-cas_chat_panel{border-radius:8px !important;border-color:var(--cas-border) !important}
.st-key-cas_chat_head{background:var(--cas-belt-2);border-radius:6px;padding:4px 12px}
.st-key-cas_chat_head p,.st-key-cas_chat_head .cas-panel-title{color:var(--cas-belt-fg) !important}
.cas-panel-title{font-size:15px;font-weight:700;line-height:32px}
[data-baseweb="tab-highlight"]{background-color:var(--cas-tab-active) !important}
/* footer: Back to top, three columns, the licence strip -- all full-bleed */
.st-key-cas_footer{background:var(--cas-belt-2);margin:32px calc(-1*var(--cas-gutter)) 0;
  width:calc(100% + 2*var(--cas-gutter)) !important;max-width:none;gap:0 !important}
.st-key-cas_back_to_top button{width:100%;background:var(--cas-backtotop) !important;border:0 !important;
  border-radius:0;min-height:48px}
.st-key-cas_back_to_top button:hover{background:var(--cas-backtotop-hover) !important}
.st-key-cas_back_to_top button p{color:var(--cas-belt-fg) !important;font-size:13px}
.st-key-cas_footcols{padding:24px var(--cas-gutter) 16px}
.st-key-cas_footcols [data-testid="stVerticalBlock"]{gap:6px}
.cas-foot h4{color:var(--cas-belt-fg);margin:0 0 8px;font-size:16px;padding:0}
.cas-foot p{color:var(--cas-footer-fg);margin:0;font-size:13px;line-height:20px}
.st-key-cas_footcols [data-testid="stBaseButton-tertiary"]{min-height:0;padding:0}
.st-key-cas_footcols [data-testid="stBaseButton-tertiary"] p{color:var(--cas-footer-fg);font-size:13px}
.st-key-cas_footcols [data-testid="stBaseButton-tertiary"]:hover p{text-decoration:underline}
.cas-footer{background:var(--cas-footer-2);color:var(--cas-belt-muted);font-size:12px;line-height:18px;
  display:flex;flex-wrap:wrap;align-items:center;gap:6px 16px;padding:12px var(--cas-gutter)}
.cas-footer .mark{color:var(--cas-belt-fg);font-size:13px}
</style>"""


def _skin_vars() -> str:
    return ";".join(f"--cas-{name}:{value}" for name, value in SKIN.items())


def inject_css():
    # st.html, not st.markdown: style-only HTML goes to the event container, so it adds
    # no element (and no 16px flex gap) to the top of the page
    st.html(_SHOP_CSS.replace("__SKIN__", _skin_vars()))


def is_nan(x) -> bool:
    """True if x is float NaN. None and other types return False."""
    try:
        return isinstance(x, float) and math.isnan(x)
    except (TypeError, ValueError):
        return False


def _num(x):
    """A finite float, or None for None / NaN / anything that is not a number."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) or math.isinf(v) else v


def _text(x) -> str:
    """Catalog/model text as one line: None and NaN (pandas' missing value) become ''."""
    if x is None or is_nan(x):
        return ""
    return " ".join(str(x).split())


def price_label(product: dict) -> str | None:
    """'$X' for a listed price, '≈ $X (est.)' for an estimate, None if neither.

    Estimates come from similar products (scripts/enrich_catalog.py); they are
    always marked so a shopper never mistakes one for the listed price.
    """
    p = _num(product.get("price"))
    if p is not None:
        return f"${p:,.2f}"
    e = _num(product.get("price_estimate"))
    if e is not None:
        return f"≈ ${e:,.2f} (est.)"
    return None


def _dev_details() -> bool:
    """Whether to show raw pipeline scores; each page's settings menu sets this."""
    return bool(st.session_state.get("show_dev_details"))


# Pipeline-state notes hidden from users; backend strings unchanged (evals match)
_REASON_TRANSLATIONS = {
    "(beyond reranked pool — embedding-similarity order)": "",
    "(rerank disabled)": "",
    "(rerank unavailable — embedding score only)": "",
}


def friendly_reason(reason: str) -> str:
    """User-facing version of a product's `reason`. Empty string = don't show."""
    if not reason:
        return ""
    return _REASON_TRANSLATIONS.get(str(reason).strip(), str(reason))


# Every char that can open markdown structure inline: links/images ([ ] ( ) !),
# autolinks and HTML (< >), emphasis (* _ ~), code (`), entities (&: "&#58;" is a
# ":"), Streamlit's LaTeX ($) and :color[...] directives (:), tables (|), headings
# (#), and the GFM bare-link literals Streamlit autolinks ("www.", "https://",
# e-mail: . / @). Each renders as itself once backslash-escaped (CommonMark)
_MD_SPECIAL = re.compile(r"([\\`*_{}\[\]()<>#!|~$&:./@])")


def escape_md(text) -> str:
    """Model-, shopper- or seller-written text as LITERAL markdown: nothing in it can
    render as a link, image, entity or emphasis. Whitespace runs (newlines included)
    become one space, so no heading, list or link definition can start a line.

    The backend screens what models write (app/query_guard.py); this is the UI's own
    half, and it also covers the shopper's own words on the fallback / guarded paths
    and catalog text (titles, stores, categories are written by sellers).
    """
    return _MD_SPECIAL.sub(r"\\\1", " ".join(str(text or "").split()))


def esc(text) -> str:
    """Any text as inert HTML for st.html. st.html parses no markdown and no LaTeX, so
    escaping the five HTML specials is the whole job (DOMPurify sanitises as well)."""
    return html.escape(_text(text), quote=True)


# Whitespace, quotes, angle brackets, backticks, backslashes and control characters
# have no place in a catalog image URL; any of them means a value that could break out
# of the src attribute, so the image is dropped rather than repaired
_URL_UNSAFE = re.compile(r"[\s\"'<>`\\\x00-\x1f\x7f]")


def safe_img(url) -> str:
    """An image URL safe to put in a src attribute, or '' (https only)."""
    if not isinstance(url, str):
        return ""
    u = url.strip()
    if not u.lower().startswith("https://") or _URL_UNSAFE.search(u):
        return ""
    return html.escape(u, quote=True)


def stars_html(rating, count, *, detail: bool = False) -> str:
    """'4.6 ★★★★★ (28)': the number, five stars filled to the exact rating, the review
    count. Pure. The stars are decorative (their contrast is low), so the number is
    always printed.

    The row is always emitted (empty when there is no rating) so every card in a grid
    row keeps the same height."""
    r, n = _num(rating), _num(count)
    if r is None or not 0 <= r <= 5:
        if n is not None and n > 0:
            word = "rating" if int(n) == 1 else "ratings"
            return f'<div class="cas-rating"><span>{int(n):,} {word}</span></div>'
        return '<div class="cas-rating"></div>'
    label = f"{r:.1f} out of 5 stars"
    count_html = ""
    if n is not None and n >= 0:
        text = (f"{int(n):,} rating{'' if int(n) == 1 else 's'}" if detail
                else f"({int(n):,})")
        count_html = f'<span class="cas-rcount">{text}</span>'
    return (
        f'<div class="cas-rating" title="{label}">'
        f'<span class="cas-rnum">{r:.1f}</span>'
        f'<span class="cas-stars" style="--pct:{r / 5 * 100:.0f}%" aria-hidden="true">'
        f"★★★★★</span>{count_html}</div>"
    )


def price_html(product: dict, *, detail: bool = False) -> str:
    """The price block: a listed price with superscript dollars and cents, an estimate
    with an 'est.' badge, or a note. Pure.

    Same semantics as price_label: an estimate is never shown as a plain price, and it
    is set in grey WITHOUT the superscript, so it never looks like a listed price."""
    big = " big" if detail else ""
    p = _num(product.get("price"))
    if p is not None:
        whole, frac = f"{p:,.2f}".split(".")
        return (
            f'<div class="cas-price{big}"><span class="sym" aria-hidden="true">$</span>'
            f'<span class="whole" aria-hidden="true">{whole}</span>'
            f'<span class="frac" aria-hidden="true">{frac}</span>'
            f'<span class="cas-sr">${p:,.2f}</span></div>'
        )
    e = _num(product.get("price_estimate"))
    if e is not None:
        note = "No listed price: estimated from similar products"
        badge = f'<span class="est" title="{note}">est.</span>'
        tail = (f'<div class="cas-meta">{note}. Never counted in the cart total.</div>'
                if detail else "")
        return f'<div class="cas-price estonly{big}">≈ ${e:,.2f}{badge}</div>{tail}'
    return '<div class="cas-price none">Price not listed</div>'


# "Highly rated" is the one merchandising badge we can back with data: a high average
# from enough ratings that it is not noise (54.5% of rows have fewer than 5 ratings)
HIGHLY_RATED_MIN = 4.5
HIGHLY_RATED_COUNT = 50


def badges_html(product: dict) -> str:
    """Data-backed badges only (ui_spec section 9); '' when the row earns none. Pure."""
    r, n = _num(product.get("average_rating")), _num(product.get("rating_number"))
    if r is not None and n is not None and r >= HIGHLY_RATED_MIN and n >= HIGHLY_RATED_COUNT:
        return ('<span class="cas-badge" title="4.5+ average from 50+ ratings">'
                "Highly rated</span>")
    return ""


def _swatch(family) -> str:
    """A colour dot from the FIXED family map ('' for an unknown family): no catalog
    text reaches the style attribute."""
    value = COLOR_HEX.get(_text(family).casefold())
    return f'<span class="cas-vsw" style="background:{value}"></span>' if value else ""


def _other_variants(product: dict) -> list:
    """The variants collapsed into this card, minus the card's own row."""
    own = product.get("catalog_index")
    return [v for v in (product.get("variants") or [])
            if isinstance(v, dict) and v.get("catalog_index") != own]


def variants_html(product: dict) -> str:
    """'● ● ● +3 colours' when the API collapsed colour/size variants into this card
    (VARIANT_COLLAPSE). Pure. 'colours' only counts OTHER colours: a size variant of the
    same colour is an 'option', never a colour we do not have."""
    others = _other_variants(product)
    if not others:
        # n_variants counts the card's own row too (the API's `variants` lists it first)
        n = int(_num(product.get("n_variants")) or 0) - 1
        return (f'<span class="cas-variants">+{n} option{"s" if n != 1 else ""}</span>'
                if n > 0 else "")
    own = _text(product.get("color")).casefold()
    colours, dots = [], []
    for v in others:
        name = _text(v.get("color")).casefold() or _text(v.get("color_family")).casefold()
        if name and name != own and name not in colours:
            colours.append(name)
            dots.append(_swatch(v.get("color_family")))
    if colours:
        n = len(colours)
        label = f"+{n} colour{'s' if n != 1 else ''}"
    else:
        n = len(others)
        label = f"+{n} option{'s' if n != 1 else ''}"
    return f'<span class="cas-variants">{"".join(dots[:6])}<span>{label}</span></span>'


def extras_html(product: dict) -> str:
    """Badges and variant swatches in one row with a fixed min-height, so the Add to
    cart buttons of a grid row still line up."""
    return f'<div class="cas-extras">{badges_html(product)}{variants_html(product)}</div>'


def _image_html(url, css_class: str) -> str:
    src = safe_img(url)
    if not src:
        return f'<div class="{css_class} empty"></div>'
    return f'<div class="{css_class}"><img src="{src}" alt="" loading="lazy"></div>'


def brand_name(product: dict) -> str:
    """The brand to print: the API's `brand` (2026-10-27: the maker named in the listing's
    specs where the store is a chip vendor -- "intel" sold a Lenovo laptop), else the
    store. One line of seller text; escape it at the sink."""
    return _text(product.get("brand")) or _text(product.get("store"))


def card_head_html(product: dict, *, compact: bool = False) -> str:
    """Image, the Sponsored line and the brand: the part of a card above the title link.
    Pure; every catalog value is escaped (stores and brands are seller text)."""
    # Every ad carries the label, so the ranking stays auditable on screen
    spons = "Sponsored" if product.get("is_sponsored") else ""
    return (
        f'<div class="cas-card{" compact" if compact else ""}">'
        f"{_image_html(product.get('img_url'), 'cas-img')}"
        f'<div class="cas-spons">{spons}</div>'
        f'<div class="cas-brand">{esc(brand_name(product))}</div></div>'
    )


def card_tail_html(product: dict, *, compact: bool = False) -> str:
    """Rating, price and the badge/variant row: the part of a card below the title."""
    return (
        f'<div class="cas-card{" compact" if compact else ""}">'
        f"{stars_html(product.get('average_rating'), product.get('rating_number'))}"
        f"{price_html(product)}{extras_html(product)}</div>"
    )


def card_html(product: dict, *, compact: bool = False) -> str:
    """The whole display half of a card as one string (the grid draws the title as a
    button between the head and the tail). Pure."""
    title = _text(product.get("Product_title")) or "Untitled"
    return (
        f"{card_head_html(product, compact=compact)}"
        f'<div class="cas-title" title="{esc(title)}">{esc(title)}</div>'
        f"{card_tail_html(product, compact=compact)}"
    )


def match_caption(product: dict) -> str:
    """The raw relevance line for a product card: Developer details only.

    Shoppers no longer see '% match' (2026-10-15): a reranker score is a pipeline
    number, and next to a weak result it read as a defect. The dev line is unchanged."""
    if not _dev_details():
        return ""
    final = product.get("final_score")
    rerank_val = product.get("rerank_score")
    embed = product.get("score")
    best_match = product.get("best_match_score")

    if final is not None and not is_nan(final):
        line = f"**{final:.0f}% match**"
        bayes = product.get("bayesian_rating")
        bits = [f"rerank {rerank_val}"]
        if bayes:
            bits.append(f"Bayes {bayes}")
        if isinstance(embed, (int, float)) and not is_nan(embed):
            bits.append(f"embed {embed:.3f}")
        return line + f"  ·  _{' · '.join(bits)}_"

    if rerank_val is not None:
        line = f"**{rerank_val}% match**"
        if isinstance(embed, (int, float)) and not is_nan(embed):
            line += f"  ·  _embed {embed:.3f}_"
        return line

    if embed is not None and not is_nan(embed):
        # Cosine similarity, not an LLM judgement — label it as similarity
        return f"≈ {float(embed) * 100:.0f}% similar  ·  _embed {float(embed):.3f}_"

    if best_match is not None:
        return (
            f"🔤 Keyword match  ·  _best-match {best_match:.3f} · "
            f"BM25 {product.get('keyword_score')}_"
        )

    return ""


def _dev_card_lines(product: dict, *, reason_chars: int | None = 90) -> None:
    """Scores, ad relevance and the reranker's note: Developer details only."""
    if not _dev_details():
        return
    if product.get("is_sponsored"):
        rel = _num(product.get("sponsored_relevance"))
        sponsor = escape_md(product.get("sponsor") or "a partner")
        bits = f"_Sponsored by {sponsor}"
        if rel is not None:
            bits += f" · relevance to query {rel:.3f} (cosine; not AI-ranked)"
        st.caption(bits + "_")
    caption = match_caption(product)
    if caption:
        st.caption(caption)
    reason = friendly_reason(product.get("reason", ""))
    if reason:
        cut = reason if reason_chars is None else reason[:reason_chars]
        more = "…" if reason_chars is not None and len(reason) > reason_chars else ""
        st.caption(f"💡 _{escape_md(cut)}{more}_")


# ---- scroll control ----------------------------------------------------------------
# Streamlit 1.58 scrolls <section data-testid="stMain">, not the window, and a rerun
# keeps its scrollTop -- so Next/Previous left the shopper at the bottom of page 2.
# A page with st.chat_input (the Assistant) has no stMain testid: the same section is
# data-testid="stAppScrollToBottomContainer" there (measured), hence the second selector.
# The script is CONSTANT text: no model, catalog or shopper text is interpolated (the
# target is a widget key reduced to [A-Za-z0-9_-]), so unsafe_allow_javascript is safe.
# The nonce makes each emission differ from the last, so it is mounted and runs again.
_SCROLL_JS = """<script>/*cas-scroll {nonce}*/(function () {{
  var doc = window.parent.document;
  var target = {target};
  var tries = 0, guarded = false, released = false;
  function main() {{
    return doc.querySelector('[data-testid="stMain"]')
        || doc.querySelector('[data-testid="stAppScrollToBottomContainer"]')
        || doc.querySelector('[data-testid="stAppViewContainer"]')
        || doc.querySelector('section.main') || doc.scrollingElement;
  }}
  /* The Assistant's container sticks to the bottom: as the panel's images load it pulled
     Back down to the chat box, and it undid Back to top 16 ms after the jump, then about
     once a second (all measured). An upward wheel is how it learns the shopper scrolled
     away, so one is sent just before and one just after the FIRST jump (one side alone
     held only one of the two jumps, measured). Sent before hold() listens, so hold()
     never reads them as the shopper's own touch; elsewhere nothing listens to them (an
     untrusted event scrolls nothing) */
  function wheelUp(m) {{
    try {{
      var W = doc.defaultView.WheelEvent;
      m.dispatchEvent(new W('wheel', {{deltaY: -120, bubbles: true, cancelable: true}}));
    }} catch (e) {{}}
  }}
  function jump(put, m) {{
    if (released) {{ put(); return; }}
    released = true;
    wheelUp(m);
    put();
    wheelUp(m);
  }}
  /* And for 4 s, until the shopper touches the page, a jump to the very bottom
     re-applies `put` (the card, or the top). No timers: hidden tabs throttle them */
  function hold(put, m) {{
    if (guarded) return;
    guarded = true;
    var t0 = Date.now(), user = false;
    function mark() {{ user = true; }}
    ['wheel', 'touchstart', 'keydown', 'mousedown'].forEach(function (e) {{
      doc.addEventListener(e, mark, {{once: true, passive: true}});
    }});
    function onScroll() {{
      if (user || Date.now() - t0 > 4000) {{ m.removeEventListener('scroll', onScroll); return; }}
      if (m.scrollTop > 0 && m.scrollTop + m.clientHeight >= m.scrollHeight - 4) put();
    }}
    m.addEventListener('scroll', onScroll, {{passive: true}});
  }}
  function go() {{
    var m = main();
    if (!m) return;
    if (target) {{
      var el = doc.querySelector('.st-key-' + target);
      /* the whole card at the top, below the sticky band (the cards' scroll-margin-top):
         centring the button left the card's image under the band on an 800px screen */
      if (el) {{
        var card = el.closest('[class*="st-key-card_"]') || el;
        var toCard = function () {{ card.scrollIntoView({{block: 'start'}}); }};
        jump(toCard, m);
        hold(toCard, m);
        return;
      }}
      if (++tries < 40) {{ setTimeout(go, 75); return; }}   /* card not rendered yet */
    }}
    var toTop = function () {{ m.scrollTop = 0; }};
    jump(toTop, m);
    hold(toTop, m);
  }}
  go(); setTimeout(go, 60); setTimeout(go, 300);
}})();</script>"""

# Hide the keyed container AND its stLayoutWrapper: the wrapper is a flex child of the
# page, so it would otherwise add one 16px gap (measured) above the title.
# cas_scroll_late = the product page's own emission after its fetch ("Related items"):
# another key, so even a run that somehow emitted at the top too cannot raise a
# duplicate-key error and take the page down
_SCROLL_CSS = (
    "<style>.st-key-cas_scroll,"
    "[data-testid=\"stLayoutWrapper\"]:has(> .st-key-cas_scroll),"
    ".st-key-cas_scroll_late,"
    "[data-testid=\"stLayoutWrapper\"]:has(> .st-key-cas_scroll_late)"
    "{display:none !important}</style>"
)


def request_scroll(target_key: str | None = None) -> None:
    """Ask the NEXT run to scroll: to the top, or to the widget with this key."""
    sset("_scroll_request", target_key or "__top__")


def apply_scroll(*, late: bool = False) -> None:
    """Emit the one-shot scroll script if requested. Call at the TOP of the page, so the
    jump happens before any slow fetch (the product page waits on cross-sell).

    late=True: the product page's single emission AFTER its fetch, for a target that
    only exists once the fetch is drawn (the suggestions). Never a second emission for
    the same click: an open with a focus requests no top scroll (_select_detail)."""
    req = sget("_scroll_request")
    if not req:
        return
    sset("_scroll_request", None)
    nonce = (sget("_scroll_nonce") or 0) + 1
    sset("_scroll_nonce", nonce)
    target = "null" if req == "__top__" else repr(re.sub(r"[^A-Za-z0-9_-]", "", req))
    st.html(_SCROLL_CSS)   # style-only: goes to the event container, takes no space
    # display:none container: no 16px flex gap / layout shift, and the script still runs
    with st.container(key="cas_scroll_late" if late else "cas_scroll"):
        st.html(_SCROLL_JS.format(nonce=nonce, target=target), unsafe_allow_javascript=True)


# ---- navigation between the results and a product page -------------------------------
def _select_detail(catalog_index: int, return_key: str | None = None,
                   variants=None, focus: str | None = None) -> None:
    """State half of opening a product page; safe as an on_click callback.

    return_key = the grid button that opened it, so Back lands on that card. A product
    opened from a recommendation strip keeps the grid card it was reached from.
    `variants` = the card's collapsed colour/size variants, for "Also available in";
    set on EVERY open (none = []): a card without variants, or a recommendation, kept
    the previous product's list, so its page offered another product's colours.

    focus="related" ("Related items" on a card): the page opens at its suggestions.
    That anchor exists only after the slow /product fetch, so render_detail_page scrolls
    there itself, and no scroll to the top is requested here: one script per run (a top
    jump's hold would also pull the view back up). Set on EVERY open, so a stale focus
    can never fire on a later open."""
    sset("view", "detail")
    sset("detail_index", int(catalog_index))
    sset("detail_response", None)
    sset("detail_variants", list(variants or []))
    if return_key:
        sset("return_card", return_key)
    sset("detail_focus", focus)
    if focus is None:
        request_scroll()
    else:
        sset("_scroll_request", None)   # nothing else may jump in that run


def _select_results() -> None:
    """State half of going back to the results; safe as an on_click callback."""
    sset("view", "search")
    request_scroll(sget("return_card"))


def open_detail(catalog_index: int, return_key: str | None = None):
    """Open the product page for this surface only (namespaced)."""
    _select_detail(catalog_index, return_key)
    st.rerun()


def back_to_results():
    _select_results()
    st.rerun()


def fetch_product(catalog_index: int, query: str, recommend: bool):
    """Load a single product + its recommendations from /product."""
    try:
        r = requests.get(
            PRODUCT_URL,
            params={
                "catalog_index": catalog_index,
                # /product caps query at 300; a long chat message must not 422 the page
                "query": (query or "")[:300],
                "recommend": str(recommend).lower(),
            },
            timeout=60,
        )
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        st.error(f"Backend error: {escape_md(e)}")   # the URL in it carries the query
        return None


def post_feedback(query: str, product_title: str, rating: int, rank, reason: str):
    try:
        requests.post(
            FEEDBACK_URL,
            json={
                "query": query,
                "product_title": product_title,
                "rating": rating,
                "rank": rank or 1,
                "reason": reason,
            },
            timeout=5,
        )
    except requests.RequestException as e:
        st.warning(f"Could not record feedback: {escape_md(e)}")


# ---- cart ---------------------------------------------------------------------------
def _add_to_cart(product: dict, qty: int = 1) -> None:
    # A callback runs before the page script, so the masthead count is already right in
    # the same run: one run per click, and no scroll request (the view stays put)
    st.toast(f"Added — {ui_cart.add(product, int(qty or 1))} in cart")


def add_to_cart_button(product: dict, key_ns: str = "", *, wide: bool = False, qty: int = 1):
    """Add-to-cart control; shows the current quantity once the item is in the cart.
    Coloured by key-class CSS (st-key-cart_*: SKIN's cart-btn tokens), not by the theme."""
    cid = product.get("catalog_index")
    if cid is None:
        return
    in_cart = ui_cart.in_cart(cid)
    label = "Add to cart" if not in_cart else f"Add another ({in_cart} in cart)"
    st.button(
        label, key=f"cart_{key_ns}_{cid}", type="primary",
        width="stretch" if wide else "content",
        on_click=_add_to_cart, args=(product, qty),
    )


def cart_line_html(product: dict, qty: int) -> str:
    """Image, title, brand and price of one cart row. Pure. Title and price go through
    st.html, never markdown: a title is seller text, and two '$' in one markdown line
    render as LaTeX. An estimate is labelled and never counted in the total."""
    price, est = _num(product.get("price")), _num(product.get("price_estimate"))
    if price is not None:
        money = f"${price:,.2f}"
        note = (f'<div class="cas-meta">${price:,.2f} × {qty} = ${price * qty:,.2f}</div>'
                if qty > 1 else "")
    elif est is not None:
        money = f'≈ ${est:,.2f}<span class="est">est.</span>'
        note = ('<div class="cas-meta">Estimated from similar products: not in the total'
                "</div>")
    else:
        money = '<span class="cas-meta">no listed price</span>'
        note = '<div class="cas-meta">No listed price: not in the total</div>'
    title = _text(product.get("Product_title")) or "Untitled"
    name = brand_name(product)
    brand = f'<div class="cas-meta">Brand: {esc(name)}</div>' if name else ""
    return (
        f'<div class="cas-cart-row">{_image_html(product.get("img_url"), "cas-img")}'
        f'<div class="cas-cart-main"><div class="cas-title" title="{esc(title)}">{esc(title)}'
        f"</div>{brand}{note}</div>"
        f'<div class="cas-cart-price">{money}</div></div>'
    )


# ---- product cards --------------------------------------------------------------------
# The card's way to the suggestions (Oct 27: "a button on each result for the cross-sell
# suggestions"). Constant text, honest under all three outcomes -- complements, the
# look-alike fallback ("Similar items") and the upsell -- and never "bought together"
# (there is no purchase data)
RELATED_LABEL = "Related items"
RELATED_HELP = ("Opens this item's page at its suggestions: related items and, where "
                "there is one, a better-rated alternative")


def _card_body(product: dict, *, cart_ns: str, view_key: str, from_grid: bool,
               compact: bool = False, related_key: str | None = None) -> None:
    """One product inside a card shell: image, brand, the TITLE as the link to the
    product page (a button keyed view_key: Back scrolls to it), rating, price, cart.
    related_key: also a "Related items" button under Add to cart (grid cards only)."""
    st.html(card_head_html(product, compact=compact))
    cid = product.get("catalog_index")
    title = _text(product.get("Product_title")) or "Untitled"
    if cid is None:
        st.html(f'<div class="cas-title">{esc(title)}</div>')
    else:
        # A button label is markdown: seller text MUST be escaped (safety.md)
        st.button(
            escape_md(title), key=view_key, type="tertiary", help=escape_md(title),
            width="stretch", on_click=_select_detail,
            args=(int(cid), view_key if from_grid else None, product.get("variants")),
        )
    st.html(card_tail_html(product, compact=compact))
    _dev_card_lines(product)
    if cid is None:
        return
    add_to_cart_button(product, key_ns=cart_ns)
    if related_key:
        # Opens the same page as the title (one /product call, no extra LLM call), at its
        # suggestions. Back returns to this card: the return key is the TITLE's
        st.button(RELATED_LABEL, key=related_key, type="secondary", help=RELATED_HELP,
                  icon=":material/chevron_right:", icon_position="right",
                  on_click=_select_detail,
                  args=(int(cid), view_key if from_grid else None, product.get("variants")),
                  kwargs={"focus": "related"})


def render_grid_card(product: dict, rank: int = 0, key_ns: str = "", bundle_type=None,
                     related: bool = False):
    """Grid card; a bundle card is one component + its `alternatives` as Option tabs.

    The title button key stays `grid_{key_ns}_{cid}`: Back scrolls to `.st-key-<key>`.
    related: a "Related items" button per product, keyed `related_{key_ns}_{cid}` (never
    `grid_*`: the title-link CSS and the tests' title lookups read that prefix)."""
    cid = product.get("catalog_index")
    shell_key = f"card_{surface()}_{key_ns}_{cid}" if cid is not None else None
    with st.container(border=True, key=shell_key):
        alternatives = product.get("alternatives") or []
        options = [product] + list(alternatives)
        if alternatives:
            ingredient = product.get("source_intent")
            bui = bundle_ui(bundle_type) or BUNDLE_UI["recipe"]
            if ingredient:
                st.caption(
                    f"{bui['slot_label']} **{escape_md(ingredient)}** — "
                    f"{len(options)} {bui['slot_word']} to choose from"
                )
            tabs = st.tabs([f"Option {i + 1}" for i in range(len(options))])
            for tab, opt in zip(tabs, options):
                with tab:
                    ocid = opt.get("catalog_index")
                    _card_body(opt, cart_ns=f"grid{key_ns}",
                               view_key=f"grid_{key_ns}_{ocid}", from_grid=True,
                               related_key=f"related_{key_ns}_{ocid}" if related else None)
        else:
            _card_body(product, cart_ns=f"grid{key_ns}", view_key=f"grid_{key_ns}_{cid}",
                       from_grid=True,
                       related_key=f"related_{key_ns}_{cid}" if related else None)


def render_mini(item: dict, key_prefix: str, *, width=None):
    """Compact card for the recommendation carousels on a product page."""
    cid = item.get("catalog_index")
    with st.container(border=True, key=f"card_{surface()}_{key_prefix}_{cid}",
                      width=width or "stretch"):
        _card_body(item, cart_ns=key_prefix, view_key=f"{key_prefix}_{cid}",
                   from_grid=False, compact=True)
        if item.get("recommend_reason"):
            st.caption(f"_{escape_md(item['recommend_reason'])}_")


def render_product_grid(products: list, *, key_ns: str = "", bundle_type=None,
                        per_row: int = 4, related: bool = False) -> None:
    """Rows of `per_row` cards (two per row on a phone, by CSS). related = the page's
    "Product recommendations" setting: with it off the product page has no suggestions,
    so the cards offer no "Related items" button."""
    with st.container(key=f"cas_grid_{key_ns or 'main'}"):
        for row_start in range(0, len(products), per_row):
            cols = st.columns(per_row, gap="medium")
            for col, (offset, product) in zip(
                    cols, enumerate(products[row_start:row_start + per_row])):
                with col:
                    render_grid_card(product, rank=row_start + offset + 1, key_ns=key_ns,
                                     bundle_type=bundle_type, related=related)


# ---- page furniture -------------------------------------------------------------------
def masthead(*, settings=None, placeholder: str = "", label: str = "Search",
             title: str | None = None, scopes=None):
    """The search band under the header: [scope | search text | 🔍] Settings  Cart n.
    Enter submits the form. With `title`, the band shows that title instead of the form
    (the Assistant searches through its chat box). `scopes` adds the department select,
    which submits with the query.

    Returns (submitted query or None, whatever `settings()` returned)."""
    submitted = None
    with st.container(key="cas_masthead", horizontal=True, vertical_alignment="center",
                      gap="small"):
        if title:
            st.html(f'<div class="cas-masthead-title">{esc(title)}</div>', width="stretch")
        else:
            key = ns("q_input")
            if key not in st.session_state:
                st.session_state[key] = sget("query") or ""
            with st.form(ns("search_form"), border=False, enter_to_submit=True,
                         width="stretch"):
                # A horizontal container, not st.columns: columns stack one per row
                # under 640px, and the control must stay one joined row
                with st.container(key="cas_searchrow", horizontal=True, gap=None,
                                  vertical_alignment="center"):
                    if scopes:
                        st.selectbox(
                            "Search in", scopes, key=ns("scope"), width=112,
                            label_visibility="collapsed",
                            help="Narrows the matched results to one department.",
                        )
                    # 300 = the API's query cap: longer text would come back as a 422
                    text = st.text_input(label, key=key, placeholder=placeholder,
                                         max_chars=300, label_visibility="collapsed")
                    # The label stays "Search" (screen readers, tests); CSS hides it
                    # behind the magnifier
                    if st.form_submit_button("Search", type="primary",
                                             icon=":material/search:"):
                        # strip only: the API (and its guard) sees the shopper's own text
                        submitted = (text or "").strip() or None
        values = None
        if settings is not None:
            with st.popover("Settings", icon=":material/tune:"):
                values = settings()
        # st.switch_page, never st.page_link (KeyError 'url_pathname' outside a full
        # page context). The Cart page is hidden from the nav: this is its way in
        if st.button(f"Cart **{ui_cart.count()}**", key=ns("topbar_cart"),
                     icon=":material/shopping_cart:"):
            st.switch_page("pages/cart.py")
    return submitted, values


top_bar = masthead   # the 2026-10-15 name


# Sub-nav shortcuts: (label, department, gender). Women/Men/Kids need the API's
# `gender` filter (C2); an API without it sends no `genders` facet and they hide.
SUBNAV = (
    ("All", None, None), ("Fashion", "Fashion", None), ("Women", "Fashion", "women"),
    ("Men", "Fashion", "men"), ("Kids", "Fashion", "kids"),
    ("Electronics", "Electronics", None), ("Grocery", "Grocery", None),
)


def _fold(values) -> list:
    return [str(v).casefold() for v in (values or [])]


def _subnav_active(filters: dict, has_query: bool) -> int:
    """Which shortcut describes what is on screen (or, before a search, the scope)."""
    if has_query:
        dept = _fold(filters.get("department"))
        gender = _fold(filters.get("gender"))
        if gender and not dept:
            dept = ["fashion"]   # only Fashion rows carry a shopper gender
    else:
        scope = st.session_state.get(ns("scope")) or "All"
        dept = [] if scope == "All" else [scope.casefold()]
        gender = [sget("scope_gender")] if sget("scope_gender") and dept == ["fashion"] else []
    for i, (_, d, g) in enumerate(SUBNAV):
        if dept == ([d.casefold()] if d else []) and gender == ([g] if g else []):
            return i
    return -1


def _on_subnav(dept, gender) -> None:
    """A shortcut FILTERS the matched results (0 LLM calls); it does not re-run the
    search inside a department. Before a search it only sets the scope, and Women /
    Men / Kids then ride along on that one search (scope_gender). On results the gender
    filters THOSE results only: the next search sends the select's department alone
    (ui_spec 3.2), since the select cannot show a gender."""
    st.session_state[ns("scope")] = dept or "All"
    sset("scope_gender", None if sget("query") else gender)
    if not sget("query"):
        return
    f = dict(sget("filters") or {})
    old = _fold(f.get("department"))
    f["department"] = [dept] if dept else []
    f["gender"] = [gender] if gender else []
    if old != _fold(f["department"]):
        f["ptype"], f["subtype"] = [], []   # the levels below a department
        f["attr"] = {}                      # and its own sections (Electronics, Grocery)
    sset("filters", f)
    sset("view", "search")
    _filters_changed()


def subnav_for(data) -> None:
    """The sub-nav where its shortcuts can work: before any results (they set the scope)
    or on a flat page with facets -- the rule that gates the rail. A bundle or no-match
    page ignores filters (app/main.py search), so a shortcut there highlighted itself and
    moved the scope over a page it never changed (2026-10-19 review). Draw it AFTER the
    run's fetch, or it describes the previous page.

    On results only shortcuts that can show something stay (2026-10-27): Women / Men /
    Kids each need their value in the `genders` facet (the API sends the key on every
    pool since 2026-10-19, as [] on Electronics or Grocery, and lists only genders with
    an on-target row -- the rail's own gender links), and a department shortcut needs
    that department in the pool -- on a Grocery page Women, Men, Kids, Fashion and
    Electronics each returned "No products match these filters", and on a pool offering
    Women and Unisex so did Men and Kids. The active shortcut always stays. Before any
    answer all seven stay: there they set the scope of the next search."""
    if not data:
        subnav()
        return
    facets = data.get("facets")
    if data.get("no_match") or not isinstance(facets, dict) or bundle_ui(data.get("bundle_type")):
        return
    departments = facets.get("departments")
    subnav(genders={key for key, _, _ in _facet_items(facets.get("genders"))},
           departments=None if departments is None
           else {key for key, _, _ in _facet_items(departments)})


def subnav(*, genders=None, departments=None) -> None:
    """Row 3 of the header: department (and gender) shortcuts. AI Search only.
    `genders` / `departments` = the pool's offered values (None: unknown, i.e. before a
    search; an older API without a `genders` key offers none): a shortcut to a value the
    pool lacks is left out unless it is the active one."""
    active = _subnav_active(sget("filters") or {}, bool(sget("query")))
    have = None if departments is None else {str(d).casefold() for d in departments}
    have_g = None if genders is None else {str(g).casefold() for g in genders}
    with st.container(key="cas_subnav", horizontal=True, gap=None,
                      vertical_alignment="center"):
        for i, (label, dept, gender) in enumerate(SUBNAV):
            if i != active and gender and have_g is not None and gender not in have_g:
                continue
            if i != active and have is not None and dept and dept.casefold() not in have:
                continue
            st.button(label, key=ns(f"subnav_{i}"),
                      type="secondary" if i == active else "tertiary",
                      on_click=_on_subnav, args=(dept, gender))


def results_bar(header_html: str, *, sortable: bool = False) -> None:
    """The full-bleed bar under the header: the results count left, Sort right."""
    with st.container(key="cas_resultsbar", horizontal=True, vertical_alignment="center",
                      horizontal_alignment="distribute"):
        st.html(header_html, width="stretch")
        if sortable:
            sort_select()


def page_window(page: int, total: int, radius: int = 2) -> list:
    """Page numbers to show: first, last, and `radius` either side of the current page,
    with '…' for a gap of two or more pages (a gap of one shows that page). Pure.

    page_window(7, 15) -> [1, '…', 5, 6, 7, 8, 9, '…', 15]"""
    total = max(1, int(total))
    page = min(max(1, int(page)), total)
    keep = {1, total} | set(range(max(1, page - radius), min(total, page + radius) + 1))
    out, prev = [], 0
    for p in sorted(keep):
        if p - prev == 2:
            out.append(prev + 1)
        elif p - prev > 2:
            out.append("…")
        out.append(p)
        prev = p
    return out


def _goto_page(p: int) -> None:
    sset("page", int(p))
    request_scroll()


def pagination(page: int, total_pages: int) -> None:
    """Previous, a numbered window and Next as one joined strip. on_click callbacks: one
    run per click, and the normal fetch path loads the new page."""
    if not total_pages or total_pages <= 1:
        return
    with st.container(horizontal=True, horizontal_alignment="center", gap=None,
                      key=ns("pager")):
        st.button("Previous", key=ns("pg_prev"), icon=":material/chevron_left:",
                  disabled=page <= 1, on_click=_goto_page, args=(page - 1,))
        for i, p in enumerate(page_window(page, total_pages)):
            if p == "…":
                st.button("…", key=ns(f"pg_gap_{i}"), disabled=True)
            else:
                st.button(str(p), key=ns(f"pg_{p}"), disabled=p == page,
                          type="primary" if p == page else "secondary",
                          on_click=_goto_page, args=(p,))
        st.button("Next", key=ns("pg_next"), icon=":material/chevron_right:",
                  icon_position="right", disabled=page >= total_pages,
                  on_click=_goto_page, args=(page + 1,))


def results_header_html(start: int, end: int, total: int, query: str,
                        total_unfiltered=None) -> str:
    """'1–12 of 172 results for “query”'. Pure; the query is the shopper's text."""
    if total <= 0:
        return (f'<div class="cas-results-head">No results for “<b class="q">{esc(query)}'
                f"</b>”</div>")
    sub = ""
    if total_unfiltered is not None and total_unfiltered != total:
        sub = f' <span class="sub">(filtered from {int(total_unfiltered):,})</span>'
    return (
        f'<div class="cas-results-head">{start:,}–{end:,} of {int(total):,} results for '
        f'“<b class="q">{esc(query)}</b>”{sub}</div>'
    )


def heading_html(text: str) -> str:
    """A section heading (21px bold) for fixed copy; escaped all the same."""
    return f'<div class="cas-h2">{esc(text)}</div>'


def weak_match_html(query: str) -> str:
    """The banner for a page where the ranking judged few rows a real match (the API's
    match_quality.weak): say so instead of letting near-misses pose as exact. Pure."""
    return (f'<div class="cas-notice">No exact matches for “<b>{esc(query)}</b>”. '
            "Showing related items.</div>")


def match_is_weak(data: dict) -> bool:
    """match_quality.weak, never when the ranking did not judge the pool (rerank off
    or fell back: 'unjudged') or the API predates the field. Nor on a page with no
    results: match_quality reads the UNFILTERED pool, and "Showing related items" over
    "No products match these filters." contradicted itself (2026-10-19 review)."""
    data = data or {}
    mq = data.get("match_quality")
    return (isinstance(mq, dict) and bool(mq.get("weak")) and not mq.get("unjudged")
            and bool(data.get("results")))


# The marketplace wordmark is OURS: the app's name and a magnifier, nothing else
def wordmark_svg(width: int = 300) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="32" '
        'viewBox="0 0 300 32" role="img" aria-label="Context-Aware Agentic Search">'
        f'<circle cx="12" cy="14" r="7.5" fill="none" stroke="{SKIN["search-btn"]}" '
        'stroke-width="3"/>'
        f'<line x1="17.5" y1="19.5" x2="23" y2="25" stroke="{SKIN["search-btn"]}" '
        'stroke-width="3" stroke-linecap="round"/>'
        '<text x="32" y="21.5" font-family="Arial, Helvetica, sans-serif" font-size="16.5" '
        f'font-weight="700" fill="{SKIN["belt-fg"]}">Context-Aware Agentic Search</text>'
        "</svg>"
    )


_FOOT_PAGES = (("AI Search", "ai", "pages/ai_search.py"),
               ("Keyword Search", "keyword", "pages/keyword_search.py"),
               ("Shopping Assistant", "assistant", "pages/shopping_assistant.py"),
               ("Cart", "cart", "pages/cart.py"))

# The dataset name is the single permitted mention of the marketplace: a legal
# citation, not branding. With a look-alike skin the non-affiliation line matters more
LICENCE_HTML = (
    '<div class="cas-footer"><b class="mark">Context-Aware Agentic Search</b>'
    '<span>Internal demo · catalog: Amazon Reviews 2023 '
    "(research licence)</span><span>Independent proof of concept, not affiliated with or "
    "endorsed by any retailer.</span></div>"
)


def footer() -> None:
    """Back to top, three columns and the licence strip; ui.py calls it after the page.
    Every shared build says so: the catalog's licence forbids client-facing use."""
    with st.container(key="cas_footer"):
        # the next run's apply_scroll() (every page calls it first) jumps to the top
        st.button("Back to top", key="cas_back_to_top", width="stretch",
                  on_click=request_scroll)
        with st.container(key="cas_footcols"):
            about, data, pages = st.columns(3, gap="large")
            about.html('<div class="cas-foot"><h4>About this demo</h4><p>Context-aware '
                       "search: an LLM reads the request, embeddings retrieve candidates, "
                       "an LLM ranks them.</p></div>")
            data.html('<div class="cas-foot"><h4>Data</h4><p>Catalog: Amazon Reviews 2023 '
                      "(McAuley Lab), research licence: not for commercial use. Images "
                      "load from the catalog's image URLs.</p></div>")
            with pages:
                st.html('<div class="cas-foot"><h4>Pages</h4></div>')
                for label, key, path in _FOOT_PAGES:
                    if st.button(label, key=f"cas_foot_{key}", type="tertiary"):
                        st.switch_page(path)
        # text, not the SVG wordmark: st.html's sanitiser drops inline SVG (measured)
        st.html(LICENCE_HTML)


# ---- filters + sort (AI Search; the API's facets contract) ----------------------------
# Values are what the API takes; the labels are the familiar marketplace wording
# ("Relevance", not "Featured": the order is relevance with labelled ads, not
# merchandising)
SORT_OPTIONS = {
    "relevance": "Relevance",
    "price_asc": "Price: Low to High",
    "price_desc": "Price: High to Low",
    "rating": "Avg. Customer Review",
}
# Multi-value filters sent as one csv param; `store` goes as REPEATED params (606
# catalog store names contain a comma, "Topsville, Inc.")
_CSV_FILTERS = ("department", "gender", "ptype", "subtype", "color", "material")
_FILTER_KINDS = _CSV_FILTERS + ("store",)
# The most `store` values GET /search accepts (app/main.py _MAX_STORES; more is a 422).
# The Brands rail offers up to 30, so it stops at this many ticks: the 21st used to
# wipe the page under "Could not reach the search service" (2026-10-19 review)
MAX_STORES = 20
# Per-department sections the API describes in facets.attributes (2026-10-27: Electronics
# Category / Item type / Works with / Condition, Grocery Category / Dietary). Each names
# its own csv /search param, so a new section needs no UI code. A name the UI could not
# send safely -- malformed, or one of /search's other params -- is never drawn or sent:
# the API's text must not be able to steer a filter the rail already owns
_ATTR_PARAM = re.compile(r"[a-z][a-z_]{0,31}")
_RESERVED_PARAMS = frozenset(_FILTER_KINDS) | {
    "query", "top_k", "page", "rerank", "recommend", "sponsored", "translator_mode",
    "sort", "min_price", "max_price", "min_rating", "follow_up", "attr"}


def _attr_param_ok(param) -> bool:
    return (isinstance(param, str) and bool(_ATTR_PARAM.fullmatch(param))
            and param not in _RESERVED_PARAMS)


def _attr_filters(f: dict) -> dict:
    """The ticked attribute values, {param: [keys]}, minus anything unsendable, each key
    once (a repeated key would be sent twice and draw two boxes with one widget key: a
    StreamlitDuplicateElementKey takes the page down). Pure."""
    attr = (f or {}).get("attr")
    if not isinstance(attr, dict):
        return {}
    return {p: list(dict.fromkeys(str(v) for v in values)) for p, values in attr.items()
            if values and isinstance(values, (list, tuple)) and _attr_param_ok(p)}


def _priced_share(facets, total) -> tuple | None:
    """(rows with a listed price, rows in the pool) when SOME lack one, else None."""
    n_priced = _num(((facets or {}).get("price") or {}).get("n_priced"))
    total = _num(total)
    if n_priced is None or not total or n_priced >= total:
        return None
    return int(n_priced), int(total)


def sort_note(sort: str | None, facets, total) -> str | None:
    """The caption under a price or rating sort; None for relevance. Pure.

    The API orders rows within relevance groups (closest matches first), and a price
    sort reads the LISTED price only. Most Fashion rows have none ('beach wedding
    outfit': 6-10 of 173), so their cards' estimates followed the priced ones and
    the page read as a broken sort (Oct 15 review): the caption says why."""
    if sort == "rating":
        return "Closest matches first, then by customer rating."
    if sort not in ("price_asc", "price_desc"):
        return None
    note = "Closest matches first, then by listed price."
    share = _priced_share(facets, total)
    if share:
        note += (f" Only {share[0]:,} of {share[1]:,} items list a price; the others "
                 "(estimates are marked “est.”) are not sorted by price.")
    return note


def filter_params(filters: dict, sort: str | None) -> dict:
    """The /search query params for the active filters and sort; {} when none. Pure.

    Only ACTIVE filters are sent: an unchanged price range sends nothing, because any
    price bound makes the API drop every row without a listed price. `store` is a list
    (requests sends it as repeated params). Attribute filters go as one sorted csv per
    section param (category=..., dietary=...)."""
    out = {}
    if sort and sort != "relevance":
        out["sort"] = sort
    f = filters or {}
    for kind in _CSV_FILTERS:
        if f.get(kind):
            out[kind] = ",".join(sorted(f[kind]))
    if f.get("min_price") is not None:
        out["min_price"] = f["min_price"]
    if f.get("max_price") is not None:
        out["max_price"] = f["max_price"]
    if f.get("min_rating"):
        out["min_rating"] = f["min_rating"]
    if f.get("store"):
        out["store"] = sorted(f["store"])
    for param, values in sorted(_attr_filters(f).items()):
        out[param] = ",".join(sorted(values))
    return out


def params_signature(params: dict) -> tuple:
    """A comparable record of what a fetch asked for (lists become tuples)."""
    return tuple(sorted((k, tuple(v) if isinstance(v, list) else v)
                        for k, v in params.items()))


def reset_filters() -> None:
    """New search or 'Clear filters': no filters, relevance order, fresh widgets (the
    widget keys carry filter_seq, so the old ones are dropped instead of reused)."""
    sset("filters", {})
    sset("sort", "relevance")
    sset("filter_seq", (sget("filter_seq") or 0) + 1)


def _filters_changed() -> None:
    sset("page", 1)
    request_scroll()


def _sync_scope(f: dict) -> None:
    """Keep the masthead's department select on what the rail shows (AI Search only:
    the other surfaces never draw it, so the value is simply unused there).

    The department only, never a gender: a rail 'Women' used to become scope_gender,
    and every later search went out as gender=women under a select reading 'Fashion'
    ("men's linen shirt" got women's items or nothing; 2026-10-19 review)."""
    dept = f.get("department") or []
    gender = f.get("gender") or []
    if not dept and gender:
        dept = ["Fashion"]   # only Fashion rows carry a shopper gender
    scope = next((s for s in SCOPES if dept and s.casefold() == str(dept[0]).casefold()),
                 "All")
    st.session_state[ns("scope")] = scope
    sset("scope_gender", None)


def _on_check(kind: str, value: str, key: str) -> None:
    f = dict(sget("filters") or {})
    chosen = [v for v in (f.get(kind) or []) if v != value]
    if st.session_state.get(key):
        chosen.append(value)
    f[kind] = chosen
    sset("filters", f)
    _filters_changed()


def _on_attr_check(param: str, value: str, key: str) -> None:
    """A tick in an API-described section: filters["attr"][param] (a fresh dict, so the
    state the last fetch was made from is never mutated in place)."""
    f = dict(sget("filters") or {})
    attr = {p: list(v) for p, v in (f.get("attr") or {}).items()}
    chosen = [v for v in attr.get(param, []) if v != value]
    if st.session_state.get(key):
        chosen.append(value)
    attr[param] = chosen
    f["attr"] = attr
    sset("filters", f)
    _filters_changed()


# A link (single-select) clears the levels below it: a department its gender, type,
# style and its own attribute sections; a type its style
_CLEARS = {"department": ("gender", "ptype", "subtype", "attr"), "ptype": ("subtype",)}


def _on_link(kind: str, value) -> None:
    """Rail link: select `value` (None = 'Any ...'); a second click on the selected
    value deselects it. The department above a picked gender is drawn as a plain
    parent (_department_section), and a click there goes UP to it (ui_spec 4.3: a level
    "sets that filter and clears levels below it"): it used to deselect the department
    and the whole path, and the scope fell back to 'All' (2026-10-19 review)."""
    f = dict(sget("filters") or {})
    current = list(f.get(kind) or [])
    same = value is not None and _fold(current) == _fold([value])
    if same and kind == "department" and f.get("gender"):
        new = current
    else:
        new = [] if value is None or same else [value]
    # Every link click clears the levels below it ('Any Department' also a gender
    # picked without a department)
    for below in _CLEARS.get(kind, ()):
        f[below] = {} if below == "attr" else []
    if kind == "gender" and not f.get("department"):
        # With no department picked, a gender moves the rail's focus to Fashion (and
        # un-picking it moves it back): the other department's ticks would filter the
        # page from a section no longer drawn
        f["attr"] = {}
    f[kind] = new
    sset("filters", f)
    if kind in ("department", "gender"):
        _sync_scope(f)
    _filters_changed()


def _handle(bound, lo: int, hi: int, rounding) -> int:
    """Where the whole-dollar slider shows an applied price bound: rounded OUT (floor
    for the minimum, ceil for the maximum), inside [lo, hi]. Pure."""
    return max(lo, min(int(rounding(bound)), hi))


def _on_price(key: str, lo: int, hi: int) -> None:
    f = dict(sget("filters") or {})
    a, b = st.session_state.get(key, (lo, hi))
    # A handle the shopper did not move keeps the applied bound. A band's ceiling is
    # fractional ('$25 to $50' = 25 / 49.99, the API's _price_bands): the slider showed
    # 49 and nudging only the LOW handle re-sent max_price=49, dropping the $49.01-49.99
    # rows the band had kept (2026-10-19 review)
    old_a, old_b = f.get("min_price"), f.get("max_price")
    if old_a is not None and a == _handle(old_a, lo, hi, math.floor):
        a = old_a
    if old_b is not None and b == _handle(old_b, lo, hi, math.ceil):
        b = old_b
    f["min_price"] = a if a > lo else None
    f["max_price"] = b if b < hi else None
    sset("filters", f)
    _filters_changed()


def _on_band(lo, hi) -> None:
    """A price band link (or 'Any price' with both None). A new slider follows it: the
    slider's key carries price_rev, so it is re-seeded from the applied range."""
    f = dict(sget("filters") or {})
    same = f.get("min_price") == lo and f.get("max_price") == hi
    f["min_price"], f["max_price"] = (None, None) if same else (lo, hi)
    sset("filters", f)
    sset("price_rev", (sget("price_rev") or 0) + 1)
    _filters_changed()


def _on_rating_link(k) -> None:
    f = dict(sget("filters") or {})
    f["min_rating"] = None if not k or f.get("min_rating") == k else k
    sset("filters", f)
    _filters_changed()


def _on_sort(key: str) -> None:
    sset("sort", st.session_state.get(key) or "relevance")
    _filters_changed()


def _on_clear() -> None:
    reset_filters()
    _sync_scope({})
    _filters_changed()


def _widget_key(kind: str, value: str = "") -> str:
    # Values are catalog/type names: hashed so the key is stable and always a valid class
    digest = hashlib.md5(str(value).encode("utf-8")).hexdigest()[:8] if value else ""
    return ns(f"f_{kind}_{sget('filter_seq') or 0}_{digest}")


def _seed(key: str, value) -> None:
    """Give a widget its persisted value before it is created. The rail is not drawn on
    the product page, so Streamlit drops its widget state there; the filters themselves
    live in sget('filters') and come back with the widgets."""
    if key not in st.session_state:
        st.session_state[key] = value


def _facet_items(raw, labels: dict | None = None, *, keep_order: bool = False) -> list:
    """[(key, label, n)] from the API's list shape [{key|name, label, n}] or a plain
    {key: n} dict (the older shape), largest first (or in the API's order); blank keys
    dropped."""
    out = []
    if isinstance(raw, dict):
        raw = [{"key": k, "n": v} for k, v in raw.items()]
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        key = item.get("key", item.get("name"))
        if not _text(key):
            continue   # rows with no value: nothing a shopper could pick
        label = _text(item.get("label")) or _label(labels or {}, key)
        n = _num(item.get("n", item.get("count")))
        out.append((str(key), label, int(n or 0)))
    return out if keep_order else sorted(out, key=lambda t: (-t[2], t[1].casefold()))


def _count(n, show: bool) -> str:
    return f" ({int(n or 0):,})" if show else ""


def _rail_heading(text: str, hint: str = "") -> None:
    tip = f' <span class="hint" title="{esc(hint)}">ⓘ</span>' if hint else ""
    st.html(f'<div class="cas-rail-h">{esc(text)}{tip}</div>')


# Under a section whose ticks must ALL hold (facets.attributes match "all": Dietary). Every
# other rail section widens with each tick, so without it a shopper reads "Vegan" +
# "Kosher" as either and wonders why the page shrank. Our constant words, about the
# filter only (what the API does with the ticks), never about the products
MATCH_ALL_NOTE = "Items match every option you tick."


def _match_all(section: dict) -> bool:
    """The API's section says a row needs every ticked value (spec 3.3). Anything else
    -- "any", missing, junk -- is the any-rule every other section follows. Pure."""
    return _text((section or {}).get("match")).casefold() == "all"


def _rail_note(text: str) -> None:
    st.html(f'<div class="cas-rail-note">{esc(text)}</div>')


def _link(label_md: str, key: str, on_click, args=(), *, selected: bool = False,
          count: str = "") -> None:
    """A rail link: a tertiary button, its name bold when selected. `label_md` is
    ESCAPED markdown (or our constant text)."""
    st.button((f"**{label_md}**" if selected else label_md) + count, key=key,
              type="tertiary", on_click=on_click, args=args)


def _with_chosen(items: list, chosen: list, labels: dict | None = None) -> list:
    # A picked value the current pool lacks stays on the rail, picked, at 0: the pool can
    # change under the same query (a setting toggled, no seed on Databricks, a degraded
    # run is never cached), and the request still sends it -- an empty page with nothing
    # picked would leave the shopper no way to see or undo why
    have = {k.casefold() for k, _, _ in items}
    return items + [(v, _label(labels or {}, v), 0) for v in chosen or []
                    if str(v).casefold() not in have]


def _department_section(facets: dict, f: dict, show_counts: bool) -> None:
    """Department > Women / Men / Kids as a single-path tree of links."""
    departments = _with_chosen(_facet_items(facets.get("departments")),
                               f.get("department"))
    genders = []
    if "genders" in facets or f.get("gender"):
        # the API's fixed order (Women, Men, Kids, Unisex), not by count
        genders = _with_chosen(_facet_items(facets.get("genders"), GENDER_LABELS,
                                            keep_order=True), f.get("gender"), GENDER_LABELS)
    if not departments and not genders:
        return
    chosen_d, chosen_g = _fold(f.get("department")), _fold(f.get("gender"))
    _rail_heading("Department")
    if chosen_d or chosen_g:
        _link("‹ Any Department", _widget_key("dept_any"), _on_link, ("department", None))

    def gender_links():
        with st.container(key=ns(f"rail_l2_{sget('filter_seq') or 0}")):
            for key, label, n in genders:
                _link(escape_md(_label(GENDER_LABELS, key, label)),
                      _widget_key("gender", key), _on_link, ("gender", key),
                      selected=key.casefold() in chosen_g, count=_count(n, show_counts))

    drew_genders = False
    for name, label, n in departments:
        selected = name.casefold() in chosen_d
        if chosen_d and not selected:
            continue   # one path, like a category tree: the chosen department only
        _link(escape_md(label), _widget_key("dept", name), _on_link, ("department", name),
              selected=selected and not chosen_g, count=_count(n, show_counts))
        if name.casefold() == "fashion" and genders:
            gender_links()
            drew_genders = True
    if genders and not drew_genders and (not chosen_d or "fashion" in chosen_d):
        gender_links()


def _link_section(title: str, kind: str, items: list, f: dict, show_counts: bool,
                  any_label: str, labels: dict) -> None:
    items = _with_chosen(items, f.get(kind), labels)
    if not items:
        return
    chosen = _fold(f.get(kind))
    _rail_heading(title)
    if chosen:
        _link(any_label, _widget_key(f"{kind}_any"), _on_link, (kind, None))
    for key, label, n in items:
        if chosen and key.casefold() not in chosen:
            continue
        _link(escape_md(label), _widget_key(kind, key), _on_link, (kind, key),
              selected=key.casefold() in chosen, count=_count(n, show_counts))


def _reviews_section(facets: dict, f: dict, show_counts: bool) -> None:
    rating = facets.get("rating") or {}
    chosen = f.get("min_rating")
    if not (any((rating or {}).values()) or chosen):
        return
    _rail_heading("Customer Reviews")
    if chosen:
        _link("‹ Any rating", _widget_key("rating_any"), _on_rating_link, (None,))
    for k in (4, 3, 2, 1):
        n = _num(rating.get(str(k), rating.get(k)))
        if not n and chosen != k:
            continue
        # Constant text only; the stars carry the theme's orange
        stars = f":orange[{'★' * k}]:gray[{'★' * (5 - k)}] & Up"
        _link(stars, _widget_key("rating", str(k)), _on_rating_link, (k,),
              selected=chosen == k, count=_count(n, show_counts))


def _price_section(facets: dict, f: dict, total, show_counts: bool) -> None:
    price = facets.get("price") or {}
    lo, hi = _num(price.get("min")), _num(price.get("max"))
    bands = []
    for b in price.get("bands") or []:
        if isinstance(b, dict) and int(_num(b.get("n", b.get("count"))) or 0) > 0:
            bands.append(b)
    has_range = lo is not None and hi is not None and math.ceil(hi) > math.floor(lo)
    applied = f.get("min_price") is not None or f.get("max_price") is not None
    if not (bands or has_range or applied):
        return
    _rail_heading("Price")
    if applied:
        _link("‹ Any price", _widget_key("price_any"), _on_band, (None, None))
    for b in bands:
        b_lo, b_hi = _num(b.get("min")), _num(b.get("max"))
        n = _num(b.get("n", b.get("count")))
        # Band labels carry '$': escaped, or two of them would render as LaTeX
        _link(escape_md(b.get("label") or ""), _widget_key("band", f"{b_lo}-{b_hi}"),
              _on_band, (b_lo, b_hi), count=_count(n, show_counts),
              selected=applied and f.get("min_price") == b_lo and f.get("max_price") == b_hi)
    if has_range:
        lo, hi = int(math.floor(lo)), int(math.ceil(hi))
        # The range is in the key: a new pool with another range gets a new slider
        # seeded from the APPLIED filter, not the old slider's state (Streamlit shows
        # a widget whose bounds changed at its full range, while max_price still went
        # out -- and dragging the other handle then dropped that ceiling). price_rev:
        # a band link moved the range, so the slider follows it
        key = _widget_key("price", f"{lo}-{hi}-{sget('price_rev') or 0}")
        a = f.get("min_price") if f.get("min_price") is not None else lo
        b = f.get("max_price") if f.get("max_price") is not None else hi
        # rounded out, so a band's 49.99 shows as 50, not 49 (_on_price keeps 49.99)
        _seed(key, (_handle(a, lo, hi, math.floor), _handle(b, lo, hi, math.ceil)))
        st.slider("Price range", min_value=lo, max_value=hi, step=1, format="$%d",
                  key=key, on_change=_on_price, args=(key, lo, hi),
                  label_visibility="collapsed")
        share = _priced_share(facets, total)
        # On Fashion pages most rows have no listed price: say how many a range keeps
        st.caption(f"Only {share[0]:,} of {share[1]:,} items list a price; a price "
                   "range hides the rest." if share
                   else "A price range hides items without a listed price.")
    elif applied:
        # no slider to draw for this pool, but the filter still applies (escaped:
        # a markdown '$' pair would render as LaTeX)
        top = f.get("max_price")
        st.caption(escape_md(f"Price: ${f.get('min_price') or 0:g} to "
                             + (f"${top:g}" if top is not None else "any")))


def _checkbox_rows(kind: str, items: list, chosen: list, show_counts: bool, *,
                   full: bool = False) -> None:
    for key, label, n in items:
        wkey = _widget_key(kind, key)
        _seed(wkey, key in (chosen or []))
        # full: the API's cap is reached, so only a ticked box can still change
        st.checkbox(escape_md(label) + _count(n, show_counts), key=wkey,
                    on_change=_on_check, args=(kind, key, wkey),
                    disabled=full and key not in (chosen or []))


def _checkbox_section(title: str, kind: str, items: list, f: dict, show_counts: bool, *,
                      visible: int = 8, limit: int = 30, more_label: str = "See more",
                      hint: str = "", labels: dict | None = None,
                      max_picked: int | None = None) -> None:
    chosen = list(f.get(kind) or [])
    items = _with_chosen(items[:limit], chosen, labels)
    if not items:
        return
    _rail_heading(title, hint)
    full = max_picked is not None and len(chosen) >= max_picked
    if full:
        st.caption(f"Up to {max_picked} {title.lower()} at a time: untick one to pick "
                   "another.")
    # a ticked value stays in view: it is listed first when it would be under See more
    picked = [t for t in items if t[0] in chosen]
    rest = [t for t in items if t[0] not in chosen]
    head = picked + rest[:max(0, visible - len(picked))]
    tail = rest[max(0, visible - len(picked)):]
    _checkbox_rows(kind, head, chosen, show_counts, full=full)
    if tail:
        with st.expander(more_label):
            _checkbox_rows(kind, tail, chosen, show_counts, full=full)


def _swatch_css(items: list) -> str:
    """One custom property per colour checkbox, from the FIXED family map. Keys are
    digests, so no catalog text reaches the stylesheet."""
    rules = []
    for key, _, _ in items:
        value = COLOR_HEX.get(str(key).casefold())
        if value:
            rules.append(f".st-key-{_widget_key('color', key)}{{--sw:{value}}}")
    return f"<style>{''.join(rules)}</style>" if rules else ""


def _is_phone() -> bool:
    """A phone's User-Agent: the rail starts collapsed there. AppTest sends no headers
    (tests see desktop)."""
    try:
        agent = st.context.headers.get("User-Agent") or ""
    except Exception:  # noqa: BLE001 -- no request context (tests, bare mode)
        return False
    return bool(re.search(r"Mobi|Android", agent))


def focus_department(facets: dict, f: dict):
    """Whose sections the rail shows: the one picked department, Fashion when only a
    gender is picked (only Fashion rows carry one), else the API's primary_department
    (the department of half the on-target rows or more). None = no majority, or an API
    before 2026-10-27: no per-department section, and Fashion's own as before. Pure."""
    chosen = (f or {}).get("department") or []
    if len(chosen) == 1 and _text(chosen[0]):
        return _text(chosen[0])
    if (f or {}).get("gender") and not chosen:
        return "Fashion"
    return _text((facets or {}).get("primary_department")) or None


def visible_attributes(facets: dict, focus) -> list:
    """The facets.attributes sections to draw: well-formed, sendable (_attr_param_ok),
    the focus department's only (none without a focus: the shopper picks a department
    first), one per param. Pure."""
    if not focus:
        return []
    out, seen = [], set()
    for s in (facets or {}).get("attributes") or []:
        if not isinstance(s, dict) or not _attr_param_ok(s.get("param")):
            continue
        if s["param"] in seen or _text(s.get("department")).casefold() != focus.casefold():
            continue
        seen.add(s["param"])
        out.append(s)
    return out


def _attr_text(facets: dict) -> dict:
    """{param: {"label", "hint", "match", "values": {key: label}}} from every section of
    this pool (any department), merged into what this surface has seen before: a ticked
    value the current pool lacks -- or a whole section it lacks -- still reads as the
    API's words, not as its keys, and still says how its ticks combine (the API applies
    the param's rule whether or not the pool shows the section). Bounded by the API's
    fixed vocabularies."""
    known = {p: {"label": e.get("label", ""), "hint": e.get("hint", ""),
                 "match": e.get("match", ""), "values": dict(e.get("values") or {})}
             for p, e in (sget("attr_text") or {}).items() if isinstance(e, dict)}
    for s in (facets or {}).get("attributes") or []:
        if not isinstance(s, dict) or not _attr_param_ok(s.get("param")):
            continue
        entry = known.setdefault(s["param"],
                                 {"label": "", "hint": "", "match": "", "values": {}})
        if _text(s.get("label")):
            entry["label"], entry["hint"] = _text(s.get("label")), _text(s.get("hint"))
        if _text(s.get("match")):
            entry["match"] = _text(s.get("match"))
        for v in s.get("values") or []:
            if isinstance(v, dict) and _text(v.get("key")):
                entry["values"][str(v["key"])] = _text(v.get("label")) or str(v["key"])
    sset("attr_text", known)
    return known


def _attribute_section(section: dict, f: dict, show_counts: bool, labels: dict, *,
                       visible: int = 8, limit: int = 30) -> None:
    """One API-described section as checkboxes (8 in view, the rest under See more).
    Heading and hint through _rail_heading (esc), labels through escape_md, keys as
    digests: the labels are the API's fixed vocabulary, but a sink never trusts its
    source. Values keep the API's order (count, then vocabulary)."""
    param = section["param"]
    chosen = _attr_filters(f).get(param) or []   # each key once (_attr_filters)
    values, keys = [], set()
    for item in _facet_items(section.get("values"), keep_order=True):
        if item[0] not in keys:   # a repeated key would repeat a widget key: page down
            keys.add(item[0])
            values.append(item)
    items = _with_chosen(values[:limit], chosen, labels)
    if not items:
        return
    _rail_heading(_text(section.get("label")) or param, _text(section.get("hint")))
    if _match_all(section):
        _rail_note(MATCH_ALL_NOTE)
    max_picked = int(_num(section.get("max_picked")) or 0)
    full = max_picked > 0 and len(chosen) >= max_picked
    if full:
        # constant text and a number: the section's label is API text, and a caption is
        # a markdown sink (Brands says the same, _checkbox_section)
        st.caption(f"Up to {max_picked} at a time: untick one to pick another.")
    picked = [t for t in items if t[0] in chosen]
    rest = [t for t in items if t[0] not in chosen]
    head = picked + rest[:max(0, visible - len(picked))]
    tail = rest[max(0, visible - len(picked)):]

    def rows(group):
        for key, label, n in group:
            wkey = _widget_key(f"a_{param}", key)
            # ALWAYS the applied filter, never a kept widget value (not _seed): a
            # department click clears filters["attr"] without renewing the widget keys,
            # and a section drawn again for the same focus showed its old tick over a
            # page no longer filtered by it (2026-10-27 review: "Vegan" ticked, nothing
            # sent). Legal here: the widget is not created yet in this run
            st.session_state[wkey] = key in chosen
            st.checkbox(escape_md(label) + _count(n, show_counts), key=wkey,
                        on_change=_on_attr_check, args=(param, key, wkey),
                        disabled=full and key not in chosen)

    rows(head)
    if tail:
        total = int(_num(section.get("n_values")) or len(items))
        with st.expander(f"See more ({max(total, len(items)):,})"):
            rows(tail)


def _attribute_sections(facets: dict, f: dict, focus, show_counts: bool) -> None:
    """The focus department's sections, then any section the shopper ticked that is no
    longer drawn (the pool changed under the same query: its primary department moved,
    or the section's values left it). Those stay, ticked, so the page is never filtered
    by something the shopper cannot see or undo -- the _with_chosen rule, per section."""
    known = _attr_text(facets)
    drawn = set()
    for section in visible_attributes(facets, focus):
        _attribute_section(section, f, show_counts,
                           known.get(section["param"], {}).get("values", {}))
        drawn.add(section["param"])
    for param in sorted(set(_attr_filters(f)) - drawn):
        entry = known.get(param) or {}
        label = entry.get("label") or param.replace("_", " ").capitalize()
        _attribute_section({"param": param, "label": label, "hint": entry.get("hint", ""),
                            "match": entry.get("match", ""), "values": []},
                           f, show_counts, entry.get("values", {}))


def _n_applied(f: dict) -> int:
    n = sum(1 for kind in _FILTER_KINDS if f.get(kind))
    n += int(f.get("min_price") is not None or f.get("max_price") is not None)
    n += len(_attr_filters(f))
    return n + int(bool(f.get("min_rating")))


def filter_rail(facets: dict, total=None) -> None:
    """The refinement rail from the API's facets: Department (> Women / Men / Kids), the
    focus department's own sections (facets.attributes: Electronics Category / Item type
    / Works with / Condition, Grocery Category / Dietary), Category and Style (Fashion),
    Customer Reviews, Price, Brands, Colour, Fabric (Fashion). Sections the API does not
    send (an older API, a non-clothing pool) are left out.

    Facets are counted over the UNFILTERED ranked pool, so the options do not move as
    filters are picked (the API leaves out a value whose rows the ranking already ruled
    out). That is also why counts show only where they are exact -- the Department
    tree with nothing else picked -- or with Developer details on: after "Women" a
    "Black (21)" would still count men's items. Any change goes back to page 1 and
    the top. `total` = the pool's size, for the priced-items caption.

    Fashion's Category / Style / Fabric follow the focus too (2026-10-27): on a pool that
    is mostly Grocery ("gift for a coffee lover") they listed the Fashion minority's
    Jewelry and Tops. A section with a value picked is drawn whatever the focus, so a
    filter in force is always on screen."""
    f = sget("filters") or {}
    dev = _dev_details()
    others_applied = any(f.get(k) for k in _FILTER_KINDS if k not in ("department", "gender")) \
        or f.get("min_price") is not None or f.get("max_price") is not None \
        or bool(f.get("min_rating")) or bool(_attr_filters(f))
    n = _n_applied(f)
    label = "Filters" + (f" ({n} applied)" if n else "")
    focus = focus_department(facets, f)
    fashion = focus is None or focus.casefold() == "fashion"
    with st.container(key="cas_rail"):
        with st.expander(label, expanded=not _is_phone(), icon=":material/filter_list:",
                         key="cas_filters"):
            _department_section(facets, f, dev or not others_applied)
            _attribute_sections(facets, f, focus, dev)
            if fashion or f.get("ptype"):
                _link_section("Category", "ptype",
                              _facet_items(facets.get("product_types"), TYPE_LABELS), f,
                              dev, "‹ Any Category", TYPE_LABELS)
            if fashion or f.get("subtype"):
                _link_section("Style", "subtype",
                              _facet_items(facets.get("subtypes"), FINE_SUBTYPE_LABELS), f,
                              dev, "‹ Any Style", FINE_SUBTYPE_LABELS)
            _reviews_section(facets, f, dev)
            _price_section(facets, f, total, dev)
            stores = _facet_items(facets.get("stores"))
            n_stores = int(_num(facets.get("n_stores")) or len(stores))
            _checkbox_section("Brands", "store", stores, f, dev,
                              more_label=f"See more ({n_stores:,} brands)",
                              max_picked=MAX_STORES)
            colors = _facet_items(facets.get("colors"))
            if colors or f.get("color"):
                st.html(_swatch_css(_with_chosen(colors, f.get("color"))))
            _checkbox_section("Colour", "color", colors, f, dev, visible=12,
                              hint="Colour of the item as listed", labels=COLOR_LABELS)
            if fashion or f.get("material"):
                _checkbox_section("Fabric", "material",
                                  _facet_items(facets.get("materials")), f, dev,
                                  visible=10, limit=10)
            if filter_params(f, None):
                st.button("Clear filters", key=_widget_key("clear"), icon=":material/close:",
                          on_click=_on_clear, width="stretch")


def sort_select() -> None:
    key = _widget_key("sort")
    _seed(key, sget("sort") or "relevance")
    st.selectbox("Sort by", list(SORT_OPTIONS), key=key, label_visibility="collapsed",
                 format_func=lambda v: f"Sort by: {SORT_OPTIONS[v]}", width=230,
                 on_change=_on_sort, args=(key,))


def clear_filters_button() -> None:
    st.button("Clear filters", key=_widget_key("clear_empty"), icon=":material/close:",
              on_click=_on_clear)


# ---- product page ----------------------------------------------------------------------
_DESC_PREVIEW = 600
_DESC_MIN = 60   # the median Fashion description is 24 chars ("Zipper closure ...")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def about_parts(text) -> list:
    """'About this item' as bullet texts. Pure. The loader joined the listing's feature
    bullets with spaces, so they cannot be rebuilt; split ONLY on explicit separators
    (';', ' - ', sentence ends), else one paragraph. [] for nothing to show."""
    text = _text(text)
    if not text:
        return []
    for splitter in (lambda t: t.split(";"), lambda t: t.split(" - "),
                     lambda t: _SENTENCE_END.split(t)):
        parts = [p.strip(" -;") for p in splitter(text)]
        parts = [p for p in parts if len(p) >= 2]
        if len(parts) >= 2:
            return parts
    return [text]


def _fashion(product: dict) -> bool:
    return _text(product.get("bsns_vrtcl_name")).casefold() == "fashion"


def _gender_label(product: dict) -> str:
    for field in ("shopper_gender", "gender"):
        value = _text(product.get(field)).casefold()
        if value in GENDER_LABELS:
            return GENDER_LABELS[value]
    return ""


def _type_label(product: dict) -> str:
    return TYPE_LABELS.get(_text(product.get("product_type")).casefold(), "")


def _style_label(product: dict) -> str:
    fine = _text(product.get("fine_subtype"))
    if fine in FINE_SUBTYPE_LABELS:
        return FINE_SUBTYPE_LABELS[fine]
    # never an internal value (other-apparel, weak-*-word) or a subtype equal to its type
    return SUBTYPE_LABELS.get(_text(product.get("product_subtype")), "")


def breadcrumb_html(product: dict) -> str:
    """'Fashion › Women › Clothing › Dresses' (Fashion levels only; empty ones left
    out). Not clickable: it describes the item. Pure, escaped."""
    levels = [_text(product.get("bsns_vrtcl_name"))]
    if _fashion(product):
        levels += [_gender_label(product), _type_label(product), _style_label(product)]
    levels = [esc(x) for x in levels if x]
    return f'<div class="cas-crumbs">{" › ".join(levels)}</div>' if levels else ""


def details_table_html(product: dict) -> str:
    """'Product details': Department, Type, Style, Colour, Material, then the listing's
    own spec pairs. Occasion is left out: the rule-based tag misfires visibly (a beach
    maxi dress tagged office wear). Only non-empty values; every cell escaped. Pure."""
    rows = []
    if _fashion(product):
        rows += [("Department", _gender_label(product)), ("Type", _type_label(product)),
                 ("Style", _style_label(product))]
    rows += [("Colour", _text(product.get("color"))),
             ("Material", _text(product.get("material")))]
    for pair in _text(product.get("specs")).split("; "):
        name, sep, value = pair.partition(": ")
        if sep and name.strip() and value.strip() and len(rows) < 16:
            rows.append((name.strip(), value.strip()))
    cells = "".join(f"<tr><th>{esc(k)}</th><td>{esc(v)}</td></tr>" for k, v in rows if v)
    if not cells:
        return ""
    return f'<div class="cas-h3">Product details</div><table class="cas-attrs">{cells}</table>'


def detail_info_html(product: dict) -> str:
    """Sponsored, title, brand, stars and price for the product page. Pure; escaped.
    'Brand: X', never a store page we do not have."""
    title = _text(product.get("Product_title")) or "Untitled"
    name = brand_name(product)
    brand = f'<div class="cas-detail-brand">Brand: {esc(name)}</div>' if name else ""
    sponsored = '<div class="cas-spons">Sponsored</div>' if product.get("is_sponsored") else ""
    return (
        f'<div class="cas-detail">{sponsored}<h1 class="cas-detail-title">{esc(title)}</h1>'
        f"{brand}"
        f"{stars_html(product.get('average_rating'), product.get('rating_number'), detail=True)}"
        f"<hr>{price_html(product, detail=True)}<hr></div>"
    )


def _about_html(parts: list) -> str:
    items = "".join(f"<li>{esc(p)}</li>" for p in parts)
    return f'<div class="cas-about"><ul>{items}</ul></div>' if len(parts) > 1 else \
        f'<div class="cas-about">{esc(parts[0])}</div>'


def _render_about(product: dict) -> None:
    about = _text(product.get("prod_description")) or _text(product.get("gen_description"))
    if len(about) < _DESC_MIN:
        return   # "Zipper closure Hand Wash Only" reads as noise, not as a description
    st.html('<div class="cas-h3">About this item</div>')
    parts = about_parts(about)
    shown, used = [], 0
    for p in parts:
        if shown and used + len(p) > _DESC_PREVIEW:
            break
        shown.append(p if len(p) <= _DESC_PREVIEW else p[:_DESC_PREVIEW].rsplit(" ", 1)[0] + "…")
        used += len(p)
    st.html(_about_html(shown))
    if len(shown) < len(parts) or shown[-1].endswith("…"):
        with st.expander("Read the full description"):
            st.html(_about_html(parts))


def _render_variants(product: dict) -> None:
    """'Also available in': the card's collapsed variants, each opening its own page --
    only when the open product is one of them (a list for another product is stale)."""
    own = product.get("catalog_index")
    variants = [v for v in (sget("detail_variants") or [])
                if isinstance(v, dict) and v.get("catalog_index") is not None]
    if not any(v.get("catalog_index") == own for v in variants) or \
            not any(v.get("catalog_index") != own for v in variants):
        return
    st.html('<div class="cas-h3">Also available in</div>')
    labels = [(_text(v.get("color")) or _text(v.get("title")) or "Option")[:40]
              for v in variants]
    with st.container(horizontal=True, gap="small"):
        for i, (v, label) in enumerate(zip(variants, labels)):
            cid = v.get("catalog_index")
            if labels.count(label) > 1:   # same colour, another size: keep them apart
                label = f"{label} ({labels[:i + 1].count(label)})"
            st.button(escape_md(label), key=f"variant_{surface()}_{cid}",
                      type="secondary" if cid == own else "tertiary", disabled=cid == own,
                      on_click=_select_detail, args=(int(cid), None, variants))


def _feedback_row(product: dict) -> None:
    kid = product.get("catalog_index")
    query = sget("query", "")
    with st.container(horizontal=True, vertical_alignment="center", gap="small"):
        st.caption("Is this a good match for your search?")
        if st.button("👍", key=f"fb_up_{surface()}_{kid}", help="Good result"):
            post_feedback(query, product.get("Product_title", ""), +1, None,
                          product.get("reason", ""))
            st.toast("Thanks — feedback recorded.")
        if st.button("👎", key=f"fb_down_{surface()}_{kid}", help="Poor result"):
            post_feedback(query, product.get("Product_title", ""), -1, None,
                          product.get("reason", ""))
            st.toast("Thanks — feedback recorded.")


def _render_buy_box(product: dict) -> None:
    """Price, quantity, Add to cart, Buy Now. No stock, delivery, seller or returns
    lines: there is no such data. Buy Now = add the quantity and open the cart."""
    cid = product.get("catalog_index")
    if cid is None:
        return
    with st.container(border=True, key="cas_buybox"):
        st.html(price_html(product, detail=False))
        qty = st.selectbox("Quantity", list(range(1, 11)), key=ns(f"qty_{cid}"),
                           format_func=lambda n: f"Quantity: {n}",
                           label_visibility="collapsed")
        add_to_cart_button(product, key_ns="detail", wide=True, qty=qty)
        if st.button("Buy Now", key=f"buynow_detail_{cid}", width="stretch"):
            ui_cart.add(product, int(qty or 1))
            st.switch_page("pages/cart.py")
        n = ui_cart.in_cart(cid)
        if n:
            st.html(f'<div class="cas-incart">In your cart: {n}</div>')
        st.caption("Demo store: no checkout or payment.")


def _render_detail_card(product: dict) -> None:
    image, info, buy = st.columns([5, 5, 3], gap="large")
    with image:
        st.html(_image_html(product.get("img_url"), "cas-detail-img"))
    with info:
        st.html(detail_info_html(product))
        _render_variants(product)
        table = details_table_html(product)
        if table:
            st.html(table)
        _render_about(product)
        _dev_card_lines(product, reason_chars=None)
        _feedback_row(product)
    with buy:
        _render_buy_box(product)


def _carousel(items: list, key_prefix: str) -> None:
    """A row of mini cards that scrolls sideways inside itself (never the page)."""
    with st.container(key=f"cas_carousel_{key_prefix}", horizontal=True, gap="medium"):
        for item in items:
            render_mini(item, key_prefix=key_prefix, width=200)


def render_detail_page(*, show_recs: bool, show_back_button: bool = True,
                       back_label: str = "Back to results"):
    """Product page, reading only this surface's namespaced state: Back + breadcrumb,
    image | details | buy box, recommendations below. apply_scroll() already ran; an open
    from "Related items" scrolls to the suggestions here, once they are drawn."""
    idx = sget("detail_index")
    focus = sget("detail_focus")
    sset("detail_focus", None)   # one-shot, like the scroll request
    with st.container(horizontal=True, vertical_alignment="center", gap="medium"):
        if show_back_button:
            st.button(back_label, key=f"detail_back_{surface()}", type="tertiary",
                      icon=":material/chevron_left:", on_click=_select_results)
        crumbs = st.empty()

    resp = sget("detail_response")
    if resp is None or resp.get("product", {}).get("catalog_index") != idx:
        with st.spinner("Loading product..."):
            resp = fetch_product(idx, sget("query", ""), show_recs)
            sset("detail_response", resp)
    if not resp:
        st.warning("Could not load this product.")
        if focus:
            # no suggestions to land on: the top, where the warning is (the open
            # requested no scroll, so the view was still where the grid left it)
            request_scroll()
            apply_scroll(late=True)
        return

    product = resp.get("product", {})
    crumb = breadcrumb_html(product)
    if crumb:
        crumbs.html(crumb)
    _render_detail_card(product)
    # The anchor "Related items" lands on: drawn even when there is nothing to suggest,
    # so the shopper sees that line instead of the top of the page
    with st.container(key="cas_related"):
        _render_related(resp)
    if focus == "related":
        # After the anchor exists: the top-of-run apply_scroll() fired before the slow
        # /product fetch, and the script's retry window (~3 s) would have missed it
        request_scroll("cas_related")
        apply_scroll(late=True)


def _render_related(resp: dict) -> None:
    """'Pairs well with this' (or 'Similar items'), then 'Better-rated alternative'."""
    recs = resp.get("recommendations") or {}
    cross = recs.get("cross_sell", [])
    upsell = recs.get("upsell", [])
    if not (cross or upsell):
        st.caption("No related products to suggest for this item.")
        return

    if cross:
        st.html('<hr class="cas-hr">')
        # Embedding look-alikes are NOT complements, and there is no purchase data at
        # all (bought_together is NULL in the dump): never "bought together"
        similar = recs.get("cross_sell_source") == "fallback" or all(
            c.get("recommend_type") == "similar" for c in cross
        )
        st.html(heading_html("Similar items" if similar else "Pairs well with this"))
        rec_errors = resp.get("errors") or recs.get("errors") or []
        if rec_errors and _dev_details():
            codes = ", ".join(str(e.get("code", "?")) for e in rec_errors)
            # 'partial' = the query guard screened some phrases; the LLM's answer WAS used
            state = ("partly applied" if all(e.get("code") == "partial" for e in rec_errors)
                     else "unavailable")
            st.caption(f"_cross-sell LLM {state} (error {codes})_")
        _carousel(cross, "d_cross")
    if upsell:
        st.html('<hr class="cas-hr">')
        st.html(heading_html("Better-rated alternative"))
        _carousel(upsell, "d_up")
