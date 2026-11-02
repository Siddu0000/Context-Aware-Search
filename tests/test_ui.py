"""Offline tests for the Streamlit UI (ui.py, ui_common.py, ui_cart.py, pages/).

No network, no LLM, no running API: every page runs in-process under
streamlit.testing.v1.AppTest with `requests.get` / `requests.post` replaced by a fake
API that records each call. Runs under pytest (`python -m pytest tests/test_ui.py`)
or plainly (`python tests/test_ui.py`).

What is pinned here is what a restyle or a Streamlit upgrade could silently break:
- the one-shot scroll script (Streamlit 1.58 keeps stMain's scrollTop across reruns,
  so without it Next leaves the shopper at the bottom of page 2) is emitted exactly
  once after a page change / View / Back, and never on an unrelated rerun;
- Enter submits the search form, and a submit costs exactly one request;
- catalog (seller), model and shopper text is inert in every sink (grid, product
  page, recommendation strip, cart): no live link, no HTML, no LaTeX from '$';
- images are https only and cannot break out of the src attribute;
- the reranker's notes and '% match' are Developer-details only;
- filters, sort and page numbers send the API contract's params, and the filter rail
  disappears when the API sends no facets (an older API, or a bundle);
- the marketplace-style shell (2026-10-19): the department scope and the sub-nav filter
  the matched pool (follow_up, 0 LLM calls), the rail's new sections send the C2 params
  (gender, subtype, color, material, repeated store, price bands), Buy Now, the cart's
  quantity stepper, the weak-match banner, variant swatches, and no badge or label the
  data cannot back (no membership marks, delivery or stock claims);
- 2026-10-27: the "Related items" button on every grid card (one /product call, ONE
  scroll script aimed at the suggestions after they are drawn, Back to the card), the
  per-department rail sections from facets.attributes (built against SYNTHETIC payloads
  in the 2026-10-27 contract; the API half lands separately), the sub-nav showing only
  shortcuts the pool can fill, the `brand` byline, and the sidebar theme mirror;
- 2026-10-30: a match "all" section (Dietary) says its ticks must all hold, and the
  synthetic rail fixtures are held to app/attributes.py's registry and vocabulary.
"""

import contextlib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import requests  # noqa: E402

EVIL = "Plan ($200-$249.99) [deal](https://evil.example) *X* #1 <img src=x>"


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


def _apptest():
    try:
        from streamlit.testing.v1 import AppTest
    except ImportError:
        _skip("streamlit is not installed")
    return AppTest


# ------------------------------------------------------------------ fake API
def _product(cid, **over):
    p = {"Product_title": f"Item {cid}", "catalog_index": cid, "price": 9.99,
         "average_rating": 4.2, "rating_number": 198, "store": "Acme",
         "img_url": f"https://img.example/{cid}.jpg",
         "reason": "(rerank disabled)", "final_score": 80.0, "rerank_score": 80}
    p.update(over)
    return p


def search_payload(page=1, *, per_page=3, pages=3, facets="absent", results=None, **over):
    """A /search response in the API's shape; facets='absent' = an API without them."""
    if results is None:
        results = [_product(page * 100 + i) for i in range(per_page)]
    data = {"user_query": "q", "interpreted_as": ["x"], "constraints": [],
            "is_recipe": False, "bundle_type": None, "translator_mode": "query_expansion",
            "rerank_requested": True, "rerank_succeeded": True, "no_match": False,
            "results": results, "sponsored": [], "recommendations": {}, "errors": [],
            "page": page, "page_size": per_page, "total_results": per_page * pages,
            "total_pages": pages, "has_prev": page > 1, "has_next": page < pages,
            "latency_ms": {}}
    if facets != "absent":
        data["facets"] = facets
        data["total_unfiltered"] = per_page * pages
    data.update(over)
    return data


FACETS = {"departments": {"Fashion": 20, "Electronics": 7},
          "product_types": {"accessory": 12, "jewelry": 3},
          "price": {"min": 4.5, "max": 120.0, "n_priced": 19},
          "rating": {"4": 15, "3": 22, "2": 25, "1": 26}}


class _Resp:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


class FakeAPI:
    """Stands in for the HTTP layer; `calls` = [(endpoint, params or json)]."""

    def __init__(self, search=None, keyword=None, product=None, chat=None):
        self.calls = []
        self.search = search or (lambda p: search_payload(int(p.get("page", 1))))
        self.keyword = keyword or (lambda p: {"results": [_product(1)], "num_matched": 1})
        self.product = product or (lambda p: {
            "product": _product(int(p["catalog_index"])), "errors": [],
            "recommendations": {"cross_sell": [], "upsell": []}})
        self.chat = chat or (lambda body: {"reply": "ok", "action": "reply"})

    def get(self, url, params=None, timeout=None, **_):
        endpoint = url.rsplit("/", 1)[-1]
        params = dict(params or {})
        self.calls.append((endpoint, params))
        handler = {"search": self.search, "keyword_search": self.keyword,
                   "product": self.product}[endpoint]
        return _Resp(handler(params))

    def post(self, url, json=None, timeout=None, **_):
        endpoint = url.rsplit("/", 1)[-1]
        self.calls.append((endpoint, dict(json or {})))
        return _Resp(self.chat(json or {}) if endpoint == "chat" else {})

    def count(self, endpoint):
        return sum(1 for e, _ in self.calls if e == endpoint)

    def last(self, endpoint):
        return [p for e, p in self.calls if e == endpoint][-1]


@contextlib.contextmanager
def fake_http(api):
    old = (requests.get, requests.post)
    requests.get, requests.post = api.get, api.post
    try:
        yield api
    finally:
        requests.get, requests.post = old


def _page(name, state=None):
    at = _apptest().from_file(str(ROOT / "pages" / name), default_timeout=60)
    for k, v in (state or {}).items():
        at.session_state[k] = v
    return at


def _run(at):
    at.run()
    assert not at.exception, at.exception
    return at


def _html(at):
    return [e.proto.body for e in at.get("html")]


def scripts(at):
    return [b for b in _html(at) if "cas-scroll" in b]


def _button(at, key):
    found = [b for b in at.button if b.key == key]
    assert found, (key, [b.key for b in at.button])
    return found[0]


def _button_prefix(at, prefix):
    found = [b for b in at.button if (b.key or "").startswith(prefix)]
    assert found, (prefix, [b.key for b in at.button])
    return found[0]


def _texts(at):
    """Every markdown-rendered string on the page (markdown, captions, labels). Button
    labels count: a product title is now the link to its page (a markdown sink)."""
    out = [m.value for m in at.markdown] + [c.value for c in at.caption]
    out += [w.value for w in at.warning] + [i.value for i in at.info]
    out += [b.label for b in at.button] + [b.help for b in at.button if b.help]
    out += [c.label for c in at.checkbox]
    return out


def _plain(label):
    """A rail link's label as the shopper reads it: no bold marker, no '(n)' count, no
    markdown escapes (labels are escape_md'd seller or API text)."""
    text = re.sub(r"\s*\(\d[\d,]*\)$", "", (label or "").replace("**", "")).strip()
    return re.sub(r"\\(.)", r"\1", text)


def _rail(at, text):
    """The rail link (a tertiary button keyed f_*) whose label reads `text`."""
    found = [b for b in at.button if "__f_" in (b.key or "") and _plain(b.label) == text]
    assert found, (text, [(b.key, b.label) for b in at.button if "__f_" in (b.key or "")])
    return found[0]


def _checkbox(at, text):
    found = [c for c in at.checkbox if _plain(c.label) == text]
    assert found, (text, [c.label for c in at.checkbox])
    return found[0]


# ------------------------------------------------------------------ scroll
def test_scroll_one_shot_on_ai_search():
    """Port of the browser-verified scratch AppTest: the script is emitted once after
    Next / a page number / Previous / View / Back, never on an unrelated rerun or an
    add-to-cart, and Back targets the card that opened the product."""
    with fake_http(FakeAPI()) as api:
        at = _run(_page("ai_search.py", {"ai__query": "q",
                                         "ai__last_response": search_payload(1)}))
        assert len(scripts(at)) == 0
        assert api.count("search") == 1   # params not yet fetched for this state

        _button(at, "ai__pg_next").click()
        _run(at)
        assert at.session_state["ai__page"] == 2 and len(scripts(at)) == 1
        assert api.count("search") == 2 and api.last("search")["page"] == 2

        _run(at)   # an unrelated rerun: no scroll, no request
        assert len(scripts(at)) == 0 and api.count("search") == 2

        _button(at, "ai__pg_3").click()
        _run(at)
        assert at.session_state["ai__page"] == 3 and len(scripts(at)) == 1

        _button(at, "ai__pg_prev").click()
        _run(at)
        assert at.session_state["ai__page"] == 2 and len(scripts(at)) == 1

        _button_prefix(at, "cart_grid_").click()
        _run(at)
        assert len(scripts(at)) == 0, "add to cart must not move the view"
        assert at.session_state["cart"], "the item reached the shared cart"
        assert api.count("search") == 4

        view = _button_prefix(at, "grid__")
        idx = int(view.key.rsplit("_", 1)[1])
        view.click()
        _run(at)
        assert at.session_state["ai__view"] == "detail" and len(scripts(at)) == 1
        assert api.count("product") == 1

        _button(at, "detail_back_ai").click()
        _run(at)
        s = scripts(at)
        assert at.session_state["ai__view"] == "search" and len(s) == 1
        assert f"'grid__{idx}'" in s[0], s[0]
        assert api.count("search") == 4, "Back re-uses the results it left"


def test_scroll_script_is_constant_text():
    """Only a key reduced to [A-Za-z0-9_-] is interpolated into the script."""
    with fake_http(FakeAPI()):
        at = _run(_page("ai_search.py", {"ai__query": "q",
                                         "ai___scroll_request": "grid__1'];alert(1);//"}))
        s = scripts(at)
        assert len(s) == 1 and "alert(1)" not in s[0] and "'];" not in s[0], s
        assert "'grid__1alert1'" in s[0], s[0]


def test_back_scrolls_the_whole_card_below_the_band():
    """Browser-measured (integration, 2026-10-15): centring the View button left the
    card's image under the sticky band, and on the Assistant the stick-to-bottom
    container then pulled Back down to the chat box. The script scrolls the CARD to
    its scroll-margin and holds it against a jump to the bottom until the shopper acts."""
    import ui_common
    js = ui_common._SCROLL_JS
    assert "closest('[class*=\"st-key-card_\"]')" in js
    assert "scrollIntoView({{block: 'start'}})" in js
    assert "'wheel', 'touchstart', 'keydown', 'mousedown'" in js
    # Integrate 2026-10-19, browser-measured on the Assistant: Back to top went to 0 and
    # the stick-to-bottom container put it back 16 ms later, then about once a second;
    # Back to a card held 4 s, then fell to the chat box. An upward wheel just before and
    # just after the FIRST jump releases the container (both stayed put for 9 s), and
    # the top is held like a card
    rendered = js.format(nonce=1, target="null")
    assert "{{" not in rendered and "deltaY: -120" in rendered, rendered
    assert "wheelUp(m);\n    put();\n    wheelUp(m);" in rendered, rendered
    assert "if (released) { put(); return; }" in rendered      # one release per emission
    # the jump (and its wheels) before hold() listens: our own wheel must not end it
    assert "jump(toCard, m);\n        hold(toCard, m);" in rendered, rendered
    assert "var toTop = function () { m.scrollTop = 0; };\n    jump(toTop, m);\n" \
        "    hold(toTop, m);" in rendered, rendered
    css = ui_common._SHOP_CSS
    # 60px header + 56px search band + 16px (browser-measured 2026-10-19; was 160 for
    # the taller rounded band of 2026-10-15)
    assert '[class*="st-key-card_"]{scroll-margin-top:72px}' in css
    assert '[class*="st-key-card_"]{scroll-margin-top:132px}' in css


# ------------------------------------------------------------------ search form
def _form_proto(at, form_id):
    """The st.form block's proto (Block subclasses keep the block's own sub-message)."""
    for node in at._tree:
        if getattr(node, "type", None) == "form" and node.proto.form.form_id == form_id:
            return node.proto.form
    raise AssertionError(f"no form {form_id}")


def test_enter_submits_search_once():
    """The search box is an st.form with enter_to_submit: Enter (= the submit button)
    runs ONE search for the typed text, and nothing re-requests it afterwards."""
    for page, form, endpoint in (("ai_search.py", "ai__search_form", "search"),
                                 ("keyword_search.py", "keyword__search_form",
                                  "keyword_search")):
        with fake_http(FakeAPI()) as api:
            at = _run(_page(page))
            assert _form_proto(at, form).enter_to_submit is True
            assert api.calls == []
            at.text_input(key=form.replace("search_form", "q_input")).set_value(
                "  red   shoes ")
            [b for b in at.button if b.label == "Search"][0].click()
            _run(at)
            assert api.count(endpoint) == 1, api.calls
            sent = api.last(endpoint)
            assert sent["query"] == "red   shoes" and sent["page"] == 1, sent
            # the band is sticky: a search sent from the bottom of a page lands at the
            # top in THIS run (review 2026-10-15), and only once
            assert len(scripts(at)) == 1
            _run(at)
            assert api.count(endpoint) == 1, "a rerun must not search again"
            assert len(scripts(at)) == 0


# ------------------------------------------------------------------ escaping
def _evil_product(cid=7, **over):
    return _product(cid, Product_title=EVIL, store=EVIL, color=EVIL, material=EVIL,
                    categ_lvl2_name=EVIL, bsns_vrtcl_name="Fashion",
                    prod_description=EVIL + " " + "word " * 200, **over)


def _assert_inert(at):
    bodies = [b for b in _html(at) if "cas-scroll" not in b]
    joined = "\n".join(bodies)
    assert "&lt;img src=x&gt;" in joined, joined[:500]
    assert "<img src=x" not in joined and "<a " not in joined
    assert "$200-$249.99" in joined   # st.html parses no LaTeX: '$' stays literal
    for text in _texts(at):
        # escape_md leaves '\<img' (a literal '<'); an UNescaped '<' could open HTML
        assert "](" not in text and "evil.example" not in text, text
        assert not re.search(r"(?<!\\)<", text), text


def _evil_facets():
    """Every rail sink fed seller or API text (2026-10-27, safety.md "To close it"): a
    store name, a colour and style label, and an attribute section's heading, hint and
    value label. Department Fashion, so the section is the focus department's."""
    return dict(
        FACETS_C2, primary_department="Fashion",
        stores=[{"name": EVIL, "n": 3}, {"name": "Acme", "n": 2}], n_stores=2,
        colors=[{"key": "black", "label": EVIL, "n": 2}],
        subtypes=[{"key": "dresses", "label": EVIL, "n": 2}],
        attributes=[{"param": "occasion_tag", "label": EVIL, "department": "Fashion",
                     "match": "any", "hint": EVIL, "n_values": 1,
                     "values": [{"key": "k1", "label": EVIL, "n": 3}]}])


def test_seller_text_is_inert_in_grid_detail_mini_and_cart():
    rec = _evil_product(8, recommend_reason=EVIL)
    card = _evil_product(7, brand=EVIL, variants=[
        {"catalog_index": 7, "title": EVIL, "color": None, "color_family": "black",
         "img_url": "javascript:alert(1)"},
        {"catalog_index": 70, "title": EVIL, "color": EVIL, "color_family": EVIL,
         "img_url": "https://img.example/70.jpg"}])
    api = FakeAPI(
        search=lambda p: search_payload(1, results=[card], pages=1, facets=_evil_facets()),
        product=lambda p: {"product": _evil_product(int(p["catalog_index"]), brand=EVIL),
                           "errors": [],
                           "recommendations": {"cross_sell": [rec], "upsell": [
                               _evil_product(9)]}})
    with fake_http(api):
        at = _run(_page("ai_search.py", {"ai__query": EVIL}))
        _assert_inert(at)                          # grid, rail (+ the results header)
        assert any("__f_a_occasion_tag_" in (c.key or "") for c in at.checkbox), \
            "the EVIL attribute section was drawn (so its sinks were checked)"
        assert any(EVIL.replace("<", "&lt;").replace(">", "&gt;") in b
                   for b in _html(at) if "cas-rail-h" in b), "heading + hint escaped"
        # the API's KEY goes out, never a label
        [c for c in at.checkbox if "__f_a_occasion_tag_" in (c.key or "")][0].check()
        _run(at)
        assert api.last("search")["occasion_tag"] == "k1", api.last("search")
        _button(at, "grid__7").click()
        _run(at)
        assert at.session_state["ai__view"] == "detail"
        _assert_inert(at)                          # product page + mini strips
        assert any("d_cross_8" == b.key for b in at.button)
        assert any((b.key or "").startswith("variant_ai_70") for b in at.button), \
            "the EVIL variant was drawn (so its label was checked)"

        _button(at, "cart_detail_7").click()       # the cart is shared across pages
        _run(at)
        cart = at.session_state["cart"]
    at = _run(_page("cart.py", {"cart": cart}))
    _assert_inert(at)


def test_safe_img_rejects_non_https_and_quotes():
    from ui_common import card_html, safe_img

    good = "https://m.media-amazon.com/images/I/81BUCfhDMFL._AC_UL1500_.jpg"
    assert safe_img(good) == good
    assert safe_img("https://x.example/a.jpg?w=1&h=2") == "https://x.example/a.jpg?w=1&amp;h=2"
    for bad in ("javascript:alert(1)", "JaVaScRiPt:alert(1)", "http://x.example/a.jpg",
                "data:image/png;base64,AAAA", '"https://x.example/a.jpg"',
                'https://x.example/a.jpg" onerror="alert(1)', "https://x.example/a'.jpg",
                "https://x.example/a b.jpg", "https://x.example/<b>", "https://x\\y",
                "https://x.example/\x00a.jpg", "", None, float("nan"), 12):
        assert safe_img(bad) == "", bad
        html = card_html({"Product_title": "t", "img_url": bad})
        assert "cas-img empty" in html and "<img" not in html, html


# ------------------------------------------------------------------ dev-only notes
def test_reason_and_match_are_developer_details_only():
    junk = _product(5, reason="screen lens cleaner, unrelated", rerank_score=30,
                    final_score=33.0)
    api = FakeAPI(search=lambda p: search_payload(1, results=[junk], pages=1))
    with fake_http(api):
        at = _run(_page("ai_search.py", {"ai__query": "gift for a watch collector"}))
        shown = " ".join(_texts(at) + _html(at))
        assert "unrelated" not in shown and "% match" not in shown
        [t for t in at.toggle if t.label == "Developer details"][0].set_value(True)
        _run(at)
        captions = [c.value for c in at.caption]
        assert any("unrelated" in c for c in captions), captions
        assert any("33% match" in c for c in captions), captions


def test_sponsored_label_and_estimated_price():
    import ui_cart
    from ui_common import card_html, cart_line_html, price_html

    ad = card_html({"Product_title": "t", "is_sponsored": True, "sponsor": "Acme"})
    assert ">Sponsored<" in ad
    assert "Sponsored" not in card_html({"Product_title": "t"})

    est = {"Product_title": "t", "price": None, "price_estimate": 105.0}
    assert "≈ $105.00" in price_html(est) and ">est.<" in price_html(est)
    assert price_html({"price": float("nan")}).endswith("Price not listed</div>")
    assert "not in the total" in cart_line_html(est, 2)

    class _State(dict):
        pass

    # ui_cart totals: an estimate (or a NaN price) is never in the real total
    old = ui_cart.st.session_state
    try:
        ui_cart.st.session_state = _State()
        ui_cart.add({"catalog_index": 1, "price": 10.0})
        ui_cart.add({"catalog_index": 2, "price": None, "price_estimate": 50.0})
        ui_cart.add({"catalog_index": 3, "price": float("nan")})
        total, priced, unpriced = ui_cart.total_price()
        assert (total, priced, unpriced) == (10.0, 1, 2)
        assert ui_cart.estimated_total() == (50.0, 1)
    finally:
        ui_cart.st.session_state = old


def test_stars_html():
    from ui_common import stars_html

    assert 'style="--pct:84%"' in stars_html(4.2, 198) and "(198)" in stars_html(4.2, 198)
    assert stars_html(None, None) == '<div class="cas-rating"></div>'
    assert stars_html(float("nan"), float("nan")) == '<div class="cas-rating"></div>'
    assert "12 ratings" in stars_html(None, 12)
    assert stars_html(7, 3) == '<div class="cas-rating"><span>3 ratings</span></div>'


# ------------------------------------------------------------------ filters + sort
def test_filters_sort_and_pages_send_the_contract_params():
    """The rail is links (Department, Category) and a rating link list now, the
    marketplace way; every click still sends the same contract params, page 1 and one
    scroll to the top."""
    api = FakeAPI(search=lambda p: search_payload(int(p.get("page", 1)), facets=FACETS))
    with fake_http(api):
        at = _run(_page("ai_search.py", {"ai__query": "gift"}))
        labels = [b.label for b in at.button if "__f_" in (b.key or "")]
        # counts only where exact: the Department tree with nothing else picked
        assert "Fashion (20)" in labels and "Accessories" in labels, labels
        assert api.count("search") == 1

        _rail(at, "Fashion").click()
        _run(at)
        sent = api.last("search")
        assert sent["department"] == "Fashion" and sent["page"] == 1, sent
        assert len(scripts(at)) == 1
        assert _rail(at, "Fashion").label.startswith("**Fashion**")   # the chosen path
        assert not [b for b in at.button
                    if "__f_" in (b.key or "") and _plain(b.label) == "Electronics"]
        assert at.session_state["ai__scope"] == "Fashion", "the scope select follows"

        _rail(at, "Accessories").click()
        _run(at)
        assert api.last("search")["ptype"] == "accessory"

        [s for s in at.selectbox if s.label == "Sort by"][0].set_value("price_asc")
        _run(at)
        assert api.last("search")["sort"] == "price_asc"

        at.slider[0].set_value((10, 120))
        _run(at)
        sent = api.last("search")
        assert sent["min_price"] == 10 and "max_price" not in sent, sent

        _rail(at, ":orange[★★★★]:gray[★] & Up").click()
        _run(at)
        assert api.last("search")["min_rating"] == 4

        _button(at, "ai__pg_next").click()        # a page change keeps every filter
        _run(at)
        sent = api.last("search")
        assert sent["page"] == 2 and sent["department"] == "Fashion" \
            and sent["sort"] == "price_asc" and sent["min_rating"] == 4, sent

        _button_prefix(at, "grid__").click()      # the rail is not drawn on the
        _run(at)                                  # product page...
        _button(at, "detail_back_ai").click()
        _run(at)                                  # ...and its filters survive it
        n = api.count("search")
        assert api.last("search")["department"] == "Fashion"
        assert _rail(at, "Fashion").label.startswith("**Fashion**")

        clear = [b for b in at.button if b.label == "Clear filters"]
        clear[0].click()
        _run(at)
        sent = api.last("search")
        assert api.count("search") == n + 1 and sent["page"] == 1
        assert not {"department", "ptype", "sort", "min_price", "max_price",
                    "min_rating"} & set(sent), sent
        assert at.session_state["ai__scope"] == "All"


def test_sort_note():
    from ui_common import sort_note
    assert sort_note("relevance", FACETS, 172) is None and sort_note(None, FACETS, 172) is None
    assert sort_note("rating", FACETS, 172) == "Closest matches first, then by customer rating."
    note = sort_note("price_asc", FACETS, 172)
    assert note.startswith("Closest matches first, then by listed price. Only 19 of 172 "
                           "items list a price"), note
    for facets, total in (({"price": {"n_priced": 172}}, 172), ({}, None), (None, 172),
                          ({"price": {"n_priced": float("nan")}}, 172)):
        assert sort_note("price_desc", facets, total) == \
            "Closest matches first, then by listed price.", (facets, total)


def test_price_sort_and_slider_say_how_many_items_list_a_price():
    """Oct 15 review: on Fashion pages ~6% of rows list a price, so a price sort read
    $2.75 ... $120.00, then '≈ $28.32 (est.)', '≈ $10.99 (est.)' -- a broken-looking
    sort -- and the slider silently hid ~95% of the results."""
    def answer(p):
        return search_payload(1, facets=FACETS, total_unfiltered=172,
                              sort=p.get("sort", "relevance"))

    api = FakeAPI(search=answer)
    with fake_http(api):
        at = _run(_page("ai_search.py", {"ai__query": "beach wedding outfit"}))
        captions = [c.value for c in at.caption]
        assert "Only 19 of 172 items list a price; a price range hides the rest." in captions
        assert not any(c.startswith("Closest matches first") for c in captions), captions
        [s for s in at.selectbox if s.label == "Sort by"][0].set_value("price_asc")
        _run(at)
        captions = [c.value for c in at.caption]
        assert any(c.startswith("Closest matches first, then by listed price. Only 19 of "
                                "172 items list a price") for c in captions), captions


def test_priced_share_divides_rows_by_rows():
    """Integrate 2026-10-19: facets.price.n_priced counts pool ROWS, total_unfiltered
    counts CARDS after the variant collapse; the caption reads pool_rows when the API
    sends it (an older API's total_unfiltered was rows)."""
    def answer(p):
        return search_payload(1, facets=FACETS, total_unfiltered=172, pool_rows=190,
                              sort=p.get("sort", "relevance"))

    with fake_http(FakeAPI(search=answer)):
        at = _run(_page("ai_search.py", {"ai__query": "beach wedding outfit"}))
        captions = [c.value for c in at.caption]
        assert "Only 19 of 190 items list a price; a price range hides the rest." in \
            captions, captions
        [s for s in at.selectbox if s.label == "Sort by"][0].set_value("price_asc")
        _run(at)
        assert any(c.startswith("Closest matches first, then by listed price. Only 19 of "
                                "190 items") for c in (c.value for c in at.caption))


def test_follow_ups_send_follow_up_and_a_submit_does_not():
    """Oct 15 review: a degraded run is never cached, so every rail or pager click re-ran
    both LLM calls and moved the pool. A change to the results on screen now sends
    follow_up=true (the API reuses that pool briefly); a Search submit never does, so
    pressing Search still retries the AI stages."""
    api = FakeAPI(search=lambda p: search_payload(int(p.get("page", 1)), facets=FACETS))
    with fake_http(api):
        at = _run(_page("ai_search.py"))
        at.text_input(key="ai__q_input").set_value("gift")
        search = [b for b in at.button if b.label == "Search"][0]
        search.click()
        _run(at)
        assert api.count("search") == 1 and "follow_up" not in api.last("search")
        [s for s in at.selectbox if s.label == "Sort by"][0].set_value("price_asc")
        _run(at)
        assert api.last("search")["follow_up"] == "true", api.last("search")
        _button(at, "ai__pg_next").click()
        _run(at)
        sent = api.last("search")
        assert sent["follow_up"] == "true" and sent["page"] == 2, sent
        _run(at)
        assert api.count("search") == 3, "follow_up is not part of what was fetched"
        [b for b in at.button if b.label == "Search"][0].click()   # the same text again
        _run(at)
        assert api.count("search") == 4 and "follow_up" not in api.last("search")


def test_rail_shows_the_applied_filters_when_the_pool_changes():
    """Review 2026-10-15: the same query can come back with another pool (AI ranking or
    Sponsored toggled -- both in the cache key; no seed on Databricks; a degraded run is
    never cached). The slider then showed its new FULL range while max_price=50 still
    went out, and dragging the other handle dropped that ceiling; a picked department
    the new pool lacked vanished from the rail while it still emptied the page."""
    moved = {"departments": {"Electronics": 7}, "product_types": {},
             "price": {"min": 30.0, "max": 300.0, "n_priced": 9},
             "rating": {"4": 3, "3": 5, "2": 6, "1": 6}}
    state = {"facets": FACETS}
    api = FakeAPI(search=lambda p: search_payload(1, facets=state["facets"]))
    with fake_http(api):
        at = _run(_page("ai_search.py", {"ai__query": "gift"}))
        _rail(at, "Fashion").click()
        _run(at)
        at.slider[0].set_value((4, 50))
        _run(at)
        assert api.last("search")["max_price"] == 50

        state["facets"] = moved
        [t for t in at.toggle if t.label == "AI ranking"][0].set_value(False)
        _run(at)
        sent = api.last("search")
        assert sent["rerank"] == "false" and sent["max_price"] == 50 \
            and sent["department"] == "Fashion", sent
        # the rail says what is applied: the department still listed and chosen, the
        # ceiling kept
        assert _rail(at, "Fashion").label.startswith("**Fashion**")
        assert at.slider[0].value == (30, 50), at.slider[0].value
        at.slider[0].set_value((40, 50))   # the other handle: the ceiling stays
        _run(at)
        sent = api.last("search")
        assert sent["min_price"] == 40 and sent["max_price"] == 50, sent

        # a pool with no price range to draw still shows the price filter it applies
        state["facets"] = dict(moved, price={"min": None, "max": None, "n_priced": 0})
        [t for t in at.toggle if t.label == "Sponsored results"][0].set_value(False)
        _run(at)
        assert not at.slider and api.last("search")["max_price"] == 50
        assert any("Price" in c.value and "50" in c.value for c in at.caption), \
            [c.value for c in at.caption]


def test_no_facets_means_no_rail_and_no_sort():
    """An API without facets (before 2026-10-15) and a bundle (facets null) keep the
    plain grid: no rail, no sort."""
    slot = _product(100, source_intent="flour [x](https://evil.example)",
                    alternatives=[_product(150), _product(151)])
    for payload in (search_payload(1),
                    search_payload(1, facets=None, bundle_type="recipe",
                                   results=[slot, _product(101)])):
        with fake_http(FakeAPI(search=lambda p, d=payload: d)):
            at = _run(_page("ai_search.py", {"ai__query": "q"}))
            assert not at.checkbox and not at.slider and not at.radio
            assert "Sort by" not in [s.label for s in at.selectbox]
            assert not [k for k in api_keys(at) if "__f_" in k]
            assert set(api_keys(at)) >= {"grid__100", "grid__101"}


def test_bundle_card_keeps_option_tabs():
    slot = _product(100, source_intent="flour", alternatives=[_product(150), _product(151)])
    payload = search_payload(1, facets=None, bundle_type="recipe", results=[slot], pages=1)
    with fake_http(FakeAPI(search=lambda p: payload)):
        at = _run(_page("ai_search.py", {"ai__query": "ingredients to make pancakes"}))
        assert [t.label for t in at.tabs] == ["Option 1", "Option 2", "Option 3"]
        assert {"grid__100", "grid__150", "grid__151"} <= set(api_keys(at))
        # each option has its own way to its suggestions (2026-10-27)
        assert {"related__100", "related__150", "related__151"} <= set(api_keys(at))
        assert any("3 brands to choose from" in c.value for c in at.caption)


def test_fetch_error_is_escaped_and_not_retried_on_every_rerun():
    calls = []

    def down(url, params=None, timeout=None, **_):
        calls.append(url)
        raise requests.ConnectionError(
            "Max retries exceeded with url: /search?query=[deal](https://evil.example)")

    old = requests.get
    requests.get = down
    try:
        at = _run(_page("ai_search.py", {"ai__query": "red shoes"}))
        errors = [e.value for e in at.error]
        assert errors and "Could not reach the search service" in errors[0]
        assert "](" not in errors[0], errors
        _run(at)
        assert len(calls) == 1, "a failed search is retried by pressing Search, not per rerun"
    finally:
        requests.get = old


def api_keys(at):
    return [b.key for b in at.button if b.key]


def test_filtered_to_nothing_shows_the_message():
    empty = search_payload(1, facets=FACETS, results=[], total_results=0, total_pages=1,
                           message="No products match these filters.")
    with fake_http(FakeAPI(search=lambda p: empty)):
        at = _run(_page("ai_search.py", {"ai__query": "q"}))
        assert any("No products match these filters." in i.value for i in at.info)
        assert any(b.label == "Clear filters" for b in at.button)


# ------------------------------------------------------------------ keyword + assistant
def test_keyword_search_pagination():
    def kw(p):
        page = int(p.get("page", 1))
        return {"results": [_product(page * 100 + i) for i in range(3)], "no_match": False,
                "num_matched": 131_132, "page": page, "page_size": 3, "total_results": 150,
                "total_pages": 50, "has_prev": page > 1, "has_next": page < 50}

    with fake_http(FakeAPI(keyword=kw)) as api:
        at = _run(_page("keyword_search.py", {"keyword__query": "leather watch"}))
        assert api.count("keyword_search") == 1
        keys = api_keys(at)
        assert {"keyword__pg_1", "keyword__pg_2", "keyword__pg_3", "keyword__pg_50"} <= \
            set(keys) and "keyword__pg_4" not in keys, keys
        _button(at, "keyword__pg_next").click()
        _run(at)
        assert api.last("keyword_search")["page"] == 2 and len(scripts(at)) == 1
        _button(at, "keyword__pg_50").click()
        _run(at)
        assert api.last("keyword_search")["page"] == 50 and len(scripts(at)) == 1
        view = _button_prefix(at, "grid__")
        view.click()
        _run(at)
        assert len(scripts(at)) == 1
        _button(at, "detail_back_keyword").click()
        _run(at)
        assert f"'{view.key}'" in scripts(at)[0]

    # An API without paging: one page, no pager, and the match count stays visible
    with fake_http(FakeAPI(keyword=lambda p: {"results": [_product(1)], "num_matched": 9})):
        at = _run(_page("keyword_search.py", {"keyword__query": "belt"}))
        assert not [k for k in api_keys(at) if "__pg_" in k]
        assert any("9" in c.value and "contain" in c.value for c in at.caption)


def test_assistant_defaults_to_12_and_scrolls_on_view_and_back():
    reply = {"reply": "Here you go", "action": "search", "new_topic": True,
             "search_query": "desk chair", "results": [_product(i) for i in range(12)]}
    with fake_http(FakeAPI(chat=lambda body: reply)) as api:
        at = _run(_page("shopping_assistant.py"))
        assert [s.value for s in at.slider if s.label == "Products to show"] == [12]
        at.chat_input[0].set_value("a desk chair")
        _run(at)
        assert api.last("chat")["top_k"] == 12
        view = _button(at, "grid_chatpanel_3")
        view.click()
        _run(at)
        assert at.session_state["assistant__view"] == "detail" and len(scripts(at)) == 1
        _button(at, "detail_back_assistant").click()
        _run(at)
        s = scripts(at)
        assert len(s) == 1 and "'grid_chatpanel_3'" in s[0], s


def test_chat_bubbles_are_literal():
    history = [{"role": "user", "content": "red shoes [deal](https://evil.example)"},
               {"role": "assistant", "content": "Sure!\n**bold** <b>x</b>"}]
    at = _run(_page("shopping_assistant.py", {"assistant__chat_history": history}))
    md = [m.value for m in at.markdown]
    assert not any("](" in m or "<b>" in m for m in md), md
    assert any("Sure\\!  \n\\*\\*bold\\*\\*" in m for m in md), md


# ------------------------------------------------------------------ pure helpers
def test_page_window():
    from ui_common import page_window

    assert page_window(1, 1) == [1]
    assert page_window(1, 3) == [1, 2, 3]
    assert page_window(7, 15) == [1, "…", 5, 6, 7, 8, 9, "…", 15]
    assert page_window(1, 15) == [1, 2, 3, "…", 15]
    assert page_window(15, 15) == [1, "…", 13, 14, 15]
    assert page_window(4, 15) == [1, 2, 3, 4, 5, 6, "…", 15]
    assert page_window(5, 15) == [1, 2, 3, 4, 5, 6, 7, "…", 15]   # a 1-page gap shows it
    assert page_window(99, 5) == [1, 2, 3, 4, 5]                   # clamped
    assert page_window(0, 0) == [1]


def test_filter_params():
    from ui_common import filter_params

    assert filter_params({}, "relevance") == {} and filter_params(None, None) == {}
    assert filter_params({"department": ["Grocery", "Fashion"], "ptype": [],
                          "min_price": None, "max_price": 50, "min_rating": 0},
                         "rating") == {"sort": "rating", "department": "Fashion,Grocery",
                                       "max_price": 50}


def test_no_page_link_anywhere():
    """st.page_link raised KeyError 'url_pathname' outside a full page context."""
    for path in [ROOT / "ui.py", ROOT / "ui_common.py", ROOT / "ui_cart.py",
                 *sorted((ROOT / "pages").glob("*.py"))]:
        assert "page_link(" not in path.read_text(encoding="utf-8"), path


def test_entry_point_boots_with_top_nav_and_footer():
    with fake_http(FakeAPI()):
        at = _run(_apptest().from_file(str(ROOT / "ui.py"), default_timeout=60))
        assert any("research licence" in b for b in _html(at))
        assert any(b.label == "Search" for b in at.button)


def test_light_theme_on_both_launch_paths():
    """config.toml for `streamlit run`; the same flags for the Databricks App, which
    does not stage the .streamlit folder."""
    try:
        import tomllib
    except ImportError:
        _skip("tomllib needs Python 3.11+")
    import importlib.util

    conf = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8"))
    assert conf["theme"]["base"] == "light" and conf["client"]["toolbarMode"] == "minimal"
    spec = importlib.util.spec_from_file_location(
        "run_app", ROOT / "databricks" / "app_bundle" / "run_app.py")
    run_app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_app)
    flags = dict(zip(run_app.STREAMLIT_THEME_FLAGS[::2], run_app.STREAMLIT_THEME_FLAGS[1::2]))

    def flat(table, prefix):
        # nested tables are dotted flags: [theme.sidebar] backgroundColor ->
        # --theme.sidebar.backgroundColor (2026-10-27: it was skipped, and the sidebar
        # colour existed on one launch path only)
        for key, value in table.items():
            if isinstance(value, dict):
                yield from flat(value, f"{prefix}{key}.")
            else:
                yield f"{prefix}{key}", str(value).lower() if isinstance(value, bool) \
                    else str(value)

    expected = dict(flat(conf["theme"], "--theme."))
    assert "--theme.sidebar.backgroundColor" in expected
    themed = {k: v for k, v in flags.items() if k.startswith("--theme.")}
    # both ways: a key on one launch path only is drift
    assert themed == expected, (set(themed.items()) ^ set(expected.items()))
    assert flags["--client.toolbarMode"] == "minimal"


def test_math_isnan_guard_in_price():
    from ui_common import price_label

    assert price_label({"price": float("nan"), "price_estimate": 3}) == "≈ $3.00 (est.)"
    assert price_label({"price": "n/a"}) is None


# ------------------------------------------------------------------ marketplace shell (2026-10-19)
# The C2 facets (GET /search, post-cache): lists of {key,label,n}, stores by name
FACETS_C2 = dict(
    FACETS,
    departments={"Fashion": 20},
    genders=[{"key": "women", "label": "Women", "n": 14},
             {"key": "men", "label": "Men", "n": 4},
             {"key": "unisex", "label": "Unisex / unspecified", "n": 2}],
    subtypes=[{"key": "dresses", "label": "Dresses", "n": 9},
              {"key": "tops_tees", "label": "Tops & tees", "n": 5}],
    colors=[{"key": "black", "label": "Black", "hex": "#111111", "n": 6},
            {"key": "multicolor", "label": "Multicolor", "hex": None, "n": 2}],
    materials=[{"key": "linen", "label": "Linen", "n": 3}],
    stores=[{"name": f"Store {i}", "n": 20 - i} for i in range(10)]
    + [{"name": "Topsville, Inc.", "n": 1}],
    n_stores=95,
    # the API's band shape (app/main._price_bands): a ceiling is the edge less a cent
    price={"min": 4.5, "max": 120.0, "n_priced": 19,
           "bands": [{"label": "Up to $25", "min": None, "max": 24.99, "n": 7},
                     {"label": "$25 to $50", "min": 25, "max": 49.99, "n": 0},
                     {"label": "$50 to $100", "min": 50, "max": 99.99, "n": 3}]},
)


def test_new_rail_sections_send_the_c2_params():
    """Department > Women/Men (gender), Style (subtype), Colour, Fabric, Brands (a
    REPEATED store param: names contain commas) and price bands, each a post-cache
    follow-up of the results on screen."""
    api = FakeAPI(search=lambda p: search_payload(1, facets=FACETS_C2))
    with fake_http(api):
        at = _run(_page("ai_search.py", {"ai__query": "beach wedding outfit",
                                         "ai__last_response": search_payload(1)}))
        labels = [_plain(b.label) for b in at.button if "__f_" in (b.key or "")]
        for want in ("Fashion", "Women", "Men", "Unisex / unspecified", "Accessories",
                     "Dresses", "Tops & tees", "Up to $25", "$50 to $100"):
            assert want in labels, (want, labels)
        assert "$25 to $50" not in labels, "a band with no rows is not offered"
        # no counts outside the Department tree (they would count the unfiltered pool)
        assert "Women (14)" in [b.label for b in at.button]
        assert "Dresses" in [b.label for b in at.button]

        _rail(at, "Women").click()
        _run(at)
        sent = api.last("search")
        assert sent["gender"] == "women" and sent["follow_up"] == "true", sent
        # the select follows the department; the gender filters these results only
        assert at.session_state["ai__scope"] == "Fashion"
        assert at.session_state["ai__scope_gender"] is None

        _rail(at, "Dresses").click()
        _run(at)
        assert api.last("search")["subtype"] == "dresses"

        _checkbox(at, "Black").check()
        _run(at)
        assert api.last("search")["color"] == "black"

        _checkbox(at, "Linen").check()
        _run(at)
        assert api.last("search")["material"] == "linen"

        # 8 brands in view, the rest under "See more"; a store with a comma survives
        assert any(e.label == "See more (95 brands)" for e in at.expander), \
            [e.label for e in at.expander]
        _checkbox(at, "Topsville, Inc.").check()
        _run(at)
        _checkbox(at, "Store 0").check()
        _run(at)
        sent = api.last("search")
        assert sent["store"] == ["Store 0", "Topsville, Inc."], sent

        _rail(at, "Up to $25").click()
        _run(at)
        sent = api.last("search")
        assert sent["max_price"] == 24.99 and "min_price" not in sent, sent
        assert at.slider[0].value == (4, 25), "the slider follows the band, rounded out"
        _rail(at, "‹ Any price").click()
        _run(at)
        assert not {"min_price", "max_price"} & set(api.last("search"))

        # 'Any Department' clears the levels below it, and only those
        _rail(at, "‹ Any Department").click()
        _run(at)
        sent = api.last("search")
        assert not {"department", "gender", "subtype"} & set(sent), sent
        assert sent["color"] == "black" and sent["store"] == ["Store 0", "Topsville, Inc."]


def test_colour_swatches_come_from_the_fixed_map():
    """The swatch colour reaches CSS only from ui_common.COLOR_HEX, keyed by a digest:
    an API hex or a catalog value never does."""
    evil = dict(FACETS_C2, colors=[{"key": "black", "label": "Black", "hex": "red;}", "n": 1},
                                   {"key": "x</style><b>", "label": "X", "n": 1}])
    with fake_http(FakeAPI(search=lambda p: search_payload(1, facets=evil))):
        at = _run(_page("ai_search.py", {"ai__query": "q"}))
        styles = [b for b in _html(at) if "--sw:" in b]
        assert len(styles) == 1 and "--sw:#111111" in styles[0], styles
        assert "red;}" not in styles[0] and "</style><b>" not in styles[0]


def test_counts_show_with_developer_details():
    with fake_http(FakeAPI(search=lambda p: search_payload(1, facets=FACETS_C2))):
        at = _run(_page("ai_search.py", {"ai__query": "q"}))
        [t for t in at.toggle if t.label == "Developer details"][0].set_value(True)
        _run(at)
        labels = [b.label for b in at.button] + [c.label for c in at.checkbox]
        assert "Dresses (9)" in labels and "Black (6)" in labels, labels


def test_old_api_facets_still_render_the_rail():
    """An API without the C2 keys (genders, subtypes, colors, stores, bands) still gets
    Department, Category, reviews and price; Women/Men/Kids leave the sub-nav. Since
    2026-10-27 so does a department the pool lacks (FACETS: Fashion and Electronics),
    whose shortcut could only empty the page."""
    with fake_http(FakeAPI(search=lambda p: search_payload(1, facets=FACETS))):
        at = _run(_page("ai_search.py", {"ai__query": "q"}))
        labels = [_plain(b.label) for b in at.button]
        assert "Fashion" in labels and "Accessories" in labels
        assert not at.checkbox and at.slider
        # the sub-nav is drawn after the run's fetch: it reads THIS response
        subnav = [b.label for b in at.button if "subnav_" in (b.key or "")]
        assert subnav == ["All", "Fashion", "Electronics"], subnav


def test_scope_and_subnav_filter_the_matched_pool():
    """The department select submits with the query (a filter on the matched pool, the
    existing `department` param); a sub-nav shortcut replaces the department/gender
    filters as a follow-up (0 LLM calls), and before a search only sets the scope."""
    facets = dict(FACETS_C2, departments={"Fashion": 20, "Electronics": 7})
    api = FakeAPI(search=lambda p: search_payload(1, facets=facets))
    with fake_http(api):
        at = _run(_page("ai_search.py"))
        assert [b.label for b in at.button if "subnav_" in (b.key or "")] == \
            ["All", "Fashion", "Women", "Men", "Kids", "Electronics", "Grocery"]
        _button(at, "ai__subnav_3").click()          # Men, before any search
        _run(at)
        assert api.calls == [] and at.session_state["ai__scope"] == "Fashion"
        at.text_input(key="ai__q_input").set_value("linen shirt")
        [b for b in at.button if b.label == "Search"][0].click()
        _run(at)
        sent = api.last("search")
        assert sent["department"] == "Fashion" and sent["gender"] == "men", sent
        assert "follow_up" not in sent
        # on results: the pool's departments only (no Grocery here) and, since the
        # 2026-10-27 review, the pool's genders only (FACETS_C2 offers women, men and
        # unisex: Kids, like a missing department, could only empty the page)
        assert [b.label for b in at.button if "subnav_" in (b.key or "")] == \
            ["All", "Fashion", "Women", "Men", "Electronics"]

        _button(at, "ai__subnav_5").click()          # Electronics, on the results
        _run(at)
        sent = api.last("search")
        assert sent["department"] == "Electronics" and "gender" not in sent \
            and sent["follow_up"] == "true" and sent["page"] == 1, sent
        assert len(scripts(at)) == 1

        at.selectbox(key="ai__scope").set_value("All")
        [b for b in at.button if b.label == "Search"][0].click()
        _run(at)
        sent = api.last("search")
        assert "department" not in sent and "gender" not in sent, sent


def _submit(at, api, text):
    at.text_input(key="ai__q_input").set_value(text)
    [b for b in at.button if b.label == "Search"][0].click()
    _run(at)
    return api.last("search")


def test_a_gender_picked_on_results_never_filters_the_next_search():
    """2026-10-19 review: a rail 'Women' became the scope's hidden gender, and every later
    search went out as gender=women under a select reading 'Fashion' ("men's linen shirt"
    got women's items or nothing). A submit sends the select's department only (ui_spec
    3.2); a Women / Men / Kids shortcut picked BEFORE any search rides on that one."""
    api = FakeAPI(search=lambda p: search_payload(1, facets=FACETS_C2))
    with fake_http(api):
        at = _run(_page("ai_search.py"))
        sent = _submit(at, api, "summer dress")
        assert "gender" not in sent and "department" not in sent, sent
        _rail(at, "Women").click()                       # the rail, on results
        _run(at)
        assert api.last("search")["gender"] == "women"
        sent = _submit(at, api, "men's linen shirt")
        assert sent["department"] == "Fashion" and "gender" not in sent, sent
        assert at.selectbox(key="ai__scope").value == "Fashion"
        _button(at, "ai__subnav_2").click()              # the sub-nav's Women, on results
        _run(at)
        assert api.last("search")["gender"] == "women"
        sent = _submit(at, api, "hiking boots")
        assert sent["department"] == "Fashion" and "gender" not in sent, sent

        at = _run(_page("ai_search.py"))                 # a new session
        _button(at, "ai__subnav_3").click()              # Men, before any search
        _run(at)
        assert _submit(at, api, "linen shirt")["gender"] == "men"
        sent = _submit(at, api, "linen trousers")
        assert sent["department"] == "Fashion" and "gender" not in sent, sent


def test_subnav_only_where_the_api_applies_filters():
    """2026-10-19 review: the sub-nav was drawn on every response, so on a bundle or a
    no-match page (the API ignores filters there) a shortcut sent a follow-up, lit up and
    moved the scope over a page that never changed. It follows the rail's rule now, drawn
    after the run's fetch: in the submit run it showed the PREVIOUS page's shortcuts."""
    slot = _product(1, source_intent="flour", alternatives=[_product(50), _product(51)])
    bundle = search_payload(1, results=[slot], pages=1, facets=None, bundle_type="recipe",
                            is_recipe=True)
    no_match = search_payload(1, results=[], pages=1, facets=None, no_match=True,
                              message="No matching products found.")
    state = {"payload": search_payload(1, facets=FACETS_C2)}
    api = FakeAPI(search=lambda p: state["payload"])

    def shortcuts(at):
        return [b.label for b in at.button if "subnav_" in (b.key or "")]

    # a flat Fashion-only page: since 2026-10-27 only the shortcuts that can show
    # something (FACETS_C2 departments = Fashion; genders women, men and unisex, so no
    # Kids: a gender shortcut needs its value in the facet, as the rail's gender links)
    fashion_page = ["All", "Fashion", "Women", "Men"]
    with fake_http(api):
        at = _run(_page("ai_search.py"))
        assert len(shortcuts(at)) == 7                   # before any search: the scope
        _submit(at, api, "summer dress")
        assert shortcuts(at) == fashion_page, shortcuts(at)
        for text, payload in (("ingredients to make pancakes", bundle), ("asdfgh", no_match)):
            state["payload"] = payload
            _submit(at, api, text)                       # the submit run itself
            assert shortcuts(at) == [], (text, shortcuts(at))
            n = api.count("search")
            _run(at)
            assert shortcuts(at) == [] and api.count("search") == n, text
        state["payload"] = search_payload(1, facets=FACETS_C2)
        _submit(at, api, "linen shirt")
        assert shortcuts(at) == fashion_page, shortcuts(at)


def test_the_department_above_a_picked_gender_goes_up_a_level():
    """2026-10-19 review: at Fashion > Women the tree draws 'Fashion' as a plain parent,
    and a click there deselected the department and the whole path (scope 'All'). ui_spec
    4.3: a level sets that filter and clears the levels below it."""
    facets = dict(FACETS_C2, departments={"Fashion": 20, "Electronics": 7})
    api = FakeAPI(search=lambda p: search_payload(1, facets=facets))
    with fake_http(api):
        at = _run(_page("ai_search.py", {"ai__query": "q",
                                         "ai__last_response": search_payload(1)}))
        _rail(at, "Fashion").click()
        _run(at)
        _rail(at, "Women").click()
        _run(at)
        sent = api.last("search")
        assert (sent["department"], sent["gender"]) == ("Fashion", "women"), sent
        assert not _rail(at, "Fashion").label.startswith("**")   # drawn as the parent
        _rail(at, "Fashion").click()
        _run(at)
        sent = api.last("search")
        assert sent["department"] == "Fashion" and "gender" not in sent, sent
        assert at.session_state["ai__scope"] == "Fashion"
        assert _rail(at, "Fashion").label.startswith("**Fashion**")
        _rail(at, "Fashion").click()                     # now the picked level: a toggle
        _run(at)
        sent = api.last("search")
        assert "department" not in sent and "gender" not in sent, sent
        assert at.session_state["ai__scope"] == "All"


class _Refused(_Resp):
    """A 4xx from the API: raise_for_status raises like requests does, with the response."""

    def raise_for_status(self):
        resp = requests.Response()
        resp.status_code = 422
        raise requests.HTTPError(
            "422 Client Error: Unprocessable Entity for url: /search?query=[x](https://e.x)",
            response=resp)


def test_brands_stop_at_the_api_cap_and_a_refused_filter_keeps_the_page():
    """2026-10-19 review: the rail offered 30 brands while the API takes 20 (a 422 above
    that), and the 21st tick wiped the page under 'Could not reach the search service'
    with no Clear filters left. The rail stops at the cap; a follow-up the API refuses
    keeps the page and says so."""
    import ui_common
    facets = dict(FACETS_C2, stores=[{"name": f"Brand {i:02d}", "n": 40 - i}
                                     for i in range(30)], n_stores=30)
    api = FakeAPI(search=lambda p: search_payload(1, facets=facets))

    def brand_boxes(at):
        return [c for c in at.checkbox if "__f_store_" in (c.key or "")]

    with fake_http(api):
        at = _run(_page("ai_search.py", {"ai__query": "q",
                                         "ai__last_response": search_payload(1)}))
        assert ui_common.MAX_STORES == 20 and len(brand_boxes(at)) == 30
        for i in range(ui_common.MAX_STORES):
            _checkbox(at, f"Brand {i:02d}").check()
            _run(at)
        assert len(api.last("search")["store"]) == ui_common.MAX_STORES
        boxes = brand_boxes(at)
        assert len(boxes) == 30 and sum(c.value for c in boxes) == 20
        assert all(c.disabled for c in boxes if not c.value), "no 21st tick"
        assert not any(c.disabled for c in boxes if c.value), "a ticked brand can go"
        assert any("Up to 20 brands at a time" in c.value for c in at.caption)
        _checkbox(at, "Brand 03").uncheck()
        _run(at)
        assert not any(c.disabled for c in brand_boxes(at))

    class API(FakeAPI):
        def get(self, url, params=None, timeout=None, **_):
            if (params or {}).get("color"):              # this API refuses the colour
                self.calls.append(("search", dict(params)))
                return _Refused({"detail": "color: unknown value"})
            return super().get(url, params=params, timeout=timeout)

    api = API(search=lambda p: search_payload(1, facets=FACETS_C2))
    with fake_http(api):
        at = _run(_page("ai_search.py", {"ai__query": "q",
                                         "ai__last_response": search_payload(1)}))
        cards = {k for k in api_keys(at) if k.startswith("grid__")}
        _checkbox(at, "Black").check()
        _run(at)
        assert api.last("search")["follow_up"] == "true"
        assert not at.error, [e.value for e in at.error]
        assert any("could not be applied" in w.value for w in at.warning)
        assert {k for k in api_keys(at) if k.startswith("grid__")} == cards, "page kept"
        assert any(b.label == "Clear filters" for b in at.button)
        n = api.count("search")
        _run(at)
        assert api.count("search") == n, "not retried on every rerun"
        _checkbox(at, "Black").uncheck()
        _run(at)
        assert not any("could not be applied" in w.value for w in at.warning)


def test_a_price_band_ceiling_survives_the_other_handle():
    """2026-10-19 review: the API's band '$50 to $100' is 50 / 99.99. The whole-dollar
    slider showed 99, and nudging only the LOW handle re-sent max_price=99, dropping the
    $99.01-99.99 rows the band had kept."""
    api = FakeAPI(search=lambda p: search_payload(1, facets=FACETS_C2))
    with fake_http(api):
        at = _run(_page("ai_search.py", {"ai__query": "q",
                                         "ai__last_response": search_payload(1)}))
        _rail(at, "$50 to $100").click()
        _run(at)
        sent = api.last("search")
        assert (sent["min_price"], sent["max_price"]) == (50, 99.99), sent
        assert at.slider[0].value == (50, 100), "rounded out, not cut to 99"
        assert _rail(at, "$50 to $100").label.startswith("**")
        at.slider[0].set_value((51, 100))                # the low handle only
        _run(at)
        sent = api.last("search")
        assert (sent["min_price"], sent["max_price"]) == (51, 99.99), sent
        at.slider[0].set_value((51, 80))                 # now the ceiling itself
        _run(at)
        sent = api.last("search")
        assert (sent["min_price"], sent["max_price"]) == (51, 80), sent
        at.slider[0].set_value((51, 120))                # to the end: no ceiling
        _run(at)
        assert "max_price" not in api.last("search")


def test_weak_match_banner():
    weak = {"band0": 2, "best_rerank": 48.0, "weak": True}
    for mq, shown in ((weak, True), (dict(weak, weak=False), False),
                      (dict(weak, unjudged=True), False), (None, False)):
        payload = search_payload(1, facets=FACETS, **({"match_quality": mq} if mq else {}))
        with fake_http(FakeAPI(search=lambda p, d=payload: d)):
            at = _run(_page("ai_search.py", {"ai__query": "dress [x](https://evil.example)"}))
            banner = [b for b in _html(at) if "No exact matches for" in b]
            assert bool(banner) is shown, (mq, banner)
            if banner:
                assert "Showing related items." in banner[0] and "<a " not in banner[0]
                assert "[x](https://evil.example)" in banner[0]   # literal, in st.html
    # 2026-10-19 review: match_quality reads the UNFILTERED pool, so a filter that empties
    # the page kept a weak flag; "Showing related items" over no items is not said
    empty = search_payload(1, facets=FACETS, results=[], total_results=0, total_pages=1,
                           message="No products match these filters.", match_quality=weak)
    with fake_http(FakeAPI(search=lambda p: empty)):
        at = _run(_page("ai_search.py", {"ai__query": "silk saree"}))
        assert not [b for b in _html(at) if "No exact matches for" in b]
        assert any("No products match these filters." in i.value for i in at.info)


def test_variant_swatches_and_also_available_in():
    own = _product(7, color="Black")
    own["variants"] = [
        {"catalog_index": 7, "title": "t", "color": "Black", "color_family": "black"},
        {"catalog_index": 8, "title": "t", "color": "Navy Blue", "color_family": "navy"},
        {"catalog_index": 9, "title": "t", "color": "Red <b>", "color_family": "red"},
        {"catalog_index": 10, "title": "t", "color": "Red <b>", "color_family": "red"},
    ]
    from ui_common import card_html, variants_html
    html_ = variants_html(own)
    assert "+2 colours" in html_ and "#1B2A4A" in html_ and "#C40000" in html_, html_
    assert "<b>" not in html_, "colour names are not printed into the card"
    sizes = dict(own, variants=[own["variants"][0], dict(own["variants"][0], catalog_index=11)])
    assert "+1 option<" in variants_html(sizes), "a same-colour variant is not a colour"
    assert variants_html(_product(1)) == "" and "cas-variants" not in card_html(_product(1))
    api = FakeAPI(search=lambda p: search_payload(1, results=[own], pages=1))
    with fake_http(api):
        at = _run(_page("ai_search.py", {"ai__query": "q"}))
        assert any("+2 colours" in b for b in _html(at))
        _button(at, "grid__7").click()
        _run(at)
        keys = api_keys(at)
        assert {"variant_ai_8", "variant_ai_9", "variant_ai_10"} <= set(keys), keys
        labels = [_plain(b.label) for b in at.button if (b.key or "").startswith("variant_")]
        assert "Red <b> (1)" in labels and "Red <b> (2)" in labels, labels
        _button(at, "variant_ai_8").click()
        _run(at)
        assert at.session_state["ai__detail_index"] == 8
        _button(at, "detail_back_ai").click()
        _run(at)
        assert "'grid__7'" in scripts(at)[0], "Back still lands on the card"


def test_a_page_never_shows_another_products_variants():
    """2026-10-19 review: the list of the last card WITH variants survived Back, so a plain
    card's page (and a recommendation's) offered that product's colours, each button
    opening the other product."""
    own = _product(7, color="Black")
    own["variants"] = [
        {"catalog_index": 7, "title": "t", "color": "Black", "color_family": "black"},
        {"catalog_index": 8, "title": "t", "color": "Navy Blue", "color_family": "navy"},
    ]
    product = lambda p: {"product": _product(int(p["catalog_index"])), "errors": [],  # noqa: E731
                         "recommendations": {"cross_sell": [_product(30)], "upsell": []}}
    api = FakeAPI(search=lambda p: search_payload(1, results=[own, _product(20)], pages=1),
                  product=product)

    def variant_keys(at):
        return [k for k in api_keys(at) if k.startswith("variant_")]

    with fake_http(api):
        at = _run(_page("ai_search.py", {"ai__query": "q"}))
        _button(at, "grid__7").click()
        _run(at)
        assert variant_keys(at) == ["variant_ai_7", "variant_ai_8"], variant_keys(at)
        _button(at, "d_cross_30").click()            # a recommendation: no variants
        _run(at)
        assert at.session_state["ai__detail_index"] == 30 and variant_keys(at) == []
        _button(at, "detail_back_ai").click()
        _run(at)
        _button(at, "grid__20").click()              # a plain card after one with variants
        _run(at)
        assert at.session_state["ai__detail_index"] == 20 and variant_keys(at) == []
        assert not any("Also available in" in b for b in _html(at))
        assert at.session_state["ai__detail_variants"] == []   # reset on every open
        # a stale list in state (any other path) is not drawn on a product outside it
        _button(at, "detail_back_ai").click()
        _run(at)
        _button(at, "grid__7").click()
        _run(at)
        at.session_state["ai__detail_index"] = 20
        _run(at)
        assert variant_keys(at) == []


def test_buy_now_adds_the_quantity_and_opens_the_cart():
    import streamlit

    opened = []
    old = streamlit.switch_page
    streamlit.switch_page = lambda page, **_: opened.append(str(page))
    try:
        with fake_http(FakeAPI()):
            at = _run(_page("ai_search.py", {"ai__query": "q", "ai__view": "detail",
                                             "ai__detail_index": 5}))
            at.selectbox(key="ai__qty_5").set_value(3)
            _run(at)
            _button(at, "buynow_detail_5").click()
            _run(at)
            assert at.session_state["cart"]["5"]["qty"] == 3
            assert opened == ["pages/cart.py"], opened
            _button(at, "cart_detail_5").click()        # Add to Cart takes the qty too
            _run(at)
            assert at.session_state["cart"]["5"]["qty"] == 6
            assert any("In your cart: 6" in b for b in _html(at))
    finally:
        streamlit.switch_page = old


def test_cart_quantity_stepper_and_subtotal():
    cart = {"1": {"qty": 2, "product": {"catalog_index": 1, "Product_title": "A",
                                        "price": 10.0}},
            "2": {"qty": 1, "product": {"catalog_index": 2, "Product_title": "B",
                                        "price": None, "price_estimate": 19.39}}}
    at = _run(_page("cart.py", {"cart": cart}))
    subtotal = [b for b in _html(at) if "cas-subtotal" in b]
    assert subtotal and "Subtotal (2 of 3 items): <b>$20.00</b>" in subtotal[0], subtotal
    assert any("19.39 more for 1 item" in c.value for c in at.caption)
    checkout = [b for b in at.button if b.label == "Proceed to checkout"]
    assert checkout and checkout[0].disabled
    _button(at, "cartplus_1").click()
    _run(at)
    assert at.session_state["cart"]["1"]["qty"] == 3
    _button(at, "cartminus_2").click()                  # at 1, minus removes the line
    _run(at)
    assert "2" not in at.session_state["cart"]
    _button(at, "cartrm_1").click()
    _run(at)
    assert at.session_state["cart"] == {}
    assert any("Your cart is empty" in b for b in _html(at))


# Words a marketplace look-alike must not claim: we have no membership programme,
# delivery, stock, bestseller, coupon or purchase data (ui_spec section 9). Neither
# marketplace whose look the skin borrows is named: the layout (2026-10-19) or the
# eBay-style palette (2026-10-26) -- only the dataset citation names one
_FORBIDDEN = re.compile(r"prime|best ?seller|choice|in stock|deliver|coupon|bought in|deal|"
                        r"amazon|ebay", re.I)


def test_no_badge_or_claim_the_data_cannot_back():
    from ui_common import (LICENCE_HTML, badges_html, card_html, cart_line_html,
                           detail_info_html, details_table_html)

    rows = [_product(1, is_sponsored=True, average_rating=4.9, rating_number=812,
                     bsns_vrtcl_name="Fashion", product_type="apparel",
                     fine_subtype="dresses", shopper_gender="women", color="Blue",
                     material="Linen", specs="Fit Type: Classic Fit; Neck Style: Round Neck"),
            _product(2, price=None, price_estimate=12.5, average_rating=None),
            _product(3, price=None)]
    for row in rows:
        for chunk in (card_html(row), detail_info_html(row), badges_html(row),
                      cart_line_html(row, 2), details_table_html(row)):
            assert not _FORBIDDEN.search(chunk), chunk
    assert "Highly rated" in badges_html(rows[0])
    assert badges_html(_product(4, average_rating=5.0, rating_number=3)) == "", \
        "a 5.0 from 3 ratings is noise, not a badge"
    # The footer's only mention of the marketplace is the dataset citation
    assert not _FORBIDDEN.search(LICENCE_HTML.replace("Amazon Reviews 2023", ""))
    assert "research licence" in LICENCE_HTML and "not affiliated" in LICENCE_HTML
    with fake_http(FakeAPI()):
        at = _run(_apptest().from_file(str(ROOT / "ui.py"), default_timeout=60))
        shown = " ".join(_texts(at) + _html(at)).replace("Amazon Reviews 2023", "")
        assert not _FORBIDDEN.search(shown), _FORBIDDEN.search(shown)


def test_product_page_details_and_pairs_well_with_this():
    from ui_common import breadcrumb_html, details_table_html

    row = _product(5, bsns_vrtcl_name="Fashion", product_type="apparel",
                   fine_subtype="dresses", product_subtype="dress", shopper_gender="women",
                   color="Blue", material="Linen", occasion="office wear",
                   specs="Fit Type: Classic Fit; Neck Style: Round <b>")
    assert breadcrumb_html(row) == \
        '<div class="cas-crumbs">Fashion › Women › Clothing › Dresses</div>'
    table = details_table_html(row)
    for cell in ("<th>Department</th><td>Women</td>", "<th>Style</th><td>Dresses</td>",
                 "<th>Fit Type</th><td>Classic Fit</td>", "Round &lt;b&gt;"):
        assert cell in table, (cell, table)
    assert "office" not in table, "occasion tags misfire visibly: never shown"
    internal = dict(row, fine_subtype=None, product_subtype="other-apparel")
    assert "other-apparel" not in breadcrumb_html(internal) + details_table_html(internal)
    rec = _product(8, recommend_reason="Goes with your search")
    api = FakeAPI(product=lambda p: {"product": row, "errors": [], "recommendations": {
        "cross_sell": [rec], "upsell": [_product(9)], "cross_sell_source": "llm"}})
    with fake_http(api):
        at = _run(_page("ai_search.py", {"ai__query": "q", "ai__view": "detail",
                                         "ai__detail_index": 5}))
        shown = " ".join(_html(at))
        assert "Pairs well with this" in shown and "Better-rated alternative" in shown
        assert "bought together" not in shown.lower()
        assert {"d_cross_8", "d_up_9", "cart_d_cross_8"} <= set(api_keys(at))


def test_about_parts():
    from ui_common import about_parts

    assert about_parts("") == [] and about_parts(None) == []
    assert about_parts("Zipper closure Hand Wash Only") == ["Zipper closure Hand Wash Only"]
    assert about_parts("95% Cotton; Imported; Hand wash") == ["95% Cotton", "Imported",
                                                               "Hand wash"]
    assert about_parts("Soft fabric - Machine wash - Imported") == [
        "Soft fabric", "Machine wash", "Imported"]
    assert about_parts("Soft fabric. Machine wash. 2 pack.") == [
        "Soft fabric.", "Machine wash.", "2 pack."]
    assert about_parts("Size 3.5 inch wide") == ["Size 3.5 inch wide"]   # no split mid-number


def test_skin_is_the_theme():
    """One token block: the Streamlit theme (both launch paths) carries SKIN's values,
    so a reskin cannot leave the widgets in the old colours."""
    try:
        import tomllib
    except ImportError:
        _skip("tomllib needs Python 3.11+")
    from ui_common import SKIN, _SHOP_CSS

    theme = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text(
        encoding="utf-8"))["theme"]
    assert theme["primaryColor"] == SKIN["link"] == theme["linkColor"]
    assert theme["backgroundColor"] == SKIN["page"]
    assert theme["textColor"] == SKIN["text"] and theme["borderColor"] == SKIN["border"]
    assert theme["orangeColor"] == SKIN["star"]   # the rail's stars AND the bot avatar
    # Streamlit's input fill, phone sidebar and rail expander header = the grey pill fill
    assert theme["secondaryBackgroundColor"] == SKIN["pill-bg"]
    assert theme["sidebar"]["backgroundColor"] == SKIN["page"]
    # every colour in the stylesheet is a token: no literal hex or rgb outside SKIN
    assert not re.search(r"#[0-9A-Fa-f]{3,8}\b|rgba?\(", _SHOP_CSS), \
        re.findall(r"#[0-9A-Fa-f]{3,8}\b|rgba?\([^)]*\)", _SHOP_CSS)


def test_masthead_keeps_search_settings_and_cart():
    """The marketplace band keeps the pinned pieces: the 'Search' submit label (hidden
    behind the magnifier), Settings with Developer details, and the Cart count."""
    with fake_http(FakeAPI()):
        at = _run(_page("keyword_search.py"))
        assert [b for b in at.button if b.label == "Search"]
        assert [t for t in at.toggle if t.label == "Developer details"]
        assert any(b.key == "keyword__topbar_cart" and b.label == "Cart **0**"
                   for b in at.button)
        assert not [s for s in at.selectbox if s.label == "Search in"], \
            "/keyword_search takes no department filter"
        assert not [b for b in at.button if "subnav_" in (b.key or "")]


def test_display_maps_match_the_backend():
    """The UI keeps its own copy of the C1 label and colour maps (it imports nothing from
    app/): the product page must name a style the way the rail (the API) does."""
    try:
        from app import product_type
    except ImportError as e:
        _skip(f"app.product_type not importable: {e}")
    import ui_common

    labels = getattr(product_type, "FINE_SUBTYPE_LABELS", None)
    families = getattr(product_type, "COLOR_FAMILIES", None)
    if labels is None or families is None:
        _skip("app.product_type has no C1 maps yet")
    assert ui_common.FINE_SUBTYPE_LABELS == dict(labels)
    assert set(ui_common.COLOR_HEX) == set(families)
    # the card's own row is in `variants` and in n_variants (app/main._collapse_variants)
    assert "+2 options" in ui_common.variants_html({"n_variants": 3})
    assert ui_common.variants_html({"n_variants": 1}) == ""


# ------------------------------------------------------------------ "Related items" (2026-10-27)
# Oct 27 meeting: "a button on each result for the cross-sell suggestions" (the title was
# the only way to the product page, and the suggestions sit at its bottom)
def _recs_api(**over):
    def product(p):
        return {"product": _product(int(p["catalog_index"])), "errors": [],
                "recommendations": {"cross_sell": [_product(80)], "upsell": [_product(90)],
                                    "cross_sell_source": "llm"}}
    return FakeAPI(product=product, **over)


def _related_keys(at):
    return [k for k in api_keys(at) if k.startswith("related_")]


def _block_keys(at):
    """The keyed containers in document order (a block's id ends in '-<key>')."""
    from streamlit.testing.v1.element_tree import Block

    out = []

    def walk(node):
        if isinstance(node, Block):
            m = re.match(r"\$\$ID-[0-9a-f]+-(.+)$", getattr(node.proto, "id", "") or "")
            if m:
                out.append(m.group(1))
            for i in sorted(node.children):
                walk(node.children[i])

    walk(at.main)
    return out


def test_related_button_opens_the_page_at_the_suggestions():
    """A real button on every grid card; a click is the title's one /product call (no
    extra LLM call) and ONE scroll script, emitted after the suggestions are drawn and
    aimed at their anchor; Back still lands on the card."""
    api = _recs_api()
    with fake_http(api):
        at = _run(_page("ai_search.py", {"ai__query": "gift for a watch collector",
                                         "ai__last_response": search_payload(1)}))
        assert _related_keys(at) == ["related__100", "related__101", "related__102"]
        b = _button(at, "related__101")
        assert b.label == "Related items" and b.proto.type == "secondary", b.proto.type
        assert b.help and not _FORBIDDEN.search(b.label + " " + b.help), b.help
        assert "unrelated" not in b.help and "bought" not in b.help.lower()
        # true under every outcome: the look-alike fallback ("Similar items") is no
        # complement, and the upsell or the whole section can be missing (2026-10-27 review)
        assert "pair well" not in b.help.lower() and "where there is one" in b.help, b.help
        n_search = api.count("search")
        b.click()
        _run(at)
        assert at.session_state["ai__view"] == "detail"
        assert at.session_state["ai__detail_index"] == 101
        assert api.count("product") == 1 and api.last("product")["recommend"] == "true"
        s = scripts(at)
        assert len(s) == 1 and "'cas_related'" in s[0], s     # ONE script, at the anchor
        # emitted AFTER the suggestions are drawn: the script looks for the anchor only
        # for ~3 s, and the slow /product fetch sits between the top of the run and here
        order = _block_keys(at)
        assert "cas_scroll" not in order, order
        assert order.index("cas_related") < order.index("cas_scroll_late"), order
        assert at.session_state["ai__detail_focus"] is None   # one-shot
        assert {"d_cross_80", "d_up_90"} <= set(api_keys(at))
        assert any("Pairs well with this" in h for h in _html(at))
        assert not _related_keys(at), "no Related button on the carousels' mini cards"
        _run(at)                                              # unrelated rerun
        assert scripts(at) == [] and api.count("product") == 1
        _button(at, "detail_back_ai").click()
        _run(at)
        s = scripts(at)
        assert len(s) == 1 and "'grid__101'" in s[0], "Back lands on the card"
        assert api.count("search") == n_search
        # and a later open from the TITLE goes to the top: the focus never lingers
        _button(at, "grid__102").click()
        _run(at)
        s = scripts(at)
        assert len(s) == 1 and "target = null" in s[0], s


def test_title_still_opens_the_top_of_the_page():
    with fake_http(_recs_api()):
        at = _run(_page("ai_search.py", {"ai__query": "q"}))
        _button(at, "grid__100").click()
        _run(at)
        s = scripts(at)
        assert len(s) == 1 and "target = null" in s[0], s


def test_related_button_follows_the_recommendations_setting():
    """Off, the product page has no suggestions: the cards offer no Related button."""
    with fake_http(_recs_api()) as api:
        at = _run(_page("ai_search.py", {"ai__query": "q"}))
        assert _related_keys(at)
        [t for t in at.toggle if t.label == "Product recommendations"][0].set_value(False)
        _run(at)
        assert not _related_keys(at)
        _button(at, "grid__100").click()
        _run(at)
        assert api.last("product")["recommend"] == "false"


def test_related_button_on_keyword_assistant_and_bundle_cards():
    def kw(p):
        return {"results": [_product(1), _product(2)], "num_matched": 2}

    with fake_http(_recs_api(keyword=kw)):
        at = _run(_page("keyword_search.py", {"keyword__query": "belt"}))
        assert _related_keys(at) == ["related__1", "related__2"]
        _button(at, "related__2").click()
        _run(at)
        assert len(scripts(at)) == 1 and "'cas_related'" in scripts(at)[0]
        _button(at, "detail_back_keyword").click()
        _run(at)
        assert "'grid__2'" in scripts(at)[0]
        [t for t in at.toggle if t.label == "Product recommendations"][0].set_value(False)
        _run(at)
        assert not _related_keys(at)
    reply = {"reply": "ok", "action": "search", "new_topic": True,
             "search_query": "desk chair", "results": [_product(i) for i in range(3)]}
    with fake_http(_recs_api(chat=lambda body: reply)):
        at = _run(_page("shopping_assistant.py"))
        at.chat_input[0].set_value("a desk chair")
        _run(at)
        assert _related_keys(at) == ["related_chatpanel_0", "related_chatpanel_1",
                                     "related_chatpanel_2"], _related_keys(at)
        _button(at, "related_chatpanel_1").click()
        _run(at)
        assert len(scripts(at)) == 1 and "'cas_related'" in scripts(at)[0]
        _button(at, "detail_back_assistant").click()
        _run(at)
        assert "'grid_chatpanel_1'" in scripts(at)[0]
        # the Assistant follows its own "Product recommendations" setting too
        [t for t in at.toggle if t.label == "Product recommendations"][0].set_value(False)
        _run(at)
        assert "grid_chatpanel_1" in api_keys(at) and not _related_keys(at), api_keys(at)


def test_no_suggestions_still_lands_on_the_section():
    with fake_http(FakeAPI()):   # empty recommendations
        at = _run(_page("ai_search.py", {"ai__query": "q"}))
        _button(at, "related__100").click()
        _run(at)
        assert any("No related products to suggest" in c.value for c in at.caption)
        assert len(scripts(at)) == 1 and "'cas_related'" in scripts(at)[0]


def test_a_failed_product_fetch_from_related_goes_to_the_top():
    """The open requested no top scroll (the suggestions were the target); with no
    page to land on, the warning must not sit off-screen where the grid was."""
    class API(FakeAPI):
        def get(self, url, params=None, timeout=None, **_):
            if url.endswith("/product"):
                self.calls.append(("product", dict(params or {})))
                raise requests.ConnectionError("down")
            return super().get(url, params=params, timeout=timeout)

    with fake_http(API()):
        at = _run(_page("ai_search.py", {"ai__query": "q"}))
        _button(at, "related__101").click()
        _run(at)
        assert any("Could not load this product." in w.value for w in at.warning)
        s = scripts(at)
        assert len(s) == 1 and "target = null" in s[0], s


def test_related_button_css_uses_tokens_and_the_card_margin():
    import ui_common
    css = ui_common._SHOP_CSS
    assert ".st-key-cas_related{scroll-margin-top:72px}" in css
    assert ".st-key-cas_related{scroll-margin-top:132px}" in css
    # a neutral outlined pill (visibly a button), never a bare link, in SKIN colours
    assert '[class*="st-key-related_"] button{background:var(--cas-page) !important;' in css
    assert "border:1px solid var(--cas-border) !important;border-radius:100px" in css
    # the late emission's container is hidden like the top one (no 16px gap)
    assert ".st-key-cas_scroll_late" in ui_common._SCROLL_CSS
    assert ":has(> .st-key-cas_scroll_late)" in ui_common._SCROLL_CSS


def test_card_byline_prefers_brand():
    """2026-10-27: Electronics rows whose store is a chip vendor carry the maker from the
    listing's specs as `brand` ("intel" sold a Lenovo laptop)."""
    import ui_cart
    from ui_common import brand_name, card_head_html, cart_line_html, detail_info_html

    row = _product(1, store="intel", brand="Lenovo")
    for chunk in (card_head_html(row), detail_info_html(row), cart_line_html(row, 1)):
        assert "Lenovo" in chunk and "intel" not in chunk, chunk
    assert brand_name(_product(2, brand=None)) == "Acme"
    assert brand_name(_product(3, brand=float("nan"), store=None)) == ""
    assert '<div class="cas-brand"></div>' in card_head_html(_product(3, store=None))

    class _State(dict):
        pass

    old = ui_cart.st.session_state
    try:
        ui_cart.st.session_state = _State()
        ui_cart.add(row)
        assert ui_cart.items()[0]["product"]["brand"] == "Lenovo"   # the cart line too
    finally:
        ui_cart.st.session_state = old
    with fake_http(FakeAPI(search=lambda p: search_payload(1, results=[row], pages=1))):
        at = _run(_page("ai_search.py", {"ai__query": "laptop"}))
        assert any('<div class="cas-brand">Lenovo</div>' in b for b in _html(at))


# ------------------------------------------------------------------ per-department rail (2026-10-27)
# SYNTHETIC payloads in the facets.attributes contract (spec 3.3; the API half lands
# separately): sections {param, label, department, match, hint, values[{key,label,n}],
# n_values} + primary_department. Keys and labels follow the 2026-10-27 taxonomy, and
# test_rail_fixtures_are_what_the_api_can_send holds them to app/attributes.py's
# registry (a key the API refuses, 'speakers', sat here until 2026-10-30).
_RATING = {"4": 30, "3": 41, "2": 44, "1": 45}
_PRICE = {"min": 7.99, "max": 249.0, "n_priced": 60,
          "bands": [{"label": "Up to $25", "min": None, "max": 24.99, "n": 21},
                    {"label": "$25 to $50", "min": 25, "max": 49.99, "n": 18}]}


def _section(param, label, department, values, *, match="any", hint=""):
    return {"param": param, "label": label, "department": department, "match": match,
            "hint": hint, "n_values": len(values),
            "values": [{"key": k, "label": lab, "n": n} for k, lab, n in values]}


_ELEC_CATEGORY = _section("category", "Category", "Electronics", [
    ("headphones", "Headphones & earbuds", 86),
    ("cases_protection", "Cases & screen protection", 31),
    ("power_batteries", "Chargers, power & batteries", 12),
    ("audio", "Speakers & audio", 9)])
_ELEC_ITEM_TYPE = _section("item_type", "Item type", "Electronics", [
    ("device", "Devices", 95), ("accessory", "Accessories", 52),
    ("part", "Parts & components", 4), ("consumable", "Batteries & supplies", 3)])
_ELEC_WORKS_WITH = _section("works_with", "Works with", "Electronics", [
    ("apple", "Apple", 21), ("samsung", "Samsung", 9), ("pc", "Dell, HP, Lenovo & PCs", 2)])
_ELEC_CONDITION = _section("condition", "Condition", "Electronics", [
    ("renewed", "Renewed / refurbished", 6), ("used", "Used", 1)],
    hint="As the listing words it")
_GROC_CATEGORY = _section("category", "Category", "Grocery", [
    ("snacks", "Chips, crackers & snacks", 149), ("bakery", "Cookies, cakes & bakery", 23),
    ("breakfast", "Breakfast, cereal & spreads", 10),
    ("non_food", "Kitchen, pet & other non-food", 2)])
_GROC_DIETARY = _section("dietary", "Dietary", "Grocery", [
    ("gluten free", "Gluten free", 203), ("non-gmo", "Non-GMO", 51), ("vegan", "Vegan", 45),
    ("kosher", "Kosher", 35), ("organic", "Organic", 23), ("dairy free", "Dairy free", 22),
    ("sugar free", "Sugar free / no added sugar", 9), ("keto", "Keto", 4),
    ("vegetarian", "Vegetarian", 3)],
    match="all", hint="As the listing words it, not a certification")

FACETS_ELECTRONICS = {
    "departments": {"Electronics": 154}, "product_types": {}, "genders": [], "subtypes": [],
    "materials": [], "colors": [{"key": "black", "label": "Black", "hex": "#111111", "n": 40}],
    "stores": [{"name": "Sony", "n": 20}, {"name": "JBL", "n": 12}], "n_stores": 2,
    "price": _PRICE, "rating": _RATING, "primary_department": "Electronics",
    "attributes": [_ELEC_CATEGORY, _ELEC_ITEM_TYPE, _ELEC_WORKS_WITH, _ELEC_CONDITION]}
FACETS_GROCERY = {
    "departments": {"Grocery": 203}, "product_types": {}, "genders": [], "subtypes": [],
    "materials": [], "colors": [], "stores": [{"name": "Kind", "n": 30}], "n_stores": 1,
    "price": _PRICE, "rating": _RATING, "primary_department": "Grocery",
    "attributes": [_GROC_CATEGORY, _GROC_DIETARY]}
# "gift for a coffee lover": 86% Grocery, a Fashion and an Electronics minority. Sections
# in the API's order (the registry's, every department's), not the primary's first
FACETS_MIXED = {
    "departments": {"Grocery": 195, "Electronics": 21, "Fashion": 10},
    "product_types": {"apparel": 6, "jewelry": 4},
    "genders": [{"key": "women", "label": "Women", "n": 6},
                {"key": "unisex", "label": "Unisex / unspecified", "n": 4}],
    "subtypes": [{"key": "tops_tees", "label": "Tops & tees", "n": 5}],
    "materials": [{"key": "cotton", "label": "Cotton", "n": 3}],
    "colors": [{"key": "black", "label": "Black", "hex": "#111111", "n": 9}],
    "stores": [{"name": "Lavazza", "n": 12}], "n_stores": 1,
    "price": _PRICE, "rating": _RATING, "primary_department": "Grocery",
    "attributes": [_ELEC_CATEGORY, _ELEC_ITEM_TYPE, _GROC_CATEGORY, _GROC_DIETARY]}


def _headings(at):
    out = []
    for body in _html(at):
        out += re.findall(r'<div class="cas-rail-h">(.*?)(?:\s*<span|</div>)', body)
    return out


def _shortcuts(at):
    return [b.label for b in at.button if "subnav_" in (b.key or "")]


def _attr_boxes(at, param):
    return [c for c in at.checkbox if f"__f_a_{param}_" in (c.key or "")]


def _match_all_notes(at):
    # the element, not the stylesheet (whose rule also names the class)
    return [b for b in _html(at) if '<div class="cas-rail-note">' in b]


def _rail_page(facets, query="q"):
    api = FakeAPI(search=lambda p: search_payload(1, facets=facets))
    return api, {"ai__query": query, "ai__last_response": search_payload(1)}


def test_grocery_page_gets_its_own_sections():
    api, state = _rail_page(FACETS_GROCERY, "gluten free snacks")
    with fake_http(api):
        at = _run(_page("ai_search.py", state))
        assert _headings(at) == ["Department", "Category", "Dietary", "Customer Reviews",
                                 "Price", "Brands"], _headings(at)
        assert _shortcuts(at) == ["All", "Grocery"], _shortcuts(at)
        assert any('title="As the listing words it, not a certification"' in b
                   for b in _html(at)), "the dietary hint"
        assert [_plain(c.label) for c in _attr_boxes(at, "category")] == [
            "Chips, crackers & snacks", "Cookies, cakes & bakery",
            "Breakfast, cereal & spreads", "Kitchen, pet & other non-food"], "API order"
        # 8 of 9 dietary values in view, the 9th under See more; no counts by default
        assert len(_attr_boxes(at, "dietary")) == 9
        assert any(e.label == "See more (9)" for e in at.expander), \
            [e.label for e in at.expander]
        assert "Sugar free / no added sugar" in [_plain(c.label) for c in at.checkbox]
        assert not any(re.search(r"\(\d+\)$", c.label) for c in at.checkbox)
        assert "Grocery (203)" in [b.label for b in at.button]   # exact: nothing picked

        _checkbox(at, "Vegan").check()
        _run(at)
        sent = api.last("search")
        assert sent["dietary"] == "vegan" and sent["follow_up"] == "true" \
            and sent["page"] == 1, sent
        assert len(scripts(at)) == 1
        # a tick makes the Department count inexact (it counts the unfiltered pool)
        assert _rail(at, "Grocery").label == "Grocery", _rail(at, "Grocery").label
        _checkbox(at, "Organic").check()
        _run(at)
        assert api.last("search")["dietary"] == "organic,vegan"   # sorted csv
        _checkbox(at, "Chips, crackers & snacks").check()
        _run(at)
        sent = api.last("search")
        assert sent["category"] == "snacks" and sent["dietary"] == "organic,vegan", sent
        import ui_common
        # the rail's "Filters (n applied)" (AppTest does not list that keyed expander)
        assert ui_common._n_applied(at.session_state["ai__filters"]) == 2
        assert _checkbox(at, "Vegan").value is True

        _button_prefix(at, "ai__pg_next").click()     # a page change keeps them
        _run(at)
        sent = api.last("search")
        assert sent["page"] == 2 and sent["category"] == "snacks", sent

        [b for b in at.button if b.label == "Clear filters"][0].click()
        _run(at)
        sent = api.last("search")
        assert not {"dietary", "category"} & set(sent), sent
        [t for t in at.toggle if t.label == "Developer details"][0].set_value(True)
        _run(at)
        assert "Vegan (45)" in [c.label for c in at.checkbox]   # counts: dev only


def test_electronics_page_gets_its_own_sections():
    api, state = _rail_page(FACETS_ELECTRONICS, "wireless earbuds")
    with fake_http(api):
        at = _run(_page("ai_search.py", state))
        assert _headings(at) == ["Department", "Category", "Item type", "Works with",
                                 "Condition", "Customer Reviews", "Price", "Brands",
                                 "Colour"], _headings(at)
        assert _shortcuts(at) == ["All", "Electronics"], _shortcuts(at)
        assert any('title="As the listing words it"' in b for b in _html(at))
        for label, param, key in (("Headphones & earbuds", "category", "headphones"),
                                  ("Devices", "item_type", "device"),
                                  ("Apple", "works_with", "apple"),
                                  ("Renewed / refurbished", "condition", "renewed")):
            _checkbox(at, label).check()
            _run(at)
            assert api.last("search")[param] == key, (param, api.last("search"))
        sent = api.last("search")
        assert (sent["category"], sent["item_type"], sent["works_with"],
                sent["condition"]) == ("headphones", "device", "apple", "renewed"), sent
        # picking the department is a department change: its sections start over
        _rail(at, "Electronics").click()
        _run(at)
        sent = api.last("search")
        assert sent["department"] == "Electronics", sent
        assert not {"category", "item_type", "works_with", "condition"} & set(sent), sent
        assert "Item type" in _headings(at)
        # ... and their boxes start over too: the sections are drawn again for the same
        # focus, and a box kept its widget state over the cleared filter (2026-10-27
        # review: four boxes shown ticked, none sent, "Filters (1 applied)")
        assert not [c.label for c in at.checkbox if c.value], \
            [c.label for c in at.checkbox if c.value]
        import ui_common
        assert ui_common._n_applied(at.session_state["ai__filters"]) == 1   # department


def test_mixed_pool_follows_the_department():
    """'gift for a coffee lover' (86% Grocery) listed Fashion's Category (Jewelry,
    Clothing) and Style (Tops & tees) from 10 Fashion rows. The rail now shows the focus
    department's sections; picking another department switches them."""
    api, state = _rail_page(FACETS_MIXED, "gift for a coffee lover")
    with fake_http(api):
        at = _run(_page("ai_search.py", state))
        h = _headings(at)
        assert h[:3] == ["Department", "Category", "Dietary"], h
        assert "Style" not in h and "Fabric" not in h and "Item type" not in h, h
        assert not [b for b in at.button if "__f_ptype_" in (b.key or "")], "no Jewelry"
        # every department is in the pool, but its genders facet offers Women (and
        # Unisex) only: Men and Kids could only empty the page (2026-10-27 review)
        assert _shortcuts(at) == ["All", "Fashion", "Women", "Electronics", "Grocery"], \
            _shortcuts(at)

        _rail(at, "Electronics").click()
        _run(at)
        h = _headings(at)
        assert "Item type" in h and "Dietary" not in h and "Style" not in h, h
        assert "Devices" in [_plain(c.label) for c in at.checkbox]
        _checkbox(at, "Devices").check()
        _run(at)
        sent = api.last("search")
        assert sent["item_type"] == "device" and sent["department"] == "Electronics", sent

        _rail(at, "‹ Any Department").click()                 # a level up clears below
        _run(at)
        sent = api.last("search")
        assert "item_type" not in sent and "department" not in sent, sent
        assert "Dietary" in _headings(at)

        # a gender with no department moves the focus to Fashion: the Grocery ticks go
        _checkbox(at, "Organic").check()
        _run(at)
        assert api.last("search")["dietary"] == "organic"
        _rail(at, "Women").click()
        _run(at)
        sent = api.last("search")
        assert sent["gender"] == "women" and "dietary" not in sent, sent
        h = _headings(at)
        assert "Style" in h and "Fabric" in h and "Dietary" not in h, h


def test_subnav_department_change_clears_attribute_filters():
    api, state = _rail_page(FACETS_MIXED)
    with fake_http(api):
        at = _run(_page("ai_search.py", state))
        _checkbox(at, "Organic").check()
        _run(at)
        assert api.last("search")["dietary"] == "organic"
        _button(at, "ai__subnav_5").click()                   # Electronics
        _run(at)
        sent = api.last("search")
        assert sent["department"] == "Electronics" and "dietary" not in sent, sent
        _checkbox(at, "Devices").check()
        _run(at)
        _button(at, "ai__subnav_5").click()                   # the active one: no change
        _run(at)
        assert api.last("search")["item_type"] == "device"
        _button(at, "ai__subnav_0").click()                   # All
        _run(at)
        sent = api.last("search")
        assert "item_type" not in sent and "department" not in sent, sent
        assert not [c.label for c in at.checkbox if c.value]


def _ticked(at):
    return [_plain(c.label) for c in at.checkbox if c.value]


def test_a_department_click_unticks_what_it_clears():
    """2026-10-27 review (both reviewers): a department change clears filters["attr"],
    but when the focus did not move the same section was drawn again and its boxes kept
    their widget state: "Vegan" or "Gluten free" shown ticked over a page no longer
    filtered by it, "Filters" counting none, and the next tick sending only itself. The
    boxes now always mirror the applied filters. Every path the reviewers probed."""
    import ui_common

    def tick(at, api, label, param, want):
        _checkbox(at, label).check()
        _run(at)
        assert api.last("search")[param] == want, api.last("search")

    def unticked(at, api, *params):
        sent = api.last("search")
        assert not set(params) & set(sent), sent
        assert _ticked(at) == [], _ticked(at)
        assert not ui_common._attr_filters(at.session_state["ai__filters"])

    # (1) the sub-nav's Grocery on a Grocery pool, (2) the rail's Grocery link
    for click in (lambda at: _button(at, "ai__subnav_6").click(),
                  lambda at: _rail(at, "Grocery").click()):
        api, state = _rail_page(FACETS_GROCERY)
        with fake_http(api):
            at = _run(_page("ai_search.py", state))
            tick(at, api, "Vegan", "dietary", "vegan")
            click(at)
            _run(at)
            assert api.last("search")["department"] == "Grocery"
            assert "Dietary" in _headings(at)            # the same section, drawn again
            unticked(at, api, "dietary")
            tick(at, api, "Kosher", "dietary", "kosher")  # sends what it shows, alone
            assert _ticked(at) == ["Kosher"], _ticked(at)

    # (3) '‹ Any Department' and (4) the sub-nav's All, each back to the primary Grocery
    for back in (lambda at: _rail(at, "‹ Any Department").click(),
                 lambda at: _button(at, "ai__subnav_0").click()):
        api, state = _rail_page(FACETS_MIXED)
        with fake_http(api):
            at = _run(_page("ai_search.py", state))
            _rail(at, "Grocery").click()
            _run(at)
            tick(at, api, "Organic", "dietary", "organic")
            back(at)
            _run(at)
            assert "department" not in api.last("search") and "Dietary" in _headings(at)
            unticked(at, api, "dietary")


def test_the_active_shortcut_stays_when_the_pool_lacks_it():
    """D4: a shortcut the pool cannot fill leaves the sub-nav, except the ACTIVE one: it
    says which scope filters the page, and is the way back out of an empty one."""
    state = {"facets": FACETS_C2}
    api = FakeAPI(search=lambda p: search_payload(1, facets=state["facets"]))
    with fake_http(api):
        at = _run(_page("ai_search.py"))
        _button(at, "ai__subnav_5").click()                   # Electronics, before a search
        _run(at)
        assert _submit(at, api, "linen shirt")["department"] == "Electronics"
        # the answer is Fashion-only (FACETS_C2): Electronics stays, as the active one
        assert _shortcuts(at) == ["All", "Fashion", "Women", "Men", "Electronics"], \
            _shortcuts(at)
        assert _button(at, "ai__subnav_5").proto.type == "secondary"

        _button(at, "ai__subnav_2").click()                   # Women, on the results
        _run(at)
        assert api.last("search")["gender"] == "women"
        assert _shortcuts(at) == ["All", "Fashion", "Women", "Men"], _shortcuts(at)
        state["facets"] = FACETS_ELECTRONICS                  # a pool with no genders
        [t for t in at.toggle if t.label == "AI ranking"][0].set_value(False)
        _run(at)
        assert api.last("search")["gender"] == "women"
        # Women stays (active); Men, Kids and Fashion go; the pool's Electronics shows
        assert _shortcuts(at) == ["All", "Women", "Electronics"], _shortcuts(at)


def test_a_picked_style_or_fabric_stays_when_the_focus_leaves_fashion():
    """Category / Style / Fabric follow the focus, but a section with a value picked is
    drawn whatever the focus (the pool can change under the same query): a filter in
    force is never off screen."""
    state = {"facets": FACETS_C2}
    api = FakeAPI(search=lambda p: search_payload(1, facets=state["facets"]))
    with fake_http(api):
        at = _run(_page("ai_search.py", {"ai__query": "q",
                                         "ai__last_response": search_payload(1)}))
        _rail(at, "Dresses").click()
        _run(at)
        _checkbox(at, "Linen").check()
        _run(at)
        state["facets"] = FACETS_GROCERY                      # primary_department Grocery
        [t for t in at.toggle if t.label == "AI ranking"][0].set_value(False)
        _run(at)
        sent = api.last("search")
        assert sent["subtype"] == "dresses" and sent["material"] == "linen", sent
        h = _headings(at)
        assert "Dietary" in h and "Style" in h and "Fabric" in h, h
        assert "Category" in h and h.count("Category") == 1, "Grocery's, not Fashion's"
        # this pool has no materials: the picked value is drawn by its key, at 0 (the
        # 2026-10-19 _with_chosen rule; Fabric has no label map)
        assert _checkbox(at, "linen").value is True
        assert _rail(at, "Dresses")
        _rail(at, "‹ Any Style").click()
        _run(at)
        assert "subtype" not in api.last("search")
        assert "Style" not in _headings(at) and "Fabric" in _headings(at)


def test_a_repeated_value_draws_one_box():
    """A key the API repeats, or one ticked twice, would repeat a widget key: a
    StreamlitDuplicateElementKey takes the page down. One box per key; sent once."""
    dietary = _section("dietary", "Dietary", "Grocery", [
        ("vegan", "Vegan", 45), ("vegan", "Vegan again", 3), ("kosher", "Kosher", 35)])
    api, state = _rail_page(dict(FACETS_GROCERY, attributes=[dietary]))
    with fake_http(api):
        at = _run(_page("ai_search.py", state))
        assert [_plain(c.label) for c in _attr_boxes(at, "dietary")] == ["Vegan", "Kosher"]
        _checkbox(at, "Vegan").check()
        _run(at)
        assert api.last("search")["dietary"] == "vegan"
        # a repeated tick, including one of a value this pool lacks (drawn at 0, by its
        # key: this section never named it). 'halal', a key the API accepts ('paleo' until
        # 2026-10-30: the real API refuses it with a 422)
        at.session_state["ai__filters"] = {"attr": {"dietary": ["halal", "vegan", "halal",
                                                                "vegan"]}}
        _run(at)
        assert api.last("search")["dietary"] == "halal,vegan", api.last("search")
        assert [(_plain(c.label), c.value) for c in _attr_boxes(at, "dietary")] == [
            ("Vegan", True), ("halal", True), ("Kosher", False)], \
            [(c.label, c.value) for c in _attr_boxes(at, "dietary")]


def test_a_section_cap_disables_the_other_boxes():
    """`max_picked` when the API sends one: at the cap only a ticked box can change, and
    a constant caption says why (as Brands does at its 20)."""
    capped = dict(_GROC_DIETARY, max_picked=1)
    api, state = _rail_page(dict(FACETS_GROCERY, attributes=[_GROC_CATEGORY, capped]))
    with fake_http(api):
        at = _run(_page("ai_search.py", state))
        assert not any(c.proto.disabled for c in at.checkbox)
        _checkbox(at, "Vegan").check()
        _run(at)
        state = {_plain(c.label): c.proto.disabled for c in _attr_boxes(at, "dietary")}
        assert state.pop("Vegan") is False and state and all(state.values()), state
        assert not any(c.proto.disabled for c in _attr_boxes(at, "category"))
        assert any(c.value == "Up to 1 at a time: untick one to pick another."
                   for c in at.caption), [c.value for c in at.caption]
        _checkbox(at, "Vegan").uncheck()
        _run(at)
        assert not any(c.proto.disabled for c in at.checkbox)


def test_a_match_all_section_says_its_ticks_must_all_hold():
    """Dietary is match "all" (spec 3.3, app/attributes.py: a product must carry EVERY
    ticked option), the one rail section whose ticks narrow instead of widen. Until
    2026-10-30 the UI never read `match`, so "Vegan" + "Kosher" looked like either. A
    constant note sits under that heading; a match "any" section (Grocery's Category,
    the whole Electronics rail) is drawn as before."""
    import ui_common

    note = f'<div class="cas-rail-note">{ui_common.esc(ui_common.MATCH_ALL_NOTE)}</div>'
    api, state = _rail_page(FACETS_GROCERY, "gluten free snacks")
    with fake_http(api):
        at = _run(_page("ai_search.py", state))
        bodies = _html(at)
        assert _match_all_notes(at) == [note], _match_all_notes(at)
        # in the Dietary section: the last heading above it is Dietary's, not Category's
        above = [b for b in bodies[:bodies.index(note)] if '<div class="cas-rail-h">' in b]
        assert above[-1].startswith('<div class="cas-rail-h">Dietary'), above
        assert len(above) == _headings(at).index("Dietary") + 1, "right under the heading"
        assert "Items match every option you tick." in note   # the words, as drawn
        _checkbox(at, "Vegan").check()
        _run(at)
        _checkbox(at, "Kosher").check()
        _run(at)
        assert api.last("search")["dietary"] == "kosher,vegan"
        assert _match_all_notes(at) == [note], "still there over the filtered page"
        assert not _FORBIDDEN.search(ui_common.MATCH_ALL_NOTE)

    # every section "any": no note. Electronics, and Grocery with Dietary sent as "any"
    # (the rule is the API's to state, never inferred from the param's name)
    as_any = dict(FACETS_GROCERY,
                  attributes=[_GROC_CATEGORY, dict(_GROC_DIETARY, match="any")])
    for facets in (FACETS_ELECTRONICS, as_any):
        api, state = _rail_page(facets)
        with fake_http(api):
            at = _run(_page("ai_search.py", state))
            assert "Category" in _headings(at)
            assert _match_all_notes(at) == [], _match_all_notes(at)


def test_match_all_reads_the_api_value_only():
    from ui_common import _match_all

    for section in ({"match": "all"}, {"match": " ALL "}, {"match": "All"}):
        assert _match_all(section), section
    for section in ({"match": "any"}, {"match": ""}, {"match": None}, {},
                    {"match": ["all"]}, {"match": "all of them"}, {"match": float("nan")},
                    None):
        assert not _match_all(section), section


def test_a_ticked_section_the_pool_lost_stays_on_the_rail():
    """The same query can come back with another pool (a setting toggled, no seed on
    Databricks): if its primary department moves, the ticked section is no longer the
    focus's. It stays drawn, ticked, so the filter in force can be seen and undone."""
    state = {"facets": FACETS_GROCERY}
    api = FakeAPI(search=lambda p: search_payload(1, facets=state["facets"]))
    with fake_http(api):
        at = _run(_page("ai_search.py", {"ai__query": "q",
                                         "ai__last_response": search_payload(1)}))
        _checkbox(at, "Vegan").check()
        _run(at)
        state["facets"] = FACETS_ELECTRONICS
        [t for t in at.toggle if t.label == "AI ranking"][0].set_value(False)
        _run(at)
        assert api.last("search")["dietary"] == "vegan"
        h = _headings(at)
        assert "Item type" in h and "Dietary" in h, h
        assert [(_plain(c.label), c.value) for c in _attr_boxes(at, "dietary")] == [
            ("Vegan", True)]
        # ... still saying its ticks must all hold (the API applies the param's rule
        # whatever the pool shows); this Electronics pool has no match-all section
        assert len(_match_all_notes(at)) == 1, _match_all_notes(at)
        _checkbox(at, "Vegan").uncheck()
        _run(at)
        assert "dietary" not in api.last("search")
        assert "Dietary" not in _headings(at) and not _match_all_notes(at)


def test_attribute_sections_are_checked_before_they_are_drawn():
    """Only well-formed, sendable sections of the focus department: never a param the
    UI already sends (it would steer that filter), a malformed one, a second section
    for the same param, another department's or a department-less one."""
    sections = [
        _section("dietary", "Dietary", "Grocery", [("vegan", "Vegan", 3)]),
        _section("dietary", "Twice", "Grocery", [("x", "Twice value", 1)]),
        _section("department", "Hijack", "Grocery", [("Fashion", "Hijack value", 1)]),
        _section("store", "Hijack store", "Grocery", [("x", "Store value", 1)]),
        _section("Bad-Param", "Bad", "Grocery", [("x", "Bad value", 1)]),
        _section("c" * 33, "Too long", "Grocery", [("x", "Long value", 1)]),
        _section("use", "Elsewhere", "Electronics", [("x", "Elsewhere value", 1)]),
        _section("spice", "Nowhere", None, [("x", "Nowhere value", 1)]),
        "not a section", {"param": None},
    ]
    facets = dict(FACETS_GROCERY, attributes=sections)
    api, state = _rail_page(facets)
    with fake_http(api):
        at = _run(_page("ai_search.py", state))
        h = _headings(at)
        assert h == ["Department", "Dietary", "Customer Reviews", "Price", "Brands"], h
        labels = [_plain(c.label) for c in at.checkbox]
        assert "Vegan" in labels and not [x for x in labels if x.endswith("value")], labels
    import ui_common
    assert ui_common.filter_params({"attr": {"dietary": ["vegan", "organic"],
                                             "Bad-Param": ["x"], "store": ["y"],
                                             "category": [], "use": "notalist"}},
                                   None) == {"dietary": "organic,vegan"}


def test_old_api_without_attributes_renders_as_before():
    """No facets.attributes / primary_department (an API before 2026-10-27): focus None,
    Fashion's sections as before, and no attribute param is ever sent."""
    import ui_common
    assert ui_common.focus_department(FACETS_C2, {}) is None
    api, state = _rail_page(FACETS_C2)
    with fake_http(api):
        at = _run(_page("ai_search.py", state))
        assert _headings(at) == ["Department", "Category", "Style", "Customer Reviews",
                                 "Price", "Brands", "Colour", "Fabric"], _headings(at)
        _checkbox(at, "Linen").check()
        _run(at)
        sent = api.last("search")
        assert set(sent) <= {"query", "top_k", "page", "rerank", "recommend", "sponsored",
                             "material", "follow_up"}, sent


def test_focus_department_and_visible_attributes():
    from ui_common import focus_department, visible_attributes

    assert focus_department(FACETS_MIXED, {}) == "Grocery"
    assert focus_department(FACETS_MIXED, {"department": ["Electronics"]}) == "Electronics"
    assert focus_department(FACETS_MIXED, {"department": ["Electronics", "Grocery"]}) \
        == "Grocery", "two departments: the pool's majority"
    assert focus_department(FACETS_MIXED, {"gender": ["women"]}) == "Fashion"
    assert focus_department(dict(FACETS_MIXED, primary_department=None), {}) is None
    assert focus_department(dict(FACETS_MIXED, primary_department=" "), {}) is None
    assert focus_department(None, None) is None
    assert [s["param"] for s in visible_attributes(FACETS_MIXED, "Grocery")] == \
        ["category", "dietary"]
    assert [s["label"] for s in visible_attributes(FACETS_MIXED, "electronics")] == \
        ["Category", "Item type"]
    assert visible_attributes(FACETS_MIXED, None) == []
    assert visible_attributes(FACETS_MIXED, "Fashion") == []
    assert visible_attributes({"attributes": None}, "Grocery") == []


def _fixture_sections():
    """Every facets.attributes section among this module's fixtures, found by shape (a
    section constant, or a payload's `attributes` list), so a new fixture is checked
    without being listed. Sections a test builds inline to be malformed, hostile or
    hypothetical (the EVIL Fashion section, a repeated key, a hijacked param) are not
    fixtures and stay out."""
    out = {}
    for name, value in sorted(globals().items()):
        if not isinstance(value, dict):
            continue
        if isinstance(value.get("attributes"), list):
            out[name] = list(value["attributes"])
        elif {"param", "department", "values"} <= set(value):
            out[name] = [value]
    return out


def test_rail_fixtures_are_what_the_api_can_send():
    """The synthetic rail payloads above stand in for GET /search's facets.attributes, so
    they must be something the API can send: each section a registry entry of
    app/attributes.py (param + department, its label, match and hint), each value a key
    of that entry's FIXED vocabulary with its label, values by count then vocabulary
    order, sections in registry order. 'speakers' (Electronics' group key is 'audio')
    passed every AppTest here while the real API answered a 422 (2026-10-30)."""
    try:
        from app import attributes
    except ImportError as e:
        _skip(f"app.attributes not importable: {e}")

    registry = {(f.param, f.department): (i, f) for i, f in enumerate(attributes.FACETS)}
    found = _fixture_sections()
    assert {"FACETS_ELECTRONICS", "FACETS_GROCERY", "FACETS_MIXED", "_ELEC_CATEGORY",
            "_GROC_DIETARY"} <= set(found), sorted(found)
    for name, sections in found.items():
        positions = []
        for s in sections:
            where = (name, s.get("param"), s.get("department"))
            assert (s.get("param"), s.get("department")) in registry, where
            i, facet = registry[(s["param"], s["department"])]
            positions.append(i)
            assert (s["label"], s["match"], s["hint"]) == (facet.label, facet.match,
                                                           facet.hint), where
            keys = [v["key"] for v in s["values"]]
            unknown = [k for k in keys if k not in facet.vocab]
            assert not unknown, (where, unknown, list(facet.vocab))
            # what the API also checks: a key it accepts for this param at all
            assert set(keys) <= set(attributes.vocabulary(s["param"])), where
            assert [v["label"] for v in s["values"]] == [facet.vocab[k] for k in keys], where
            counts, order = {v["key"]: v["n"] for v in s["values"]}, list(facet.vocab)
            assert keys == sorted(keys, key=lambda k: (-counts[k], order.index(k))), \
                (where, "values by count, then vocabulary order")
            assert len(keys) == len(set(keys)) <= attributes.MAX_VALUES_SHOWN, where
        assert positions == sorted(positions), (name, "sections in registry order")
    # a payload's primary department is one the registry describes (or None)
    for name in ("FACETS_ELECTRONICS", "FACETS_GROCERY", "FACETS_MIXED"):
        primary = globals()[name]["primary_department"]
        assert primary in {d for _, d in registry}, (name, primary)


def test_no_claim_on_the_new_rail_and_card_button():
    """The words a marketplace look-alike must not use stay off the new sinks: the card
    button, the per-department sections (labels and hints), the sub-nav."""
    for facets in (FACETS_ELECTRONICS, FACETS_GROCERY, FACETS_MIXED):
        with fake_http(_recs_api(search=lambda p, f=facets: search_payload(1, facets=f))):
            at = _run(_page("ai_search.py", {"ai__query": "q"}))
            assert _related_keys(at)
            shown = " ".join(_texts(at) + [h for h in _html(at) if "cas-scroll" not in h])
            shown = shown.replace("Amazon Reviews 2023", "")   # the dataset citation
            assert not _FORBIDDEN.search(shown), _FORBIDDEN.search(shown)


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
            print(f"FAIL  {name}: {type(e).__name__}: {str(e)[:800]}")
    print(f"\n{len(tests) - failed - skipped} passed, {failed} failed, {skipped} skipped")
    sys.exit(1 if failed else 0)
