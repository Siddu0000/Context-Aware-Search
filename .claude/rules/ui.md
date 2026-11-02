---
paths:
  - "ui.py"
  - "ui_common.py"
  - "ui_cart.py"
  - "pages/**"
  - ".streamlit/**"
  - "tests/test_ui.py"
  - "databricks/app_bundle/run_app.py"
---
# Shop UI layout (load when working on the Streamlit UI)

The invariants (one token block, follow_up on non-submit fetches, no `st.stop`, Back to the
card, no marketplace names / marks / claims) are in CLAUDE.md ("Marketplace-style UI" decision,
"UI naming conventions"); escaping rules in safety.md. This file holds the 2026-10-19 layout
detail (owner call 9, marketplace-style; the mock and spec it follows -- the audit's
`ui_spec.md` / `ui_mock.html` -- are agent-made, NOT stakeholder-approved, and outside the repo)
and the 2026-10-27 round's card button and department rail sections (lead decisions D3 / D4
of that round's spec, in its session archive outside the repo; sections below).

## Cards (2026-10-19)
Image on a light square, a "Sponsored" line on every ad, brand, the TITLE as the link to the
product page (same `grid_{ns}_{cid}` key, so Back still lands on the card), rating number +
stars + (count), a listed price in superscript dollars and cents (an estimate stays grey
"≈ $X est.", never superscript, never in cart totals), "Highly rated" only at 4.5+ from 50+
ratings, "+N colours" swatches from `variants` (a same-colour size variant counts as an
"option", never a colour), an Add to cart pill (outlined blue since 2026-10-26; yellow before).
4 per row (2 per row at 375 px), `st.html`. Since 2026-10-27 the brand line -- and the product
page's "Brand:" line and the cart line -- read the API's `brand` (the maker the listing names
where the store is a chip vendor), else `store`; the Brands facet and the `store` filter read
the same. And a "Related items" button sits under Add to cart (next section).

## Related items button (2026-10-27; D3, the meeting's cross-sell / upsell ask)
- ONE button opens both suggestion strips: `st.button("Related items", type="secondary",
  icon=":material/chevron_right:", icon_position="right")`, no emoji, stacked under Add to cart
  on every GRID card of AI Search, Keyword Search and the Shopping Assistant (one per bundle
  Option tab); never on the product page's carousel mini cards or on cart lines. It follows the
  page's EXISTING "Product recommendations" setting (default on; no new knob).
- Key `related_{key_ns}_{cid}`, never `grid_*`: the title-link CSS and the tests' title lookups
  read that prefix. Help text: "Opens this item's page at its suggestions: related items and,
  where there is one, a better-rated alternative" (no "pairs well" promise: there is not always
  an upsell, and "Similar items" is the fallback).
- Click = the page the title opens (`_select_detail(..., focus="related")`), landing on the
  `cas_related` container: "Pairs well with this" (or "Similar items"), then "Better-rated
  alternative", headings unchanged. The anchor is drawn even with nothing to suggest ("No
  related products to suggest for this item."). The scroll is requested AFTER the
  recommendations draw (`apply_scroll(late=True)`, key `cas_scroll_late`): the top-of-run
  scroll fires before the slow /product fetch. A failed product fetch scrolls to the top,
  where the warning is.
- Back returns to the card: the return key is the TITLE's (`grid_{ns}_{cid}`), as after a
  title click.
- Look: tokens only, no new SKIN token: an outlined pill, `--cas-page` background, 1px
  `--cas-border`, `--cas-text` 13px, radius 100px, hover `--cas-pill-hover`, focus `--cas-link`;
  scroll margins 72 / 132 px as for a card.
- Deviations from D3: a secondary pill stacked under Add to cart (the round's design report
  had a same-row tertiary link), the help text reworded, no "For: <title>" line. Owner:
  wording and look.
- What it lands on: /product recommendations are now cached per product + the shopper's query
  (CLAUDE.md), except an EMPTY cross-sell reply, which is never pinned: the page re-asks on
  every open and can flip "Similar items" -> "Pairs well with this" (owner question). The
  better-rated alternative stays within the item's role / watch kind (CLAUDE.md Architecture
  8), so some items show none (offline, 0 LLM, 1,232 searches: 49 of 215 phones). Residual:
  the dev-only caption "cross-sell LLM unavailable (error <code>)" now also shows for an
  upsell search failure.

## Shell
- Masthead = one joined control: department select (AI Search only; it post-filters the
  matched pool through `department`, 0 LLM calls, not retrieval-time scoping) + search text +
  magnifier (`st.form(enter_to_submit=True)`; the submit label "Search" is visually hidden) +
  Settings + the Cart button (count in bold, cart icon); sticky from 640 px (60 px header +
  56 px band), two rows on a phone.
- Sub-nav (AI Search): All / Fashion / Women / Men / Kids / Electronics / Grocery, a follow_up
  filter, drawn only before any response or on a flat page with facets (`subnav_for`; on
  results only the shortcuts the pool can fill since 2026-10-27: "Department rail sections"
  below). A Search submit sends only the select's department; a gender picked on results
  (rail or sub-nav) filters those results and never carries to the next search; a shortcut
  picked BEFORE any search applies to that one search.
- Rail (AI Search; hidden when `facets` is absent or null, i.e. bundles / no-match):
  Department > gender links (a click on the department above a picked gender goes up one
  level), the focus department's sections (2026-10-27: below), Category, Style (Fashion),
  Customer Reviews "★★★★ & Up", price bands + slider (bounds rounded outward; an unmoved
  handle keeps a band's x.99) + the priced-share caption, Brands top 8 +
  "See more" (repeated `store`; stops at 20 ticks = `MAX_STORES`), Colour swatches (fixed
  `COLOR_HEX` map only), Fabric on clothing pools, Clear filters. Counts show only on the exact
  Department tree or with Developer details; starts collapsed on a phone (User-Agent); a ticked
  value the new pool lacks stays ticked at count 0; an older API without the new facets still
  gets Department / Category / Reviews / Price. A 4xx on a follow-up keeps the page with a
  fixed-text warning.
- Results bar "1–12 of N results for “query”" (+ "filtered from N") + sort pill; weak-match
  banner "No exact matches for “q”. Showing related items." from `match_quality.weak` (never
  when unjudged, never over an empty page); numbered pager (current ±2, first/last) +
  Previous/Next.
- Product page: Back + breadcrumb (Fashion › Women › Clothing › Dresses), image | details
  ("Also available in" variant buttons, a Product details table without Occasion -- the
  rule-based tag misfires -- "About this item" split only on explicit separators) | buy box
  (quantity, Add to Cart outlined blue, Buy Now filled blue -- yellow / orange before
  2026-10-26 --, "Demo store: no checkout or payment."; no stock, delivery or seller lines);
  recommendation carousels below, inside the `cas_related` container that "Related items"
  lands on (2026-10-27).
- Assistant: restyled chat panel (Clear conversation in its header, input pinned).
- Footer: Back to top, three columns, a licence strip with the non-affiliation line.
- Gutters 24 / 20 / 16 px by width (>= 1024, 640-1023, < 640 px: `.stApp --cas-gutter`).

## Department rail sections (2026-10-27; D4, the meeting's Electronics / Grocery rail)
- ONE generic mechanism: the rail draws the sections `facets.attributes` lists, each
  `{param, label, department, match, hint, values[{key, label, n}], n_values}`, from the
  registry `app/attributes.py` (`FACETS`); labels are `app/catalog_taxonomy.py`'s fixed
  vocabulary, never seller text. A new section is a registry entry, not UI code.
- Electronics: Category (`category`, 21 shop groups), Item type (`item_type`: Devices /
  Accessories / Parts & components / Batteries & supplies; protection plans never offered),
  Works with (`works_with`, 10 keys: apple, samsung, google, amazon, microsoft, playstation,
  nintendo, pc, camera, fitness; the "Amazon" label is an owner question, safety.md),
  Condition (`condition`: "Renewed / refurbished", "Used"; hint "As the listing words it").
  Grocery: Category (`category`, 18), Dietary (`dietary`, 15 values, e.g. "Sugar free / no added
  sugar"; match ALL; hint "As the listing words it, not a certification").
- API side (contract: `app/attributes.py`, `app/main.py` `search()`): csv OR repeated, the join
  capped at 500 chars; an unknown key, another param's key, a non-string or > 500 chars is a 422
  before any cache read or LLM call; applied after the cache lookup (0 LLM calls), never in the
  page_cache key; `filters_applied` echoes them. `n` counts ROWS of the unfiltered pool; a value
  is offered with >= 1 on-target row; at most 30 shown.
- Which sections: the focus department = the one picked department, else Fashion when only a
  gender is picked, else the API's `primary_department` (the department of >= 50% of the
  on-target rows), else none -- no attribute section until the shopper picks a department.
  Fashion's own Category / Style and Fabric show only for focus Fashion or none.
- Rail order: Department, the attribute sections, Category / Style (Fashion), Customer
  Reviews, Price, Brands, Colour, Fabric (Fashion).
- Checkboxes: 8 in view, the rest under "See more (n)"; the hint is a tooltip (ⓘ) on the
  heading; keys are digests `f_a_<param>_<seq>_<digest>`; the API gets the KEY, never a label.
  A section with `max_picked` (the API sends none today) disables the other boxes at the cap,
  with the constant caption "Up to N at a time: untick one to pick another."
- Match: an "any" section widens with each tick. A section whose `match` reads "all" (Dietary)
  shows the constant "Items match every option you tick." under its heading (`MATCH_ALL_NOTE`,
  `st.html` + `esc`, colour `--cas-text-2`), else a shopper reads "Vegan" + "Kosher" as either
  and wonders why the page shrank. A leftover ticked Dietary keeps the note; "any" sections are
  byte-identical to before. The wording is provisional (owner).
- State: `filters["attr"]`, sent as a sorted csv with `follow_up=true`. Any department change
  -- and a gender link with no department -- clears it, and the boxes always mirror the applied
  state, never a kept widget value. A ticked section the new pool lost stays drawn, ticked,
  labelled from a per-surface memory (`_attr_text`): the page is never filtered by something
  the shopper cannot see or undo. A picked Category, Style or Fabric stays drawn when the
  focus leaves Fashion.
- Sub-nav gating: on results the sub-nav hides a department shortcut the pool lacks and a
  Women / Men / Kids shortcut whose own value the `genders` facet lacks (on a Grocery page
  each of those had returned "No products match these filters"); the active shortcut always
  stays; before any search all seven stay (they set the next search's scope). The per-gender
  part goes beyond D4's "hide when empty": lead sign-off pending; revert = one line in
  `subnav()` (ui_common.py:1318).
- Owner questions: the per-gender gating; keeping ticks on a department click that does not
  move the focus; the Dietary hint as a tooltip, not a caption; the note wording; the watch
  gift's `primary_department` flips with the intents (Electronics 52% on one intent set,
  Fashion 88% on another), and ticking Category "Watches" there drops the 112 Fashion
  wristwatches; a picked Fabric value the pool lacks shows its raw key ("linen").

## Palette (2026-10-26: eBay-style, colours only)
Sai: "looks too much Amazon, use eBay color grading but don't change anything else". Source:
eBay's open-source design tokens (eBay Skin, MIT, archived and continued as eBay/evo-web, MIT:
https://github.com/eBay/evo-web/blob/36905785fec740849b8a9a68d03324a66fef23ca/packages/skin/dist/tokens/evo-light.css);
"derived" values = eBay's 4% hover layer over the base. Contrast (computed when the palette was chosen): every text
pair >= 4.5:1, every control edge >= 3:1, except the WCAG-exempt disabled pager label
(`disabled-text` #8F8F8F, 3.23:1; eBay's disabled #C7C7C7, 1.69:1, was the first pick and all but
vanished) and unfilled star; decorative hairlines (`divider`) not counted. User avatar: white icon
on `avatar-user` 4.86:1.
Never eBay's (or Amazon's) name, logo, four-colour wordmark, marks or Market Sans font.
- SKIN, 2026-10-19 value -> 2026-10-26 value (the REVERT record): `belt` #131921 -> #FFFFFF;
  `belt-2` #232F3E -> #F7F7F7; `belt-fg` #FFFFFF -> #191919; `belt-muted` #CCCCCC -> #707070;
  `search-btn` #FEBD69 -> #0968F6; `search-btn-hover` #F3A847 -> #0964EC; `scope-bg` #E6E6E6 ->
  #F7F7F7; `scope-border` #CDCDCD -> #8F8F8F; `focus` #FF9900 -> #005FCC; `cart-count` #F08804 ->
  #D50B0B; `page` #FFFFFF (unchanged); `page-grey` #EAEDED -> #F7F7F7; `text` #0F1111 -> #191919;
  `text-2` #565959 -> #707070; `link` #007185 -> #0968F6; `link-hover` #C7511F -> #707070; `query`
  #C7511F -> #191919; `border` #D5D9D9 -> #8F8F8F; `divider` #E7E7E7 -> #E5E5E5; `img-bg` #F7F8F8
  -> #F7F7F7; `star` #FFA41C -> #191919; `star-off` #E3E6E6 -> #C7C7C7; `cart-btn` #FFD814 ->
  #FFFFFF; `cart-btn-border` #FCD200 -> #0968F6; `cart-btn-hover` #F7CA00 -> #F7F7F7;
  `cart-btn-active` #F0B800 -> #F5F9FF; `buy-btn` #FFA41C -> #0968F6; `buy-btn-border` #FF8F00 ->
  #0968F6; `buy-btn-hover` #FA8900 -> #0964EC; `pill-bg` #F0F2F2 -> #F7F7F7; `pill-hover` #F7FAFA
  -> #F5F5F5; `disabled-text` #6F7373 -> #8F8F8F; `backtotop` #37475A -> #E5E5E5;
  `backtotop-hover` #485769 -> #DCDCDC; `footer-fg` #DDDDDD -> #707070; `footer-2` #131A22 ->
  #FFFFFF; `swatch-border` #888C8C -> #8F8F8F; `bubble-assistant` #F0F2F2 -> #F7F7F7;
  `bubble-user` #E6F2F4 -> #D4E5FE; `tab-active` #E77600 -> #191919; `notice-bg` #F0F8FA ->
  #F7F7F7; `shadow-bar` rgba(15,17,17,.15) -> rgba(0,0,0,.15); `shadow-pill` rgba(213,217,217,.5)
  -> rgba(0,0,0,0) (eBay's buttons carry no drop shadow). NEW: `search-btn-fg` #FFFFFF,
  `cart-btn-fg` #0968F6, `buy-btn-fg` #FFFFFF, `search-border` #363636, `avatar-user` #0968F6.
- `_SHOP_CSS` (colour declarations, plus added edges that are box-shadows or a background colour,
  none with a layout effect): the search button and its magnifier read
  `search-btn-fg` (were `text`); the shared Add to cart / Buy Now text rule is split into two
  rules with the same selectors, reading `cart-btn-fg` / `buy-btn-fg` (both were `text`); the
  quantity stepper outline reads `border` (was `cart-btn`, now white). ADDED edges:
  `box-shadow:0 0 0 2px var(--cas-search-border)` on `.st-key-cas_searchrow` (the
  `:focus-within` 3px focus ring replaces it); after the visual review,
  `box-shadow:0 1px 0 var(--cas-divider)` on `.st-key-cas_masthead` (>= 640 px, the sticky band)
  and on `header[data-testid="stHeader"]` (< 640 px only: on a phone the band scrolls away; on
  desktop a line there would split the wordmark row from the search row) -- white content had
  slid under the white band with no edge; `box-shadow:0 0 0 1px var(--cas-divider)` on
  `.st-key-cas_cartlist` / `.st-key-cas_cartsub` (white panels on #F7F7F7 had flattened); the
  Assistant's `[data-testid="stChatMessageAvatarUser"]` background reads `avatar-user` (it was
  Streamlit's default red #FF4B4B before and after the reskin: no theme key here sets it).
  Not changed: the white licence strip (`footer-2`) between the #F7F7F7 footer columns and the
  #F7F7F7 cart page below it.
- Theme (.streamlit/config.toml = run_app.STREAMLIT_THEME_FLAGS): primaryColor / linkColor
  #007185 -> #0968F6, textColor #0F1111 -> #191919, borderColor #D5D9D9 -> #8F8F8F, orangeColor
  #FFA41C -> #191919 (the rail's stars and, in Streamlit 1.58, the Assistant's bot avatar
  background: orange -> near-black), secondaryBackgroundColor #FFFFFF -> #F7F7F7 (eBay's input
  fill; also tints the phone sidebar and rail expander header; mirrored and test-pinned between
  config.toml and the App flags by `test_light_theme_on_both_launch_paths`, tied to SKIN `pill-bg`
  by `test_skin_is_the_theme`); the rest unchanged. Since 2026-10-27 the App flags also mirror
  `[theme.sidebar] backgroundColor` #FFFFFF (the mirror test flattens nested tables;
  `test_skin_is_the_theme` ties it to SKIN `page`).
- Checked 2026-10-26: all suites; a fixed-step headless capture (9 screens x 1280 / 375 px, fake
  API on spare ports) before vs after, and again before vs the review fixes: 0 geometry, text,
  layout-line or non-colour-style differences (the fixes changed only shadows, the user avatar
  and the disabled pager label). NOT checked: a visible browser, Sai's eye.

## Browser-check record (what was and was NOT checked)
- Browser-checked 2026-10-19 21:31-21:53 (screenshot file times) on the integrate build,
  against a local test API (gpt-oss-20b), at 1280 and 375 px: no horizontal page scroll seen.
  That was BEFORE the late 2026-10-19 UI fixes (sub-nav gating, the department-tree parent
  click, keeping the page on a 4xx, the weak banner on an empty page, the brand cap): none of
  those was browser-checked.
- 800 px was checked only against a FAKE API (recorded responses), which carries no
  X-LLM-Calls header.
- "X-LLM-Calls 0 on every filter / sort / page request" is NOT recorded anywhere:
  `logs/api.log` logs only requests that made LLM calls (tests assert 0 calls offline).
- 2026-10-27 round (Related items, rail sections, Dietary note, sub-nav gating, brand line):
  NOT browser-checked at all. Offline only: test_ui 74 (AppTest + fake HTTP; its synthetic
  `facets.attributes` fixtures are checked against `app.attributes` by
  `test_rail_fixtures_are_what_the_api_can_send`), and the API half on the real catalog (119
  offered values, then 75 in the merge smoke, each a valid filter, 0 mismatches, 0 LLM calls).
  AppTest runs no CSS or JS, so unchecked: the pill's look and the stacked buttons at 375 / 800
  / 1280 px, where the Related jump lands under the sticky band (132 / 72 px), the Assistant's
  stick-to-bottom container with the late scroll, Back from a Related open, the note's
  spacing, and the rail sections on screen.
- Before a demo: one manual pass on the FINAL build in a VISIBLE browser (CLAUDE.md, Gotchas:
  Streamlit 1.58).
- Browser automation: after an eval-driven reload the first synthetic Search submit or Enter
  is often lost -- retry, or check in a visible browser.
