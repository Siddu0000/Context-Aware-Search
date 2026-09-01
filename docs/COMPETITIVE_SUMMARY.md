# CAS vs Market — Comparison Summary (with sources)

Prepared for Niharika · 2026-08-24. Condensed from the full fact-checked study
(docs/COMPETITIVE_STUDY.md). Every product row links to the official page it was
verified against; pricing is only quoted where published. Confidence flags kept
where a claim could not be re-verified.

## A. Scope decisions from the 2026-08-24 call

| Gap / capability | Decision |
|---|---|
| Online A/B testing | **Out of scope** — another team owns this |
| Purchase-history personalization | **On hold** — scalability + returns/cancellation staleness risk (POS lag); revisit maybe as premium tier |
| Store-level / real-time inventory | Not now — data doesn't support it; substitution partially covered by existing similar-products + grocery 3-option carousel |
| Visual / image search | **Not in scope** |
| Catalog attribute enrichment (persisted) | **Do it** — one-time (or seasonal) AI enrichment of catalog fields; partially done already |
| Analytics dashboard | Later — after Databricks (track conversion/effectiveness) |
| Multilingual / multi-market | No — not now |
| Autocomplete / typeahead | Only if a simple add-on; otherwise hold |
| GEO / agent-channel readability | On hold — belongs to the agentic-commerce workstream |
| Multi-turn conversational refinement | **Integrate** — done: session refinement now retains relevant results and updates in place |
| Sub-second latency | Experiment after Databricks migration |
| Retrieval diversity / gender skew | **Definite fix** — done: gender-balanced retrieval shipped 2026-08-24 |
| Enterprise SLA / SOC 2 | Not a priority now |

## B. Feature matrix — what we have vs what we don't

| Capability | CAS | Who ships it |
|---|---|---|
| Semantic / vector retrieval over the product catalog | ✅ | Algolia (NeuralSearch, Elevate tier only), Empathy.co, Constructor, Elastic (ELSER) |
| Hybrid keyword-first routing with AI fallback on lexical miss | ✅ | Empathy.co (AI Mode), Algolia (NeuralSearch, single index), commercetools (Lexical/Semantic/Hybrid with RRF), Google (intent classifier routes simple vs complex) |
| Typed hard-constraint extraction that caps the rerank score ceiling | ✅ | _none found in 18 vendors checked_ |
| Per-result human-readable generated reason on every ranked product | ✅ | Constructor (Merchant Intelligence Agent, Beta — merchant-facing, not shopper-facing) |
| Recipe / meal query decomposed into per-INGREDIENT cards with up to 3 brand alternatives | ✅ | Instacart Cart Assistant (recipe to cart, auto-filled), Northfork (ingredient to single in-stock item), Constructor (dietary-needs missions), Google Conversational Commerce |
| Published, reproducible offline eval harness (P@10 / MRR / NDCG@10 + recipe completeness + injection/robustness) | ✅ | _none found in 18 vendors checked_ |
| Day-one cold-start ranking of newly added SKUs from content alone | ✅ | Marqo (markets it explicitly) |
| Cross-sell / upsell grounded in the real catalog | ✅ | Stylitics (outfitting, inventory-aware, human QA), Attraqt/Crownpeak XO (AI + co-piloted business rules), Rebuy (from $25/mo), Nosto |
| Relevance-gated sponsored ads, never blended into organic, bid never exposed | ✅ | 84.51°/Kroger Boosted Products in Search (relevance + inventory-aware ad quality score, since 2018), Criteo x Albertsons Media Collective (sponsored products INSIDE AI search carousels, 2026), Instacart Carrot Ads, Miso.ai (ads inside generative answers) |
| Merchandiser console (boost / bury / pin / rules / synonyms) usable without engineering | ❌ | Coveo Merchandising Hub, Algolia Visual Merchandising Studio, Empathy.co Playboard, Lucidworks Commerce Studio |
| Online A/B testing and experimentation | ❌ | Prefixbox (every paid tier), Fast Simon (Top Pro, $299.99/mo), Coveo, Lily AI (core proof mechanism) |
| Session / purchase-history personalization | ❌ | Algolia (real-time, Elevate), Constructor (clickstream-native), Bloomreach (CDP-fed), Coveo (incl. first-time visitors) |
| Store-level / real-time inventory awareness in ranking | ❌ | Constructor, Algolia (H-E-B México: real-time inventory by store), Prefixbox (FMCG Multistore), Instacart |
| Out-of-stock substitution engine | ❌ | Instacart (Siamese two-tower, >95% candidate recall), Wynshop/Halla (a whole company built on search + substitute + recommend), Algolia Intelligent Grocery Solution, Constructor |
| Visual / image search (photo input, shop-the-look, reverse image) | ❌ | Syte (named at C&A), Google Lens / AI Commerce Search (multimodal text+image), Athos Commerce Conversational Assistant, Netcore Unbxd |
| Persisted catalog attribute enrichment (written back, not query-time only) | ❌ | Lily AI (the entire business), Constructor (pack size / volume normalization), Stylitics (Catalog Enrichment + published Lily AI comparison), Syte (15,000-attribute lexicon) |
| Analytics dashboard for business users | ❌ | Algolia (advanced analytics: CTR, CVR, click position), Coveo, Constructor, Prefixbox (2-year history) |
| Multilingual / multi-market | ❌ | Prefixbox (90+ languages), Coveo (50+ languages, 50+ markets), Doofinder, Luigi's Box |
| Autocomplete / typeahead / query suggestions | ❌ | Algolia, Constructor (auto-generated typo/synonym/reformulation), Coveo, Bloomreach |
| GEO / agent-channel readability (catalog legible to ChatGPT, Gemini, AI shopping agents) | ❌ | Athos Commerce GEO Assistant, Lily AI, commercetools AI Hub, Bold Metrics (Agentic Sizing Protocol) |
| Multi-turn conversational discovery with session memory | 🟡 | Empathy.co AI Mode, Constructor AI Shopping Agent, Google Conversational Commerce (cross-device context), Instacart Cart Assistant |
| Sub-second latency on the AI path | ❌ | Algolia (~20ms cited by MILKRUN), Instacart (300ms tail-query target after LoRA merge + H100), Google, Adobe |
| Retrieval diversity / dedup ('70 paneer sellers') | ❌ | Instacart (published diversity-based reranking), Constructor (attribute + pack-size normalization) |
| Enterprise plumbing: SLA, SSO, SOC 2, data residency | ❌ | Algolia (99.99% + SSO, Elevate), Coveo, Lily AI (SOC 2 Type II, SAML/OAuth SSO, RBAC), Stylitics (99.99% uptime) |

## C. Vendor quick reference (official links)

### Enterprise e-commerce search & product-discovery platforms

| Product | Official link | Published pricing | Named clients (verified) |
|---|---|---|---|
| Algolia (NeuralSearch + Agent Studio) | https://www.algolia.com/ | PUBLISHED — RE-VERIFIED 2026-08-13 at algolia.com/pricing, all figures confirmed exactly. Free: $0/mo, 10K search requests/mo, 50K records, 5K Recommend requests/mo, 5K crawls/mo.  | H-E-B México (grocery, 88+ stores across; Co-op (UK grocery); Auto Mercado (Costa Rica supermarket, 44; Lacoste (fashion) |
| Constructor | https://constructor.com/ | NOT PUBLISHED — quote-only. RE-VERIFIED at vendr.com/marketplace/constructor-io: average contract value ~$150,000, redline threshold ~$200,000, proposed price point ~$300,001. Vend | VERIFIED as logos/mentions on constructo; Target Australia; Belk (department store); Bonobos (menswear) |
| Bloomreach (Discovery / Clarity / Loomi AI) | https://www.bloomreach.com/ | NOT PUBLISHED. Per the original report's verification of bloomreach.com/en/pricing/discovery and /pricing/clarity: subscription = module fee + usage fee, annual billing only, decli | Albertsons (US GROCERY incl. Safeway, Vo; STALE-REFERENCE WARNING (new, and strate; Boohoo Group (fashion); Canadian Tire |
| Coveo | https://www.coveo.com/ | NOT PUBLISHED, but METERING is published, which is unusually useful. Per the original report's verification of coveo.com/en/pricing: standard unit = 100,000 queries per month; Pass | Freedom Furniture (AU/NZ); Caleres / Famous Footwear (footwear-fash; FleetPride (B2B truck parts); SAP |
| Netcore Unbxd | https://netcoreunbxd.com/ | NOT PUBLISHED. No public plans; custom quotes built from search volume, traffic and feature set. No credible Vendr/marketplace benchmark surfaced in either pass. Reputationally pos | New York & Company (fashion) +8% CVR; Ca; Deborah Lippmann (beauty) +40% conversio; City Furniture +20% CVR; Jerome's Furnit; Backcountry +11% search session demand;  |
| Lucidworks (Fusion / Lucidworks Platform / Commerce Studio) | https://lucidworks.com/ | NOT PUBLISHED. Metering model (per third-party summaries): Fusion for Commerce priced by REQUESTS PER YEAR (RPY); Fusion for the Workplace by records indexed. The ITQlick estimate  | Coppel (Mexican department-store retaile; Lenovo; Mouser Electronics and TE Connectivity; KILLED / DO NOT USE |
| Attraqt / Crownpeak Product Discovery (Fredhopper + XO) — now owned by Rezolve AI | https://www.crownpeak.com/fredhopper/ | NOT PUBLISHED. Their Shopify app 'Fredhopper Product Discovery' (listed 2025-07-25) is FREE TO INSTALL with charges billed separately by Crownpeak outside the Shopify invoice — a s | ASOS, JD Sports, Screwfix, Calvin Klein,; KILLED; KILLED |
| Google Cloud — AI Commerce Search (formerly Vertex AI Search for commerce / Retail Search) | https://cloud.google.com/solutions/retail-product-discovery | NOT VERIFIED. Google Cloud publishes retail/search pricing pages but every fetch attempt this pass returned 404 or truncated content (cloud.google.com/retail/pricing, /vertex-ai-se | Albertsons Companies; Google's own MQ blog names NO specific c; Separately reported (search-snippet leve |
| Empathy.co (Empathy Platform / Playboard / APISearch / Motive) | https://empathy.co/ | NOT PUBLISHED. Three named tiers (Empathy Platform / APISearch / Motive) segmented by customer size, but no public prices and no Vendr or marketplace benchmark found. Do not quote  | Kroger (US GROCERY); Carrefour (grocery); Fashion/accessories: Tous (jewellery), P; Other: BSH (home appliances), Casa del L |
| Athos Commerce (consolidation of Searchspring + Klevu) | https://athoscommerce.com/ | NOT PUBLISHED on the Athos site. Klevu and Searchspring historically both had published or semi-published mid-market pricing, which makes the disappearance of public pricing notabl | Michael Stars (fashion/apparel); No grocery customers identified on the s; Klevu's and Searchspring's own historic  |

### AI-native / mid-market / SMB product-search vendors and newer LLM-era ("agentic") entrants

| Product | Official link | Published pricing | Named clients (verified) |
|---|---|---|---|
| Empathy.co | https://empathy.co/ | NOT PUBLISHED anywhere. Three named tiers (Motive -> APISearch -> Empathy Platform) imply a small/mid/enterprise ladder, but no figures are disclosed on any page fetched. Do not es | Kroger (VERIFIED; Carrefour (VERIFIED; Toys R Us (VERIFIED; Vodafone (VERIFIED |
| Constructor | https://constructor.com/ | NOT PUBLISHED. No pricing disclosed on the site; enterprise sales motion. Do not estimate. | Sephora; Under Armour; Gap; REI |
| Lily AI | https://www.lily.ai/ | NOT PUBLISHED. A free 30-day trial on 500 products is offered; no figures disclosed. | Coach; M&S (Marks & Spencer); J.Crew; Kate Spade |
| Athos Commerce (Klevu + Searchspring + Intelligent Reach) | https://athoscommerce.com | MIXED, and the published half is VERIFIED EXACTLY. Re-fetched from https://apps.shopify.com/klevu-smart-search : Site Search $649/mo (includes 50k search requests), Recommendations | adidas; Yamaha; Clarins; Samsung |
| Prefixbox | https://www.prefixbox.com/en-us/ | PUBLISHED ENTERPRISE PRICES — RE-VERIFIED EXACTLY from https://www.prefixbox.com/en-us/site-search-pricing-for-ecommerce : AI Search EUR 1,250/mo; AI Navigation EUR 550/mo add-on;  | Carrefour; Auchan; Rossmann; Leroy Merlin |
| Instacart Cart Assistant (Instacart Enterprise AI) | https://company.instacart.com/enterprise-blog/cart-assistant-from-instacart | NOT PUBLISHED — enterprise partnership motion, almost certainly bundled into the broader Instacart retailer relationship rather than sold as a line item. That bundling is itself th | Kroger; Sprouts Farmers Market; Good Food Holdings; McKeever's |
| Cooklist | https://www.grocerydive.com/news/cooklist-agentic-ai-grocery-shopping-supermarkets-kroger-wegmans/822950/ | NOT DISCLOSED in the coverage. | Wegmans (VERIFIED); Kroger, including the Fred Meyer, Ralphs; Live across 700+ US stores, with expansi; Plus 10 additional regional and national |
| Luigi's Box | https://www.luigisbox.com/ | NOT PUBLISHED — quote-based, priced on usage with a 30-day free trial and no separate charge for core products; base price depends on website usage plus catalogue size; three integ | Kosik.cz; Notino; Under Armour; KiK |
| Fast Simon | https://www.fastsimon.com/ | TWO PICTURES, BOTH RE-VERIFIED. Their own pricing page shows Starter/Essential/Growth/Enterprise all as 'Request Pricing' with a free trial. The Shopify listing at https://apps.sho | Steve Madden; Juicy Couture; Hey Dude; Ally Fashion |
| Doofinder | https://www.doofinder.com/en/ | PUBLISHED AND CHEAP — RE-VERIFIED EXACTLY from https://www.doofinder.com/en/price : Basic $49/mo (up to 10k requests/mo per product), Pro $149/mo (150k), Advanced $349/mo (400k), E | Crocs; Volkswagen; Blue Banana; Eurekakids (AI Assistant handles 40% of  |
| Nosto | https://www.nosto.com/ | NOT PUBLISHED as figures, but the MODEL is notable: Nosto's own comparison content states pricing is FIXED BY GMV so merchants use search and discovery WITHOUT QUERY-BASED LIMITS.  | VERIFIED on the CXP page: Muji, Belstaff; DOWNGRADED: UNIQLO and Gymshark. Both we |
| Marqo | https://www.marqo.ai/ | NOT PUBLISHED — marqo.ai/pricing redirects to a book-a-demo page. Open-source self-hosted is free (Apache 2.0); Marqo Cloud is usage-based with a free trial. | Mejuri (+19.8% search revenue per user); Redbubble (+$11M incremental revenue); KICKS CREW (+17.7% conversion uplift); Shutterstock (+23% search satisfaction) |
| Algolia | https://www.algolia.com/pricing | PUBLISHED AND FULLY USAGE-BASED — VERIFIED from https://www.algolia.com/pricing : Free (10K search requests/mo, 50K records, 5K recommendation requests, 5K crawls, no credit card). | Not extracted from the pricing page I fe |
| Syte | https://www.syte.ai/ | NOT PUBLISHED. No pricing on the site; enterprise/mid-market sales motion. | Prada; Farfetch; PrettyLittleThing; Decathlon |
| GroupBy (now absorbed into Rezolve AI) | https://www.groupbyinc.com/ | NOT PUBLISHED and not verifiable following the redirect. | NOT VERIFIED THIS SESSION |
| Searchspring (a division of Athos Commerce) | https://searchspring.com | NOT PUBLISHED by the vendor. The original report cited third-party trackers quoting roughly $599-1,099/mo. I could not verify those against any Searchspring or Athos page. RECOMMEN | Hat Club; SuperATV; MacSales; Costumebox |
| Findify (now Maropost Merchandising Cloud) | https://maropost.com/platform/merchandising-cloud | PUBLISHED, RE-VERIFIED EXACTLY from https://apps.shopify.com/findify-search : Premium $499/mo ($5,400/yr with 10% discount — AI personalized search and autocomplete, up to 100k vis | Historic (from acquisition-era coverage); Current Maropost page names much smaller |
| Vantage Discovery (acquired by Shopify) | https://www.crunchbase.com/organization/vantage-discovery | N/A — the standalone product is gone; capabilities surface inside Shopify's native search rather than as a purchasable third-party product. | Not disclosed by name; Effectively now: Shopify, as acquirer |
| Swap (swap-commerce.com) | https://www.swap-commerce.com/storefront | NOT PUBLISHED. | NONE DISCLOSED in the coverage |
| Miso.ai | https://miso.ai/ | NOT PUBLISHED as figures, but the MODEL is stated and unusual: 'a simple monthly fee that gives you unlimited API requests and responses' with no token-based charges (https://docs. | O'Reilly; PCWorld; Macworld; TechHive |
| Rebuy | https://www.rebuyengine.com/ | FULLY PUBLISHED AND THE CHEAPEST HERE — RE-VERIFIED EXACTLY from https://www.rebuyengine.com/pricing : Rebuy Monetize free (merchant earns $0.20-$0.35+ per transaction, est. $175/m | NONE ASSERTED. No named brands appear on |
| Daydream and the D2C agentic cohort (Phia, OneOff, Gensmo) | https://www.modernretail.co/technology/why-the-ai-shopping-agent-wars-will-heat-up-in-2026/ | N/A — consumer products, not licensed software. | N/A |

### GROCERY retail search & discovery (P0 target vertical)

| Product | Official link | Published pricing | Named clients (verified) |
|---|---|---|---|
| Instacart (AI Solutions / Cart Assistant / Catalog Intelligence / Carrot Ads / Storefront Pro / Caper Carts) | https://www.prnewswire.com/news-releases/instacart-announces-new-enterprise-ai-solutions-to-democratize-ai-for-grocers-of-all-sizes-302603735.html | NOT PUBLISHED. Carrot Ads launched 2022-03-24 on an explicit revenue-sharing model with Instacart (percentage undisclosed); pilot retailers were Schnuck Markets, Good Food Holdings | VERIFIED as AI Solutions launch retailer; VERIFIED scope detail: Sprouts is first ; VERIFIED on the Carrot Ads page itself; DOWNGRADED: 'McKeever's (Store View)' co |
| Google Cloud — Conversational Commerce agent on Vertex AI (+ Gemini Enterprise for CX) | https://www.googlecloudpresscorner.com/2025-09-10-Google-Cloud-Launches-Conversational-Commerce-Agent,-Delivering-AI-Enabled,-Personalized-Shopping-Experiences-for-Customers | Not published — 'contact sales'; consumed via the Vertex AI console. Do not quote a figure. | VERIFIED; VERIFIED |
| OpenAI (as the model vendor behind the Albertsons AI Shopping Assistant) | https://www.albertsonscompanies.com/newsroom/press-releases/news-details/2025/Albertsons-Companies-Accelerates-Digital-Transformation-with-the-Albertsons-AI-Shopping-Assistant-Redefining-the-Grocery-Shopping-Experience/default.aspx | Not published; model/API consumption plus internal build cost. Do not quote a figure. | Albertsons Companies (Albertsons, Safewa |
| Empathy.co | https://empathy.co/customers/ | Not published — enterprise sales. | VERIFIED on Empathy.co's own customers p; GROCERY RELEVANCE: Kroger and Carrefour ; CAVEAT: these are logos on the vendor's  |
| Ocado Technology — Ocado Smart Platform (OSP) | https://www.ocadogroup.com/about-us/osp-partners | Not published — long-term platform partnership contracts with upfront fees and ongoing fees typically tied to capacity/volume. Do not quote figures. | VERIFIED on Ocado Group's own OSP partne |
| Algolia — Intelligent Grocery Solution | https://www.algolia.com/about/news/algolia-unveils-intelligent-grocery-solution | PUBLISHED — rare in this segment, and VERIFIED directly on https://www.algolia.com/pricing. Free: 10K search requests/month, 50K records, 5K recommendation requests/month, 5K crawl | VERIFIED with exact quotes; VERIFIED with exact quote; KILLED: Co-op |
| Constructor | https://constructor.com/customers | Not published — demo/contact only. Corporate growth claims (82% customer growth in FY26, 322 billion shopping interactions) come from a PRNewswire release and are vendor-stated. | VERIFIED on constructor.com/customers wi; VERIFIED logos also on the page: Sephora; CORRECTION TO THE ORIGINAL DRAFT: it cla |
| Wynshop + Halla ('Taste Intelligence') | https://wynshop.com/wynshop-acquires-grocery-ai-pioneer-halla/ | Not published. | Wynshop customer base per trade coverage; CAVEAT: these are WYNSHOP platform custo |
| Chicory | https://chicory.co/ | Not published — CPG-media-funded model (brands pay; retailers integrate). INFERENCE: no licence fee to the retailer, which is why it spreads fast. Flag as inference. | VERIFIED on chicory.co: Giant Eagle, Gen; VERIFIED via trade coverage: Ahold Delha |
| Northfork | https://northfork.ai/ | Not published; API integration required, enterprise sales. | VERIFIED on northfork.ai: Walmart, Sains; Sainsbury's is the only UK grocer found  |
| Samsung Food (formerly Whisk) | https://www.grocerydive.com/news/kroger-whisk-team-up-on-shoppable-lists/587247/ | Not published for the B2B API. Consumer app is free. | VERIFIED; Walmart, Amazon Fresh, Ahold Delhaize an |
| SideChef | https://www.sidechef.com/business/recipe-platform/shoppable-recipe-button-comparison | Not published. | CORRECTION TO THE ORIGINAL DRAFT, which ; These are CPG brands, not grocery retail |
| Pear Commerce | https://theshelbyreport.com/2026/07/29/ahold-delhaize-usa-adds-shoppable-recipes-product-pages/ | Not published. INFERENCE: CPG-funded like Chicory. Flag as inference. | VERIFIED |
| SmartCommerce (Click2Cart) | https://progressivegrocer.com/ahold-delhaize-usa-rolls-out-smartcommerces-click2cart-capability | Not published. | Ahold Delhaize USA |
| Criteo (retail media inside AI conversational search) | https://www.grocerydive.com/news/albertsons-media-collective-criteo-ai-search-brand-ads-ecommerce-omnichannel/823557/ | Not published (retail-media platform fees / revenue share). | Albertsons Media Collective (Albertsons, |
| 84.51° / Kroger Precision Marketing — Boosted Products in Search + Dynamic Positioner | https://ir.kroger.com/news/news-details/2018/Kroger-Precision-Marketing-Launches-Boosted-Products-in-Search/default.aspx | CPC auction for advertisers; not applicable as a software licence. | Kroger and all its banners (84.51° is Kr |
| Mercatus (DXPro + AisleOne) | https://www.mercatus.com/platform-overview/ | Not published; two tiers exist but no costs disclosed. | NONE VERIFIED. No customer logos on the  |
| Swiftly | https://www.swiftly.com/news/key-food-stores-co-operative-inc-partners-with-swiftly-to-modernize-digital-shopper-experience-and-unlock-retail-media-growth | Not published. INFERENCE: platform fee plus retail-media revenue share. | Key Food Stores Co-Operative (460+ super; Merchants Distributors / MDI (wholesale); UNFI (retail media network launch); Alliance Retail Group; Webstop |
| Firework (AVA) | https://www.grocerydive.com/news/the-fresh-market-generative-artificial-intelligence-video-commerce-firework/649678/ | Not published. | The Fresh Market; Lowe's cited in Firework's own material  |
| NIQ (Label Insight + NIQ Brandbank) — product attribute data for dietary/allergen filtering | https://nielseniq.com/global/en/landing-page/label-insight/ | Not published — content licensing agreements. | United Supermarkets (named on the NIQ Br |
| Tomoro AI (the consultancy behind Tesco's assistant) | https://www.retailgazette.co.uk/blog/2026/04/tesco-trials-ai-shopping-assistant-with-280000-colleagues-ahead-of-customer-rollout/ | Not published (consultancy engagement). | Tesco |
| Bloomreach — CHECKED, NO VERIFIABLE GROCERY REFERENCE | https://www.bloomreach.com/en/customers | Not published. | NONE GROCERY VERIFIED. The customers pag; DO NOT assert Bloomreach grocery referen |
| Lily AI — CHECKED AND EXCLUDED FROM THIS SEGMENT | https://www.lily.ai/ | Not published. | Named on lily.ai with testimonials: M&S,; VERIFIED NEGATIVE: the site contains no  |
| Bringg — OUT OF SCOPE, RECOMMEND DROPPING | https://www.bringg.com/ | Not published. | Walmart, Best Buy, Coca-Cola cited in th |

### Fashion retail search & discovery (P0 vertical)

| Product | Official link | Published pricing | Named clients (verified) |
|---|---|---|---|
| Daydream ("Powered by Daydream") | https://www.prnewswire.com/news-releases/daydream-launches-ai-powered-search-and-discovery-solution-for-fashion-brands-and-retailers-302837597.html | NOT PUBLISHED — not disclosed in the launch announcement. [V] Funding context: $50M seed round announced 2024-06-20, investors including Forerunner Ventures, Index Ventures, Google | LIVE on Powered by Daydream at launch: S; SIGNED to join: Anine Bing, Mansur Gavri; NOTE the tier: these are contemporary/pr |
| Syte | https://www.syte.ai/ | NOT PUBLISHED. Enterprise quote-only; no self-serve tier. [V] *** CORRECTION *** The original report cited "a comparable visual-search widget (ViSenze) at $200–500/mo up to 50K SKU | C&A; Also confirmed named on the fashion page; Company-level: "more than 100 customers ; *** CORRECTION TO THE ORIGINAL REPORT ** |
| Lily AI | https://lily.ai/industries/fashion | NOT PUBLISHED. Enterprise contract, sales-led; page directs to "Book a Scoping Call." No tiers or ranges disclosed. [V] Company scale [A — NOT deck-safe]: Latka lists ~$17.6M ARR;  | CONFIRMED on the fetched Lily AI page: C; Tapestry (Coach's parent) reported "doub; *** CORRECTION; Attributed testimonials confirmed on-pag |
| Athos Commerce (Klevu + Searchspring) | https://athoscommerce.com/ | *** THIS CORRECTS A KEY CLAIM IN THE ORIGINAL REPORT *** Third-party sources cite Klevu pricing "from $449/month with a free trial" [S], but that is legacy Klevu data. The CURRENT  | Klevu's pre-merger fashion customer base; Searchspring historically served mid-mar; NOTE: I did not open a primary Athos cli |
| Constructor / Bloomreach / Algolia (incumbent commerce search platforms) | https://constructor.com/blog/forrester-wave-commerce-search-product-discovery-solutions-q3-2025 | NOT PUBLISHED for any of the three; enterprise, sales-led, typically catalog- and traffic-scaled. | Constructor, CONFIRMED on its own site: ; Constructor published case: "How Petco d; Bloomreach is the platform into which Li; *** CORRECTION *** The original cited "C |
| Stylitics | https://stylitics.com/ | NOT PUBLISHED, but the MODEL is disclosed and remains the most useful pricing datapoint in the segment: "Our pricing is customized to your catalog size, channels, and business goal | Published on Stylitics' own site with fi; Madewell [V]; From the $80M Series C press release: "m; *** CORRECTION *** The original said "mo |
| Crownpeak / Attraqt (incl. Fredhopper) | https://www.crownpeak.com/resources/blogs/crownpeak-enters-agreement-to-acquire-attraqt/ | NOT PUBLISHED; enterprise, sales-led, sold as part of the Crownpeak DXP. | ASOS, JD Sports, Screwfix, Calvin Klein,; 300+ brands, manufacturers and retailers; NOTE: I did not open a Crownpeak primary |
| YesPlz AI | https://yesplz.ai/product-discovery | *** MAJOR CORRECTION — THIS WAS THE ORIGINAL REPORT'S CLEAREST FACTUAL ERROR. *** The original claimed YesPlz "appears to be the ONLY vendor in this set with published pricing" and | WConcept; Mango and Zalando appear in YesPlz marke |
| Marqo | https://www.marqo.ai/blog/improving-search-relevance-in-fashion | Not established. Marqo has open-source roots and a cloud offering, so a self-serve tier likely exists — NOT VERIFIED in either pass. Do not cite a price. | Mejuri (fine jewellery/accessories); *** ADDITIONAL CLIENTS THE ORIGINAL REPO; One unnamed retailer with $130M attribut |
| Empathy.co | https://empathy.co/ | NOT PUBLISHED; no performance metrics on the homepage either. [V] | CONFIRMED on the Empathy.co homepage: KR; *** UNVERIFIED; NOTE: Empathy.co's homepage does NOT men |
| Google — Vertex AI Search for commerce, Google Lens, and AI Mode | https://cloud.google.com/solutions/vertex-ai-search-commerce | Lens: N/A. Vertex AI Search for commerce: usage-based Google Cloud pricing, published in GCP documentation — the only genuinely public pricing in this entire landscape, and worth p | Albertsons (Google Cloud conversational ; Platform-level for Lens: used by over a ; Retailer-side proof that visual search i; *** NEW, UNVERIFIED, AND IMPORTANT IF TR |
| ViSenze (a Rezolve Ai company) | https://www.visenze.com/ai-search-discovery/ | NOT PUBLISHED. *** EXPLICIT CORRECTION *** The original report used "ViSenze at $200–500/mo up to 50K SKUs" as an indicative price band for the segment. I found NO primary source f | Flipkart; Essilor [S]; NOTE: I did not verify these on a ViSenz |
| Fit Analytics | https://www.fitanalytics.com/ | NOT PUBLISHED. Corporate history for context: Snap acquired Fit Analytics in March 2021 for a reported $124.4M, and the company has since decoupled from Snap and operates independe | Alpha Industries; "250+ global partners" claimed [V]; Calvin Klein, Vans, Intersport, The Icon |
| Bold Metrics | https://boldmetrics.com/ | NOT PUBLISHED; demo-request CTAs only, no tiers or figures anywhere on the site. [V] They publish a free "Fit & Sizing Technology Buyer's Guide" at https://info.boldmetrics.com/fit | CONFIRMED on the homepage: New Balance, ; Men's Wearhouse 47.4% average return-rat; NOTE the scale nuance: "25+ brands" is a |
| Vue.ai (Mad Street Den) | https://www.vue.ai/ | NOT PUBLISHED. Claims "At a fraction of the cost. At 5X the speed" without figures. [S] | CONFIRMED on the current homepage: HDFC ; Note how few are fashion retailers; thredUP founder Chris Homer testimonial  |
| Increff | https://www.increff.com/increff-ai | NOT PUBLISHED on the fetched page. [V] Listed on Capterra and GetApp, typically indicating a mid-market SaaS motion with quote-based pricing. | NO named clients and NO performance stat; Company-level claim of "700+ global bran |
| 3DLOOK | https://3dlook.ai/ | Page references pricing but no rates surfaced. Demo/sales-led. | Logos shown for Bravo, Magic Fit, Vevo, ; Claimed retailer outcomes of "4x increas; 3DLOOK's content hub is, however, a wide |

### Cloud/platform search products + the build-vs-buy picture (Google, Azure, AWS, Elastic, Ve

| Product | Official link | Published pricing | Named clients (verified) |
|---|---|---|---|
| Google Cloud — AI Commerce Search (formerly Vertex AI Search for commerce / Agent Search for commerce) | https://docs.cloud.google.com/retail/docs/what-is-it | Consumption-based. $2.50 per 1,000 search & browse requests; predictions/recommendations from $0.27 per 1,000 (first 20M/month tier); training & tuning $2.50 per node-hour; $600 fr | Albertsons Companies; Galeries Lafayette; Adorama, Michaels, Nordstrom; Cotopaxi |
| Microsoft Azure AI Search (underpins Foundry IQ) | https://learn.microsoft.com/en-us/azure/search/agentic-retrieval-overview | FULLY VERIFIED — I re-pulled the live meters from the Azure Retail Prices API (East US) and every figure in the original report checks out exactly. Basic $0.101/SU-hour; Standard S | — |
| AWS — Amazon OpenSearch Service (managed + Serverless + S3 Vectors) | https://aws.amazon.com/opensearch-service/pricing/ | VERIFIED on the AWS pricing page: c6g.large.search $0.113/hr, r6g.xlarge.search $0.335/hr, or1.xlarge.search $0.418/hr. Storage: UltraWarm and OR1 managed storage $0.024/GB-month;  | — |
| AWS — Amazon Personalize | https://aws.amazon.com/personalize/pricing | VERIFIED in full. Data ingestion $0.05/GB. v2 recipes: training $0.002 per 1,000 interactions ingested; real-time AND batch inference $0.15 per 1,000 requests. Custom solutions: tr | — |
| Elastic (Elasticsearch + ELSER) | https://www.elastic.co/industries/retail-ecommerce | VERIFIED on https://www.elastic.co/pricing/serverless-search — every figure in the original report is correct. Search VCU from $0.09/hr, Ingest VCU from $0.14/hr, Machine Learning  | Kroger; H-E-B; Walmart; Home Depot |
| Vespa.ai | https://vespa.ai/industries/e-commerce/ | VERIFIED on https://cloud.vespa.ai/price-calculator.html — every figure in the original report is exact. Startup $0.05/vCPU-hr, $0.005/GB-mem-hr, $0.0002/GB-disk-hr, $0.03/GB-GPU-m | Vinted; Kleinanzeigen; Groupon; Decathlon |
| Databricks — Mosaic AI Vector Search (+ Agent Bricks) | https://www.databricks.com/product/pricing/vector-search | VERIFIED on the pricing page. Standard endpoint 4.00 DBU/hour supporting ~2M vectors at 768 dimensions; Storage-Optimized 18.29 DBU/hour supporting ~64M vectors at 768d. Effective  | Grupo Casas Bahia; 7-Eleven; Burberry; Coop |
| Shopify — Search & Discovery (native semantic search) | https://help.shopify.com/en/manual/online-store/storefront-search/search-and-discovery-modify-search | FREE (first-party Shopify app) — VERIFIED on the App Store listing. Gating is by plan and catalog size, not price, and ALL THREE LIMITS ARE VERIFIED VERBATIM in Shopify's help docs | — |
| commercetools — Storefront Search (Product Catalog) + AI Hub | https://commercetools.com/blog/b2b-product-spotlight-storefront-search | Not published. commercetools uses consumption-based pricing keyed on ORDER VOLUME plus SKU count, API call volume and project complexity; editions are Core, Foundry and Premium. Th | CORRECTION |
| SAP Commerce Cloud — Search Service + Intelligent Selling Services | https://learning.sap.com/courses/transforming-search-dynamics-using-search-service-and-intelligent-search-in-sap-commerce-cloud/understanding-intelligent-selling-services | Not published. IMPORTANT CORRECTION TO A LOAD-BEARING CLAIM: the original report asserted that customers 'must PURCHASE A SEPARATE SEARCH ADD-ON' and cited the SAP learning page fo | Galeries Lafayette |
| Adobe Commerce — Live Search (GAP FILL: missed entirely by the original report) | https://experienceleague.adobe.com/en/docs/commerce/live-search/overview | VERIFIED: the Adobe docs state Live Search is 'included in your license' — it is not a separate paid add-on for Adobe Commerce customers. Adobe Commerce licence pricing itself is n | — |
| Salesforce Commerce Cloud — Einstein Search / Agentforce | https://kovil.ai/agentforce/playbook/agentforce-pricing-guide-2026 | HANDLE WITH CARE — SOURCING DOWNGRADED. Salesforce.com returned HTTP 403 to every fetch I attempted, including the pricing pages the original report cited as sources, so NONE of th | — |

## D. Where the detail lives

Full write-ups per vendor (features, fashion/grocery fit, demo/trial links, and
what each does that CAS does not) are in **docs/COMPETITIVE_STUDY.md**, sections
4-10, with per-claim URLs and confidence flags.
