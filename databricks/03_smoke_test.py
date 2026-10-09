"""Step 3: end-to-end smoke test on Databricks -- retrieval + one LLM call.

Retrieval goes through the APP's path (app.search -> app.vector_search: embed the
query with VS_EMBEDDING_MODEL, then query_vector), and a self-retrieval check
embeds three products' own search_text and expects each back at rank 1: that
fails if the stored catalog vectors and the query-time embeddings ever come
from different models or different text.

The LLM call uses the model Cell 2 configures and the same JSON mode the app
sends, and its reply is ASSERTED, not just printed: a smoke test that passes on
an empty or non-JSON reply proves nothing. 08_llm_smoke.py checks every LLM
call site; this is the one-call version that also touches the index.
"""

import json
import os

from openai import OpenAI  # FM APIs speak the OpenAI protocol

import app.config as cfg
from app import search, vector_search
from app.llm_client import _message_text  # reasoning models reply in typed blocks

# LLM_MODEL, not MODEL: every script is exec()ed into the SAME notebook globals
LLM_MODEL = os.environ.get("OPENAI_MODEL", "databricks-qwen3-next-80b-a3b-instruct")

checks = []  # (name, ok, detail)

cfg.SEARCH_BACKEND = "databricks"
vector_search.set_spark(spark)
print(f"index {cfg.VS_INDEX} | query mode {cfg.VS_QUERY_MODE} | embeddings {cfg.VS_EMBEDDING_MODEL}")
# query_type="hybrid" (keyword+vector RRF) also exists — re-run router evals first
hits = search.search_products(["breathable cotton top for a humid day"], top_k=5)
print(f"retrieval: {len(hits)} hits")
for h in hits:
    print(f"  - {h['score']:.3f}  {str(h.get('Product_title'))[:70]}")
checks.append(("retrieval returns hits", bool(hits), f"{len(hits)} hits"))

# Self-retrieval: a product's own text must find that product (or an identical-text twin)
cat = search.get_dataframe()
probe = cat.sample(3, random_state=7)
got = vector_search.query_many(probe["search_text"].tolist(), top_k=1)
hashes = cat["text_hash"].tolist()
ok = all(g and hashes[g[0][0]] == h for g, h in zip(got, probe["text_hash"]))
checks.append(("own text retrieves own product", ok,
               f"{[g[0][0] if g else None for g in got]} vs {probe.index.tolist()}"))

llm = OpenAI(
    api_key=os.environ["DATABRICKS_TOKEN"],
    base_url=f"{os.environ['DATABRICKS_HOST']}/serving-endpoints",
)
resp = llm.chat.completions.create(
    model=LLM_MODEL,
    messages=[{
        "role": "user",
        "content": (
            "Convert this shopper query into exactly 3 concise product search "
            "intents as JSON {\"search_terms\": [...]}: "
            "\"warm wool sweater for winter\""
        ),
    }],
    temperature=0.2,
    # The app sends this unless the model is known to reject it (llm_client._NO_JSON_MODE)
    response_format={"type": "json_object"},
)
choice = resp.choices[0]
text = _message_text(choice.message.content)
print(f"llm ({LLM_MODEL}):", json.dumps(text)[:200])

checks.append(("finish_reason == stop", choice.finish_reason == "stop",
               repr(choice.finish_reason)))
try:
    terms = json.loads(text).get("search_terms")
    ok = isinstance(terms, list) and bool(terms) and all(
        isinstance(t, str) and t.strip() for t in terms)
    checks.append(("reply is JSON with search_terms", ok, repr(terms)[:120]))
except (json.JSONDecodeError, AttributeError) as e:
    checks.append(("reply is JSON with search_terms", False, repr(e)[:120]))

print()
for name, ok, detail in checks:
    print(f"{'PASS' if ok else 'FAIL'}  {name:34s} {detail}")
failed = [name for name, ok, _ in checks if not ok]
if failed:
    # Raise, so a notebook cell (and a job) turns red instead of scrolling past
    raise AssertionError(f"SMOKE TEST FAILED ({LLM_MODEL}): {', '.join(failed)}")
print(f"\nSMOKE TEST PASSED — index and {LLM_MODEL} live on Databricks.")
