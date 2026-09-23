"""Step 3: end-to-end smoke test on Databricks -- retrieval + one LLM call.

The LLM call uses the model Cell 2 configures and the same JSON mode the app
sends, and its reply is ASSERTED, not just printed: a smoke test that passes on
an empty or non-JSON reply proves nothing. 08_llm_smoke.py checks every LLM
call site; this is the one-call version that also touches the index.
"""

import json
import os

from databricks.vector_search.client import VectorSearchClient
from openai import OpenAI  # FM APIs speak the OpenAI protocol

from app.llm_client import _message_text  # reasoning models reply in typed blocks

# The workspace catalog. `dev` is what we have write access to on the LatentView
# workspace; override with CAS_CATALOG / CAS_SCHEMA for any other workspace.
CATALOG = os.getenv("CAS_CATALOG", "dev")
SCHEMA = os.getenv("CAS_SCHEMA", "cas")
ENDPOINT = os.getenv("CAS_ENDPOINT", "cas-search")
INDEX = f"{CATALOG}.{SCHEMA}.products_index"
MODEL = os.environ.get("OPENAI_MODEL", "databricks-qwen3-next-80b-a3b-instruct")

checks = []  # (name, ok, detail)

vs = VectorSearchClient()
idx = vs.get_index(endpoint_name=ENDPOINT, index_name=INDEX)
# query_type="hybrid" (keyword+vector RRF) also exists — re-run router evals first
hits = idx.similarity_search(
    query_text="something breathable for a humid day",
    columns=["catalog_index", "Product_title", "categ_lvl2_name", "price"],
    num_results=5,
)
rows = hits.get("result", {}).get("data_array", [])
print(f"retrieval: {len(rows)} hits")
for r in rows:
    print("  -", str(r[1])[:60])
checks.append(("retrieval returns hits", bool(rows), f"{len(rows)} hits"))

llm = OpenAI(
    api_key=os.environ["DATABRICKS_TOKEN"],
    base_url=f"{os.environ['DATABRICKS_HOST']}/serving-endpoints",
)
resp = llm.chat.completions.create(
    model=MODEL,
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
print(f"llm ({MODEL}):", json.dumps(text)[:200])

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
    raise AssertionError(f"SMOKE TEST FAILED ({MODEL}): {', '.join(failed)}")
print(f"\nSMOKE TEST PASSED — index and {MODEL} live on Databricks.")
