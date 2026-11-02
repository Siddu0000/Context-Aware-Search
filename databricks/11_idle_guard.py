"""Step 11: tear down the Vector Search index (and stop the App) after 3 idle days.

Why: the endpoint bills ~$7/day while it hosts an index, and a rebuild (02 resume())
costs ~$15-20 plus ~48 min. Keeping the index up for N idle days costs N x ~$7, so past
about 3 days a rebuild is cheaper (Sai, 2026-10-15).

"Used" = a queryIndex call on INDEX in system.access.audit (the App and every eval query
the index that way), or the index being (re)built. Idle = no use for IDLE_DAYS. Then it
runs 02 pause() (deletes ONLY the index: the endpoint stops billing 24h later, the stored
vectors stay, so resume() needs no re-embedding) and 09 app_stop().

Usage in a notebook (dry run first; it only prints what it would do):
    exec(open("databricks/11_idle_guard.py").read())
    check()                 # dry run
    check(dry_run=False)    # really tear down if idle
Schedule: Workflows -> Create job -> this notebook, task code `check(dry_run=False)`,
daily trigger, on the same cluster / serverless the runbook uses. The job's run-as
identity needs SELECT on system.access.audit and the rights 02/09 already need.
"""

import datetime as dt
import sys

sys.path.insert(0, "databricks")

IDLE_DAYS = 3


def _last_query(index_name, lookback_days=30):
    """Latest queryIndex audit event for the index, or None. Audit lags ~15 min-hours,
    which is irrelevant at a 3-day threshold."""
    rows = spark.sql(f"""
        SELECT max(event_time) AS last_use
        FROM system.access.audit
        WHERE service_name = 'vectorSearch'
          AND action_name IN ('queryIndex', 'queryVectorIndex')
          AND event_date >= date_sub(current_date(), {int(lookback_days)})
          AND (request_params['name'] = '{index_name}'
               OR request_params['index_name'] = '{index_name}')
    """).collect()
    return rows[0]["last_use"] if rows else None


def _index_created(index_name):
    """When the index was (re)built: a fresh resume() counts as use, else it would be
    deleted the morning after a rebuild that nobody has queried yet."""
    from app import vector_search

    try:
        info = vector_search.index_info(index_name)
    except Exception:  # noqa: BLE001 -- not found / no access: treat as unknown
        return None
    ts = (info or {}).get("creation_timestamp") or (info or {}).get("status", {}).get(
        "creation_timestamp")
    if ts is None:
        return None
    return dt.datetime.fromtimestamp(int(ts) / 1000, tz=dt.timezone.utc)


def check(dry_run=True, idle_days=IDLE_DAYS):
    import app.config as cfg

    exec(open("databricks/02_vector_index.py").read(), globals())  # pause(), _index_info()
    if _index_info() is None:
        print(f"{cfg.VS_INDEX} does not exist: nothing is billing for it. Done.")
        return "no-index"

    now = dt.datetime.now(dt.timezone.utc)
    try:
        last_q = _last_query(cfg.VS_INDEX)
    except Exception as e:  # noqa: BLE001
        # Without the audit table we cannot prove idleness: never delete blind
        print(f"Cannot read system.access.audit ({type(e).__name__}): keeping the index. "
              "Grant SELECT on it to the job's identity.")
        return "unknown"
    if last_q is not None and last_q.tzinfo is None:
        last_q = last_q.replace(tzinfo=dt.timezone.utc)
    created = _index_created(cfg.VS_INDEX)
    last_use = max([t for t in (last_q, created) if t is not None], default=None)

    if last_use is None:
        print("No query and no creation time found: keeping the index (cannot prove idle).")
        return "unknown"
    idle = now - last_use
    print(f"Last use of {cfg.VS_INDEX}: {last_use:%Y-%m-%d %H:%M} UTC "
          f"(query {last_q}, built {created}) -> idle {idle.days}d {idle.seconds // 3600}h.")
    if idle < dt.timedelta(days=idle_days):
        print(f"Under {idle_days} days: keeping it (~$7/day).")
        return "keep"

    if dry_run:
        print(f"DRY RUN: would run pause() (delete {cfg.VS_INDEX}) and stop App "
              f"{cfg.APP_NAME}. Re-run with check(dry_run=False).")
        return "would-delete"

    exec(open("databricks/09_deploy_app.py").read(), globals())  # app_stop()
    try:
        app_stop()  # an App without its index is useless but still bills
    except Exception as e:  # noqa: BLE001 -- no App deployed is fine
        print(f"App stop skipped ({type(e).__name__}).")
    pause()
    print("Rebuild before the next demo: 02 resume() (~48 min), then 09 app_grant() and "
          "app_start().")
    return "deleted"


if __name__ == "__main__":
    check()
