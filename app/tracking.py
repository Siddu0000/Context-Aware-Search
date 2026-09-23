"""Optional MLflow tracking: traces for app requests, one MLflow run per eval.

Every entry point is a no-op unless MLFLOW_ENABLED=true AND mlflow imports, and
every mlflow call is wrapped: tracking must never be able to break a search or
an eval. Local dev therefore needs no mlflow at all.

Traces come from mlflow.openai.autolog(), which patches the OpenAI SDK -- the
path used by Groq, api.openai.com and Databricks FM APIs alike. The Gemini and
Anthropic backends are not traced (they have their own autolog flavours; add
them if either becomes a production provider).
"""

import logging
import re
import warnings
from contextlib import ExitStack, contextmanager

import app.config as cfg

logger = logging.getLogger(__name__)

_autolog_on = False


def _mlflow():
    """The mlflow module when tracking is enabled and installed, else None."""
    if not cfg.MLFLOW_ENABLED:
        return None
    try:
        import mlflow
    except ImportError:
        logger.warning("MLFLOW_ENABLED=true but mlflow is not installed; tracking off.")
        return None
    return mlflow


def _set_experiment(mlflow) -> None:
    if cfg.MLFLOW_EXPERIMENT:
        mlflow.set_experiment(cfg.MLFLOW_EXPERIMENT)


def _safe_key(key: str) -> str:
    # MLflow keys allow only [/\w.\- ]; "P@10" would raise and drop the metric
    return re.sub(r"[^/\w.\- ]", "_", str(key).replace("@", "_at_"))


def enable_autolog() -> bool:
    """Trace every OpenAI-SDK call: prompt, reply, token usage, latency."""
    global _autolog_on
    mlflow = _mlflow()
    if mlflow is None or _autolog_on:
        return _autolog_on
    try:
        # Databricks returns reasoning replies as a LIST of content blocks where the
        # OpenAI SDK schema says str; pydantic then warns on every traced call. The
        # shape is expected and handled by llm_client._message_text -- silence only it.
        warnings.filterwarnings(
            "ignore", message=r"(?s).*Pydantic serializer warnings.*", category=UserWarning
        )
        _set_experiment(mlflow)
        mlflow.openai.autolog()
        _autolog_on = True
        logger.info("MLflow OpenAI autolog on (experiment=%s).", cfg.MLFLOW_EXPERIMENT)
    except Exception as e:  # noqa: BLE001 -- never let tracking take the service down
        logger.warning("MLflow autolog failed (%r); continuing untracked.", e)
    return _autolog_on


@contextmanager
def request_span(name: str, attributes: dict | None = None):
    """Parent span so one request's LLM calls land in ONE trace. Yields span or None."""
    mlflow = _mlflow() if _autolog_on else None
    with ExitStack() as stack:
        span = None
        if mlflow is not None:
            try:
                span = stack.enter_context(
                    mlflow.start_span(name=name, attributes=attributes or {})
                )
            except Exception as e:  # noqa: BLE001
                logger.debug("MLflow span start failed (%r).", e)
        yield span


def set_span_attributes(span, attributes: dict) -> None:
    if span is None:
        return
    try:
        span.set_attributes(attributes)
    except Exception as e:  # noqa: BLE001
        logger.debug("MLflow span attributes failed (%r).", e)


class _Run:
    """Handle yielded by eval_run(); logging methods swallow their own errors."""

    def __init__(self, mlflow=None, run_id=None):
        self._mlflow = mlflow
        self.run_id = run_id

    def log(self, params=None, metrics=None, artifact=None, tags=None) -> None:
        if self._mlflow is None:
            return
        mlflow = self._mlflow
        try:
            if params:
                mlflow.log_params({_safe_key(k): str(v)[:500] for k, v in params.items()})
            if metrics:
                clean = {}
                for k, v in metrics.items():
                    try:
                        clean[_safe_key(k)] = float(v)
                    except (TypeError, ValueError):
                        continue
                mlflow.log_metrics(clean)
            if tags:
                # `valid` lets the MLflow UI filter out runs where an LLM stage fell back
                mlflow.set_tags({_safe_key(k): str(v) for k, v in tags.items()})
            if artifact:
                mlflow.log_artifact(str(artifact))
        except Exception as e:  # noqa: BLE001 -- the CSV is still written either way
            logger.warning("MLflow eval logging failed (%r).", e)


@contextmanager
def eval_run(run_name: str):
    """One MLflow run around a whole eval, so its LLM traces link to the run."""
    mlflow = _mlflow()
    run = None
    if mlflow is not None:
        try:
            _set_experiment(mlflow)
            run = mlflow.start_run(run_name=run_name)
        except Exception as e:  # noqa: BLE001
            logger.warning("MLflow start_run failed (%r); eval runs untracked.", e)
            run = None
    if run is None:
        yield _Run()
        return
    ok = False
    try:
        yield _Run(mlflow, run.info.run_id)
        ok = True
    finally:
        try:
            mlflow.end_run(status="FINISHED" if ok else "FAILED")
        except Exception as e:  # noqa: BLE001
            logger.debug("MLflow end_run failed (%r).", e)
