import io
import json
import logging
import os
import re
import base64
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import requests
from dotenv import load_dotenv
from flask import Flask, Response, g, jsonify, render_template_string, request, send_file, url_for
from flask_cors import CORS
from requests import Response as RequestsResponse
from analytics_runtime import AnalyticsRuntime, create_job_handlers

try:
    import redis
except ImportError:
    redis = None

try:
    from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

    PROMETHEUS_ENABLED = True
except ImportError:
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"
    PROMETHEUS_ENABLED = False

    class _NoOpMetric:
        def labels(self, **_: Any) -> "_NoOpMetric":
            return self

        def inc(self, amount: int = 1) -> None:
            return None

        def observe(self, value: float) -> None:
            return None

        def set(self, value: float) -> None:
            return None

    def Counter(*_: Any, **__: Any) -> _NoOpMetric:
        return _NoOpMetric()

    def Gauge(*_: Any, **__: Any) -> _NoOpMetric:
        return _NoOpMetric()

    def Histogram(*_: Any, **__: Any) -> _NoOpMetric:
        return _NoOpMetric()

    def generate_latest() -> bytes:
        return b'# metrics_unavailable{reason="prometheus_client_missing"} 1\n'


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


@dataclass(frozen=True)
class AppConfig:
    host: str
    port: int
    debug: bool
    request_timeout_seconds: int
    max_comments_per_request: int
    max_comment_length: int
    log_level: str
    youtube_api_key: str | None
    model_path: Path
    vectorizer_path: Path
    async_worker_count: int
    job_ttl_seconds: int
    redis_url: str | None
    job_queue_name: str
    dead_letter_queue_name: str
    max_job_attempts: int

    @classmethod
    def from_env(cls) -> "AppConfig":
        return cls(
            host=os.getenv("HOST", "0.0.0.0"),
            port=int(os.getenv("PORT", "5000")),
            debug=os.getenv("DEBUG", "false").lower() == "true",
            request_timeout_seconds=int(os.getenv("REQUEST_TIMEOUT_SECONDS", "30")),
            max_comments_per_request=int(os.getenv("MAX_COMMENTS_PER_REQUEST", "200")),
            max_comment_length=int(os.getenv("MAX_COMMENT_LENGTH", "1000")),
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
            youtube_api_key=os.getenv("YOUTUBE_API_KEY"),
            model_path=BASE_DIR / "lgbm_model.pkl",
            vectorizer_path=BASE_DIR / "tfidf_vectorizer.pkl",
            async_worker_count=int(os.getenv("ASYNC_WORKER_COUNT", "4")),
            job_ttl_seconds=int(os.getenv("JOB_TTL_SECONDS", "3600")),
            redis_url=os.getenv("REDIS_URL"),
            job_queue_name=os.getenv("JOB_QUEUE_NAME", "analytics_jobs"),
            dead_letter_queue_name=os.getenv("DEAD_LETTER_QUEUE_NAME", "analytics_jobs_dead_letter"),
            max_job_attempts=int(os.getenv("MAX_JOB_ATTEMPTS", "3")),
        )


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for field_name in ("request_id", "path", "status_code", "duration_ms", "job_id", "job_type"):
            value = getattr(record, field_name, None)
            if value is not None:
                payload[field_name] = value
        return json.dumps(payload)


def configure_logging(level: str) -> logging.Logger:
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    if not root_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        root_logger.addHandler(handler)
    else:
        for handler in root_logger.handlers:
            handler.setFormatter(JsonFormatter())

    return logging.getLogger("youtube_sentiment_api")


REQUEST_COUNT = Counter(
    "youtube_sentiment_http_requests_total",
    "Total HTTP requests.",
    ["method", "path", "status_code"],
)
REQUEST_LATENCY = Histogram(
    "youtube_sentiment_http_request_duration_seconds",
    "HTTP request latency.",
    ["method", "path"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)
INFERENCE_COMMENT_COUNT = Histogram(
    "youtube_sentiment_inference_comment_count",
    "Number of comments sent to inference endpoints.",
    buckets=(1, 5, 10, 25, 50, 100, 200, 500),
)
EXTERNAL_CALL_COUNT = Counter(
    "youtube_sentiment_external_requests_total",
    "External dependency calls.",
    ["provider", "outcome"],
)
MODEL_READY = Gauge(
    "youtube_sentiment_model_ready",
    "Whether the model and vectorizer are loaded successfully.",
)
ASYNC_JOBS_IN_PROGRESS = Gauge(
    "youtube_sentiment_async_jobs_in_progress",
    "Number of jobs currently running in the async executor.",
)
ASYNC_JOB_COUNT = Counter(
    "youtube_sentiment_async_jobs_total",
    "Async job submissions and outcomes.",
    ["job_type", "status"],
)
ASYNC_JOB_DURATION = Histogram(
    "youtube_sentiment_async_job_duration_seconds",
    "Async job duration.",
    ["job_type"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
)
QUEUE_DEPTH = Gauge(
    "youtube_sentiment_queue_depth",
    "Current Redis queue depth by queue type.",
    ["queue_type"],
)


class ValidationError(Exception):
    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


class JobNotReadyError(Exception):
    pass


def utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass
class JobRecord:
    job_id: str
    job_type: str
    status: str
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None = None
    result: Any = None
    error: str | None = None
    content_type: str | None = None
    artifact_filename: str | None = None
    attempts: int = 0
    max_attempts: int = 1
    dead_lettered: bool = False

    def to_response(self) -> dict[str, Any]:
        payload = {
            "job_id": self.job_id,
            "job_type": self.job_type,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error": self.error,
            "attempts": self.attempts,
            "max_attempts": self.max_attempts,
            "dead_lettered": self.dead_lettered,
        }
        if self.status == "completed" and isinstance(self.result, dict):
            payload["result"] = self.result
        if self.status == "completed" and isinstance(self.result, bytes):
            payload["artifact_ready"] = True
        return payload


class LocalJobManager:
    def __init__(
        self,
        max_workers: int,
        ttl_seconds: int,
        logger: logging.Logger,
        handlers: dict[str, Callable[[dict[str, Any]], tuple[Any, str | None, str | None]]],
        max_attempts: int,
    ) -> None:
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="analytics")
        self.ttl_seconds = ttl_seconds
        self.logger = logger
        self.handlers = handlers
        self.max_attempts = max_attempts
        self._jobs: dict[str, JobRecord] = {}
        self._lock = threading.Lock()

    def submit(self, job_type: str, payload: dict[str, Any]) -> JobRecord:
        self._cleanup_expired_jobs()
        now = utc_now()
        job = JobRecord(
            job_id=str(uuid.uuid4()),
            job_type=job_type,
            status="queued",
            created_at=now,
            updated_at=now,
            max_attempts=self.max_attempts,
        )
        with self._lock:
            self._jobs[job.job_id] = job
        ASYNC_JOB_COUNT.labels(job_type=job_type, status="queued").inc()
        self.executor.submit(self._run_job, job.job_id, payload)
        return job

    def _run_job(self, job_id: str, payload: dict[str, Any]) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.status = "running"
            job.updated_at = utc_now()

        started_at = time.perf_counter()
        ASYNC_JOBS_IN_PROGRESS.inc()
        ASYNC_JOB_COUNT.labels(job_type=job.job_type, status="running").inc()

        try:
            handler = self.handlers[job.job_type]
            with self._lock:
                job.attempts += 1
            result, content_type, artifact_filename = handler(payload)
            duration = time.perf_counter() - started_at
            ASYNC_JOB_DURATION.labels(job_type=job.job_type).observe(duration)
            with self._lock:
                job.status = "completed"
                job.updated_at = utc_now()
                job.completed_at = job.updated_at
                job.result = result
                job.content_type = content_type
                job.artifact_filename = artifact_filename
            ASYNC_JOB_COUNT.labels(job_type=job.job_type, status="completed").inc()
        except Exception as exc:
            should_retry = False
            with self._lock:
                job.error = str(exc)
                job.updated_at = utc_now()
                if job.attempts < job.max_attempts:
                    job.status = "queued"
                    should_retry = True
                else:
                    job.status = "failed"
                    job.completed_at = job.updated_at
                    job.dead_lettered = True
            if should_retry:
                ASYNC_JOB_COUNT.labels(job_type=job.job_type, status="retry").inc()
                self.executor.submit(self._run_job, job.job_id, payload)
            else:
                ASYNC_JOB_COUNT.labels(job_type=job.job_type, status="failed").inc()
                self.logger.exception(
                    "async_job_failed",
                    extra={"job_id": job.job_id, "job_type": job.job_type},
                )
        finally:
            ASYNC_JOBS_IN_PROGRESS.inc(-1)

    def get(self, job_id: str) -> JobRecord | None:
        self._cleanup_expired_jobs()
        with self._lock:
            return self._jobs.get(job_id)

    def get_artifact(self, job_id: str) -> tuple[bytes, str, str | None]:
        job = self.get(job_id)
        if job is None:
            raise ValidationError("Job not found.", 404)
        if job.status != "completed":
            raise JobNotReadyError()
        if not isinstance(job.result, bytes) or not job.content_type:
            raise ValidationError("Job does not produce a downloadable artifact.", 400)
        return job.result, job.content_type, job.artifact_filename

    def _cleanup_expired_jobs(self) -> None:
        cutoff = utc_now().timestamp() - self.ttl_seconds
        with self._lock:
            expired_ids = [
                job_id
                for job_id, job in self._jobs.items()
                if job.updated_at.timestamp() < cutoff
            ]
            for job_id in expired_ids:
                del self._jobs[job_id]


class RedisJobManager:
    def __init__(
        self,
        redis_url: str,
        queue_name: str,
        dead_letter_queue_name: str,
        ttl_seconds: int,
        logger: logging.Logger,
        max_attempts: int,
    ) -> None:
        if redis is None:
            raise RuntimeError("redis package is not available.")
        self.client = redis.Redis.from_url(redis_url, decode_responses=False)
        self.queue_name = queue_name
        self.dead_letter_queue_name = dead_letter_queue_name
        self.ttl_seconds = ttl_seconds
        self.logger = logger
        self.max_attempts = max_attempts

    def submit(self, job_type: str, payload: dict[str, Any]) -> JobRecord:
        now = utc_now()
        job = JobRecord(
            job_id=str(uuid.uuid4()),
            job_type=job_type,
            status="queued",
            created_at=now,
            updated_at=now,
            max_attempts=self.max_attempts,
        )
        self._write_job(job)
        message = json.dumps({"job_id": job.job_id, "job_type": job_type, "payload": payload})
        self.client.rpush(self.queue_name, message)
        self.update_queue_metrics()
        ASYNC_JOB_COUNT.labels(job_type=job_type, status="queued").inc()
        return job

    def get(self, job_id: str) -> JobRecord | None:
        data = self.client.get(self._job_key(job_id))
        self.update_queue_metrics()
        if not data:
            return None
        payload = json.loads(data.decode("utf-8"))
        result = payload.get("result")
        if payload.get("result_is_bytes") and isinstance(result, str):
            result = base64.b64decode(result.encode("utf-8"))
        return JobRecord(
            job_id=payload["job_id"],
            job_type=payload["job_type"],
            status=payload["status"],
            created_at=datetime.fromisoformat(payload["created_at"]),
            updated_at=datetime.fromisoformat(payload["updated_at"]),
            completed_at=datetime.fromisoformat(payload["completed_at"]) if payload.get("completed_at") else None,
            result=result,
            error=payload.get("error"),
            content_type=payload.get("content_type"),
            artifact_filename=payload.get("artifact_filename"),
            attempts=payload.get("attempts", 0),
            max_attempts=payload.get("max_attempts", self.max_attempts),
            dead_lettered=payload.get("dead_lettered", False),
        )

    def get_artifact(self, job_id: str) -> tuple[bytes, str, str | None]:
        job = self.get(job_id)
        if job is None:
            raise ValidationError("Job not found.", 404)
        if job.status != "completed":
            raise JobNotReadyError()
        if not isinstance(job.result, bytes) or not job.content_type:
            raise ValidationError("Job does not produce a downloadable artifact.", 400)
        return job.result, job.content_type, job.artifact_filename

    def get_dead_letter_jobs(self, limit: int = 20) -> list[dict[str, Any]]:
        self.update_queue_metrics()
        items = self.client.lrange(self.dead_letter_queue_name, 0, max(0, limit - 1))
        jobs = []
        for raw in items:
            payload = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
            job = self.get(payload["job_id"])
            jobs.append(
                {
                    "job_id": payload["job_id"],
                    "job_type": payload["job_type"],
                    "payload": payload["payload"],
                    "status": job.status if job else "missing",
                    "attempts": job.attempts if job else None,
                    "max_attempts": job.max_attempts if job else None,
                    "error": job.error if job else None,
                    "dead_lettered": job.dead_lettered if job else True,
                }
            )
        return jobs

    def replay_dead_letter_job(self, job_id: str) -> bool:
        items = self.client.lrange(self.dead_letter_queue_name, 0, -1)
        for index, raw in enumerate(items):
            payload = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
            if payload["job_id"] != job_id:
                continue

            job = self.get(job_id)
            if job is None:
                return False

            job.status = "queued"
            job.updated_at = utc_now()
            job.completed_at = None
            job.error = None
            job.dead_lettered = False
            job.attempts = 0
            self._write_job(job)

            pipeline = self.client.pipeline()
            pipeline.lrem(self.dead_letter_queue_name, 1, raw)
            pipeline.rpush(self.queue_name, json.dumps(payload))
            pipeline.execute()
            self.update_queue_metrics()
            return True
        return False

    def update_queue_metrics(self) -> None:
        try:
            QUEUE_DEPTH.labels(queue_type="primary").set(self.client.llen(self.queue_name))
            QUEUE_DEPTH.labels(queue_type="dead_letter").set(self.client.llen(self.dead_letter_queue_name))
        except Exception:
            return None

    def _write_job(self, job: JobRecord) -> None:
        result = job.result
        result_is_bytes = isinstance(result, bytes)
        if result_is_bytes:
            result = base64.b64encode(result).decode("utf-8")
        payload = {
            "job_id": job.job_id,
            "job_type": job.job_type,
            "status": job.status,
            "created_at": job.created_at.isoformat(),
            "updated_at": job.updated_at.isoformat(),
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "result": result,
            "result_is_bytes": result_is_bytes,
            "error": job.error,
            "content_type": job.content_type,
            "artifact_filename": job.artifact_filename,
            "attempts": job.attempts,
            "max_attempts": job.max_attempts,
            "dead_lettered": job.dead_lettered,
        }
        self.client.setex(self._job_key(job.job_id), self.ttl_seconds, json.dumps(payload))

    def _job_key(self, job_id: str) -> str:
        return f"job:{job_id}"


def error_response(message: str, status_code: int) -> tuple[Response, int]:
    request_id = getattr(g, "request_id", None)
    return (
        jsonify(
            {
                "error": {
                    "message": message,
                    "status_code": status_code,
                    "request_id": request_id,
                }
            }
        ),
        status_code,
    )


def parse_json_body() -> dict[str, Any]:
    if not request.is_json:
        raise ValidationError("Request must have application/json content type.", 415)
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        raise ValidationError("Request body must be a JSON object.")
    return payload


def validate_comments(
    comments: Any,
    config: AppConfig,
    *,
    require_timestamps: bool = False,
) -> list[Any]:
    if not isinstance(comments, list) or not comments:
        raise ValidationError("comments must be a non-empty list.")
    if len(comments) > config.max_comments_per_request:
        raise ValidationError(
            f"comments exceeds MAX_COMMENTS_PER_REQUEST={config.max_comments_per_request}.",
            413,
        )

    validated: list[Any] = []
    for item in comments:
        if require_timestamps:
            if not isinstance(item, dict):
                raise ValidationError("Each comment must be an object with text and timestamp.")
            text = item.get("text")
            timestamp = item.get("timestamp")
            if not isinstance(text, str) or not text.strip():
                raise ValidationError("Each comment object must include non-empty text.")
            if len(text) > config.max_comment_length:
                raise ValidationError(
                    f"Comment length exceeds MAX_COMMENT_LENGTH={config.max_comment_length}."
                )
            if not isinstance(timestamp, str) or not timestamp.strip():
                raise ValidationError("Each comment object must include a non-empty timestamp.")
            validated.append({"text": text, "timestamp": timestamp})
        else:
            if not isinstance(item, str) or not item.strip():
                raise ValidationError("Each comment must be a non-empty string.")
            if len(item) > config.max_comment_length:
                raise ValidationError(
                    f"Comment length exceeds MAX_COMMENT_LENGTH={config.max_comment_length}."
                )
            validated.append(item)
    return validated


def validate_sentiment_counts(data: Any) -> dict[str, int]:
    if not isinstance(data, dict):
        raise ValidationError("sentiment_counts must be an object.")
    counts: dict[str, int] = {}
    for key in ("1", "0", "-1"):
        value = data.get(key, 0)
        if not isinstance(value, int) or value < 0:
            raise ValidationError("sentiment_counts values must be non-negative integers.")
        counts[key] = value
    return counts


def validate_sentiment_data(
    sentiment_data: Any,
    config: AppConfig,
) -> list[dict[str, Any]]:
    if not isinstance(sentiment_data, list) or not sentiment_data:
        raise ValidationError("sentiment_data must be a non-empty list.")
    if len(sentiment_data) > config.max_comments_per_request:
        raise ValidationError(
            f"sentiment_data exceeds MAX_COMMENTS_PER_REQUEST={config.max_comments_per_request}.",
            413,
        )

    validated: list[dict[str, Any]] = []
    for item in sentiment_data:
        if not isinstance(item, dict):
            raise ValidationError("Each sentiment_data entry must be an object.")
        timestamp = item.get("timestamp")
        sentiment = item.get("sentiment")
        if not isinstance(timestamp, str) or not timestamp.strip():
            raise ValidationError("Each sentiment_data entry needs a non-empty timestamp.")
        if not isinstance(sentiment, int) or sentiment not in {-1, 0, 1}:
            raise ValidationError("Each sentiment must be one of -1, 0, or 1.")
        validated.append({"timestamp": timestamp, "sentiment": sentiment})
    return validated


def timed_external_request(
    session: requests.Session,
    method: str,
    url: str,
    *,
    provider: str,
    timeout: int,
    **kwargs: Any,
) -> RequestsResponse:
    try:
        response = session.request(method=method, url=url, timeout=timeout, **kwargs)
        response.raise_for_status()
        EXTERNAL_CALL_COUNT.labels(provider=provider, outcome="success").inc()
        return response
    except requests.RequestException:
        EXTERNAL_CALL_COUNT.labels(provider=provider, outcome="failure").inc()
        raise


def create_app() -> Flask:
    config = AppConfig.from_env()
    logger = configure_logging(config.log_level)

    app = Flask(__name__)
    app.config["APP_CONFIG"] = config
    CORS(app)
    try:
        runtime = AnalyticsRuntime(config.model_path, config.vectorizer_path, logger)
        MODEL_READY.set(1)
        logger.info("Sentiment model artifacts loaded successfully.")
    except Exception as exc:
        MODEL_READY.set(0)
        logger.exception("Failed to load model artifacts: %s", exc)
        runtime = None

    handlers = create_job_handlers(runtime) if runtime is not None else {}
    redis_healthy = False
    if config.redis_url and redis is not None:
        try:
            jobs = RedisJobManager(
                config.redis_url,
                config.job_queue_name,
                config.dead_letter_queue_name,
                config.job_ttl_seconds,
                logger,
                config.max_job_attempts,
            )
            jobs.client.ping()
            redis_healthy = True
            job_backend = "redis"
        except Exception as exc:
            logger.warning("Redis job backend unavailable, falling back to local jobs: %s", exc)
            jobs = LocalJobManager(
                config.async_worker_count,
                config.job_ttl_seconds,
                logger,
                handlers,
                config.max_job_attempts,
            )
            job_backend = "local"
    else:
        jobs = LocalJobManager(
            config.async_worker_count,
            config.job_ttl_seconds,
            logger,
            handlers,
            config.max_job_attempts,
        )
        job_backend = "local"

    def ensure_runtime() -> AnalyticsRuntime:
        if runtime is None:
            raise ValidationError("Model artifacts are not loaded.", 503)
        return runtime

    def ensure_redis_backend() -> RedisJobManager:
        if not isinstance(jobs, RedisJobManager):
            raise ValidationError("Redis job backend is not active.", 409)
        jobs.update_queue_metrics()
        return jobs

    def submit_json_job(job_type: str, payload: dict[str, Any]) -> Response:
        job = jobs.submit(job_type, payload)
        response = job.to_response()
        response["status_url"] = url_for("get_job", job_id=job.job_id, _external=False)
        response["artifact_url"] = None
        return jsonify(response), 202

    def submit_artifact_job(job_type: str, payload: dict[str, Any]) -> Response:
        job = jobs.submit(job_type, payload)
        response = job.to_response()
        response["status_url"] = url_for("get_job", job_id=job.job_id, _external=False)
        response["artifact_url"] = url_for("get_job_artifact", job_id=job.job_id, _external=False)
        return jsonify(response), 202

    @app.before_request
    def before_request() -> None:
        g.request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        g.request_started_at = time.perf_counter()

    @app.after_request
    def after_request(response: Response) -> Response:
        duration_seconds = time.perf_counter() - g.request_started_at
        duration_ms = round(duration_seconds * 1000, 2)
        response.headers["X-Request-ID"] = g.request_id
        REQUEST_COUNT.labels(
            method=request.method,
            path=request.path,
            status_code=str(response.status_code),
        ).inc()
        REQUEST_LATENCY.labels(method=request.method, path=request.path).observe(duration_seconds)
        logger.info(
            "request_complete",
            extra={
                "request_id": g.request_id,
                "path": request.path,
                "status_code": response.status_code,
                "duration_ms": duration_ms,
            },
        )
        return response

    @app.errorhandler(ValidationError)
    def handle_validation_error(error: ValidationError) -> tuple[Response, int]:
        return error_response(str(error), error.status_code)

    @app.errorhandler(JobNotReadyError)
    def handle_job_not_ready(_: JobNotReadyError) -> tuple[Response, int]:
        return error_response("Job is not finished yet.", 409)

    @app.errorhandler(Exception)
    def handle_unexpected_error(_: Exception) -> tuple[Response, int]:
        logger.exception(
            "unhandled_exception",
            extra={"request_id": getattr(g, "request_id", None), "path": request.path},
        )
        return error_response("Internal server error.", 500)

    @app.route("/")
    def home() -> Response:
        return Response(
            render_template_string(
                """
                <!doctype html>
                <html lang="en">
                  <head>
                    <meta charset="utf-8" />
                    <meta name="viewport" content="width=device-width, initial-scale=1" />
                    <title>CommentPulse</title>
                    <style>
                      :root {
                        color-scheme: dark;
                        --bg: #07111f;
                        --bg-soft: #0f1b2d;
                        --panel: rgba(15, 23, 42, 0.82);
                        --panel-strong: rgba(10, 16, 29, 0.92);
                        --panel-border: rgba(148, 163, 184, 0.14);
                        --text: #e2e8f0;
                        --muted: #8aa0bb;
                        --accent: #38bdf8;
                        --accent-strong: #0ea5e9;
                        --positive: #22c55e;
                        --neutral: #f59e0b;
                        --negative: #ef4444;
                        --shadow: 0 28px 50px rgba(2, 6, 23, 0.34);
                      }

                      * { box-sizing: border-box; }

                      body {
                        margin: 0;
                        font-family: "Segoe UI", Arial, sans-serif;
                        background:
                          radial-gradient(circle at top left, rgba(14, 165, 233, 0.18), transparent 28%),
                          radial-gradient(circle at 85% 15%, rgba(34, 197, 94, 0.12), transparent 18%),
                          linear-gradient(180deg, #07111f 0%, #0b1220 100%);
                        color: var(--text);
                        min-height: 100vh;
                        padding: 28px 16px 40px;
                      }

                      .shell {
                        width: min(1180px, 100%);
                        margin: 0 auto;
                        display: grid;
                        gap: 18px;
                      }

                      .hero, .panel {
                        background: var(--panel);
                        border: 1px solid var(--panel-border);
                        border-radius: 24px;
                        box-shadow: var(--shadow);
                        backdrop-filter: blur(12px);
                      }

                      .hero {
                        padding: 30px;
                        overflow: hidden;
                        position: relative;
                      }

                      .hero::after {
                        content: "";
                        position: absolute;
                        inset: auto -20% -40% 38%;
                        height: 320px;
                        background: radial-gradient(circle, rgba(56, 189, 248, 0.16), transparent 60%);
                        pointer-events: none;
                      }

                      .badge {
                        display: inline-block;
                        font-size: 11px;
                        letter-spacing: 0.14em;
                        text-transform: uppercase;
                        font-weight: 700;
                        color: var(--accent);
                        background: rgba(56, 189, 248, 0.12);
                        border: 1px solid rgba(56, 189, 248, 0.24);
                        border-radius: 999px;
                        padding: 7px 12px;
                        margin-bottom: 16px;
                      }

                      .hero-grid {
                        display: grid;
                        grid-template-columns: minmax(0, 1.2fr) minmax(320px, 0.8fr);
                        gap: 22px;
                        align-items: stretch;
                      }

                      h1 {
                        margin: 0 0 10px;
                        font-size: clamp(34px, 5vw, 58px);
                        line-height: 1.02;
                      }

                      .subtitle {
                        margin: 0;
                        color: var(--muted);
                        font-size: 17px;
                        line-height: 1.6;
                        max-width: 700px;
                      }

                      .cta-row, .meta-grid, .link-grid, .stats-grid, .detail-grid {
                        display: grid;
                        gap: 14px;
                      }

                      .cta-row {
                        grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
                        margin-top: 24px;
                      }

                      .link-grid {
                        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                      }

                      .stats-grid {
                        grid-template-columns: repeat(3, minmax(0, 1fr));
                        margin-top: 22px;
                      }

                      .detail-grid {
                        grid-template-columns: minmax(0, 1.2fr) minmax(300px, 0.8fr);
                      }

                      .panel {
                        padding: 22px;
                      }

                      .panel h2 {
                        margin: 0 0 16px;
                        font-size: 18px;
                      }

                      .eyebrow {
                        color: var(--accent);
                        font-size: 11px;
                        text-transform: uppercase;
                        letter-spacing: 0.12em;
                        font-weight: 700;
                        margin-bottom: 10px;
                      }

                      .meta-grid {
                        grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
                      }

                      .metric, .stat-card {
                        background: rgba(15, 23, 42, 0.92);
                        border: 1px solid rgba(148, 163, 184, 0.1);
                        border-radius: 16px;
                        padding: 16px;
                      }

                      .metric-label {
                        color: var(--muted);
                        font-size: 12px;
                        text-transform: uppercase;
                        letter-spacing: 0.08em;
                        margin-bottom: 8px;
                      }

                      .metric-value {
                        font-size: 20px;
                        font-weight: 700;
                      }

                      .stat-card {
                        min-height: 118px;
                        display: grid;
                        align-content: center;
                        gap: 6px;
                      }

                      .stat-card.positive { border-color: rgba(34, 197, 94, 0.26); }
                      .stat-card.neutral { border-color: rgba(245, 158, 11, 0.24); }
                      .stat-card.negative { border-color: rgba(239, 68, 68, 0.24); }

                      .stat-label {
                        color: var(--muted);
                        font-size: 12px;
                        letter-spacing: 0.08em;
                        text-transform: uppercase;
                      }

                      .stat-value {
                        font-size: 38px;
                        font-weight: 800;
                        line-height: 1;
                      }

                      .stat-value.positive { color: var(--positive); }
                      .stat-value.neutral { color: var(--neutral); }
                      .stat-value.negative { color: var(--negative); }

                      .demo-shell {
                        background: linear-gradient(180deg, rgba(2, 6, 23, 0.9), rgba(15, 23, 42, 0.92));
                        border: 1px solid rgba(148, 163, 184, 0.14);
                        border-radius: 22px;
                        padding: 18px;
                        position: relative;
                        z-index: 1;
                      }

                      .demo-shell h2 {
                        margin: 0 0 8px;
                        font-size: 20px;
                      }

                      .demo-copy {
                        color: var(--muted);
                        font-size: 14px;
                        line-height: 1.6;
                        margin: 0 0 14px;
                      }

                      textarea {
                        width: 100%;
                        min-height: 200px;
                        resize: vertical;
                        border: 1px solid rgba(148, 163, 184, 0.14);
                        border-radius: 16px;
                        background: rgba(7, 18, 33, 0.95);
                        color: var(--text);
                        padding: 14px 16px;
                        font: inherit;
                        line-height: 1.55;
                      }

                      textarea:focus, button:focus, a:focus {
                        outline: 2px solid rgba(56, 189, 248, 0.48);
                        outline-offset: 2px;
                      }

                      .demo-actions {
                        display: flex;
                        flex-wrap: wrap;
                        gap: 12px;
                        margin-top: 14px;
                        align-items: center;
                      }

                      button, .button-link {
                        appearance: none;
                        border: 0;
                        border-radius: 14px;
                        padding: 12px 16px;
                        font: inherit;
                        font-weight: 700;
                        cursor: pointer;
                        transition: transform 0.16s ease, opacity 0.16s ease, background 0.16s ease;
                      }

                      button:hover, .button-link:hover {
                        transform: translateY(-1px);
                      }

                      .primary-button {
                        background: linear-gradient(135deg, var(--accent-strong), var(--accent));
                        color: #04111f;
                      }

                      .secondary-button {
                        background: rgba(56, 189, 248, 0.12);
                        color: var(--text);
                        border: 1px solid rgba(56, 189, 248, 0.18);
                      }

                      .status-pill {
                        display: inline-flex;
                        align-items: center;
                        gap: 8px;
                        padding: 10px 12px;
                        border-radius: 999px;
                        font-size: 12px;
                        color: var(--muted);
                        background: rgba(15, 23, 42, 0.72);
                        border: 1px solid rgba(148, 163, 184, 0.12);
                      }

                      .status-dot {
                        width: 8px;
                        height: 8px;
                        border-radius: 50%;
                        background: var(--neutral);
                        box-shadow: 0 0 0 6px rgba(245, 158, 11, 0.08);
                      }

                      .status-dot.ready { background: var(--positive); box-shadow: 0 0 0 6px rgba(34, 197, 94, 0.08); }
                      .status-dot.error { background: var(--negative); box-shadow: 0 0 0 6px rgba(239, 68, 68, 0.08); }

                      a.card-link {
                        text-decoration: none;
                        color: var(--text);
                        display: block;
                        background: rgba(15, 23, 42, 0.92);
                        border: 1px solid rgba(148, 163, 184, 0.1);
                        border-radius: 16px;
                        padding: 16px;
                        transition: transform 0.16s ease, border-color 0.16s ease;
                      }

                      a.card-link:hover {
                        transform: translateY(-2px);
                        border-color: rgba(56, 189, 248, 0.35);
                      }

                      a.card-link {
                        min-height: 112px;
                      }

                      .link-title {
                        font-size: 15px;
                        font-weight: 700;
                        margin-bottom: 6px;
                      }

                      .link-body {
                        color: var(--muted);
                        font-size: 13px;
                        line-height: 1.5;
                      }

                      .predictions {
                        display: grid;
                        gap: 10px;
                        margin-top: 16px;
                        max-height: 320px;
                        overflow: auto;
                        padding-right: 4px;
                      }

                      .prediction-item {
                        display: grid;
                        gap: 8px;
                        background: rgba(8, 15, 28, 0.92);
                        border: 1px solid rgba(148, 163, 184, 0.1);
                        border-radius: 16px;
                        padding: 14px;
                      }

                      .prediction-meta {
                        display: flex;
                        justify-content: space-between;
                        gap: 10px;
                        align-items: center;
                      }

                      .sentiment-chip {
                        display: inline-flex;
                        align-items: center;
                        gap: 8px;
                        border-radius: 999px;
                        padding: 7px 10px;
                        font-size: 11px;
                        letter-spacing: 0.08em;
                        text-transform: uppercase;
                        font-weight: 700;
                      }

                      .sentiment-chip.positive {
                        background: rgba(34, 197, 94, 0.14);
                        color: #b8f7cb;
                        border: 1px solid rgba(34, 197, 94, 0.22);
                      }

                      .sentiment-chip.neutral {
                        background: rgba(245, 158, 11, 0.14);
                        color: #fde3a7;
                        border: 1px solid rgba(245, 158, 11, 0.2);
                      }

                      .sentiment-chip.negative {
                        background: rgba(239, 68, 68, 0.14);
                        color: #fecaca;
                        border: 1px solid rgba(239, 68, 68, 0.2);
                      }

                      .prediction-time {
                        color: var(--muted);
                        font-size: 12px;
                      }

                      .prediction-text {
                        margin: 0;
                        font-size: 14px;
                        line-height: 1.55;
                        color: var(--text);
                      }

                      .stack {
                        display: grid;
                        gap: 18px;
                      }

                      .architecture-list {
                        display: grid;
                        gap: 12px;
                      }

                      .architecture-step {
                        display: grid;
                        grid-template-columns: 34px 1fr;
                        gap: 12px;
                        align-items: start;
                      }

                      .architecture-step-index {
                        width: 34px;
                        height: 34px;
                        border-radius: 12px;
                        background: rgba(56, 189, 248, 0.12);
                        border: 1px solid rgba(56, 189, 248, 0.2);
                        display: grid;
                        place-items: center;
                        color: var(--accent);
                        font-size: 13px;
                        font-weight: 800;
                      }

                      .architecture-step h3 {
                        margin: 0 0 4px;
                        font-size: 15px;
                      }

                      .architecture-step p {
                        margin: 0;
                        color: var(--muted);
                        font-size: 13px;
                        line-height: 1.55;
                      }

                      .empty-state {
                        color: var(--muted);
                        font-size: 14px;
                        line-height: 1.6;
                        margin: 0;
                      }

                      .footnote {
                        color: var(--muted);
                        font-size: 12px;
                        line-height: 1.6;
                        margin-top: 14px;
                      }

                      @media (max-width: 920px) {
                        .hero-grid,
                        .detail-grid {
                          grid-template-columns: 1fr;
                        }

                        .stats-grid {
                          grid-template-columns: 1fr;
                        }
                      }

                      @media (max-width: 640px) {
                        body {
                          padding: 18px 12px 28px;
                        }

                        .hero, .panel {
                          border-radius: 20px;
                        }

                        .hero, .panel {
                          padding: 18px;
                        }
                      }
                    </style>
                  </head>
                  <body>
                    <main class="shell">
                      <section class="hero">
                        <div class="hero-grid">
                          <div>
                            <div class="badge">CommentPulse</div>
                            <h1>Live Sentiment Inference for YouTube-Style Comments</h1>
                            <p class="subtitle">
                              A production-style ML application with model-backed sentiment prediction, async analytics,
                              Redis worker support, observability, and a Chrome extension workflow built around real
                              comment analysis.
                            </p>
                            <div class="stats-grid">
                              <div class="stat-card positive">
                                <div class="stat-label">Throughput</div>
                                <div class="stat-value positive">77.4</div>
                                <div class="link-body">req/s after hot-path optimization in local load testing</div>
                              </div>
                              <div class="stat-card neutral">
                                <div class="stat-label">Latency</div>
                                <div class="stat-value neutral">249ms</div>
                                <div class="link-body">p95 latency on the optimized local benchmark</div>
                              </div>
                              <div class="stat-card negative">
                                <div class="stat-label">Reliability</div>
                                <div class="stat-value negative">0%</div>
                                <div class="link-body">request failures in the verified optimized test run</div>
                              </div>
                            </div>
                          </div>

                          <div class="demo-shell">
                            <div class="eyebrow">Live Model Demo</div>
                            <h2>Try the deployed classifier</h2>
                            <p class="demo-copy">
                              Paste one comment per line and run inference against the live backend.
                              This uses the same prediction API exposed to the extension workflow.
                            </p>
                            <textarea id="commentsInput" spellcheck="false" placeholder="Great tutorial and very clear explanations.
The video was useful but the audio quality needs work.
This helped me understand the concept much faster."></textarea>
                            <div class="demo-actions">
                              <button class="primary-button" id="runDemoButton" type="button">Run Sentiment Inference</button>
                              <button class="secondary-button" id="sampleButton" type="button">Load Sample Comments</button>
                              <div class="status-pill">
                                <span class="status-dot" id="statusDot"></span>
                                <span id="statusText">Ready for input</span>
                              </div>
                            </div>
                            <p class="footnote">
                              The deployed app accepts timestamped comments. This demo generates timestamps automatically
                              on the client so you can try the live model directly from the homepage.
                            </p>
                          </div>
                        </div>
                      </section>

                      <section class="detail-grid">
                        <div class="panel stack">
                          <div>
                            <div class="eyebrow">Inference Output</div>
                            <h2>Prediction Summary</h2>
                            <div class="meta-grid">
                              <div class="metric">
                                <div class="metric-label">Positive</div>
                                <div class="metric-value" id="positiveCount">0</div>
                              </div>
                              <div class="metric">
                                <div class="metric-label">Neutral</div>
                                <div class="metric-value" id="neutralCount">0</div>
                              </div>
                              <div class="metric">
                                <div class="metric-label">Negative</div>
                                <div class="metric-value" id="negativeCount">0</div>
                              </div>
                            </div>
                          </div>

                          <div>
                            <div class="eyebrow">Predictions</div>
                            <h2>Comment-Level Results</h2>
                            <p class="empty-state" id="predictionsEmpty">
                              Run the demo above to see model predictions for each input comment.
                            </p>
                            <div class="predictions" id="predictionsList"></div>
                          </div>
                        </div>

                        <div class="stack">
                          <section class="panel">
                            <div class="eyebrow">System Snapshot</div>
                            <h2>Deployment + Runtime</h2>
                            <div class="meta-grid">
                              <div class="metric">
                                <div class="metric-label">Service</div>
                                <div class="metric-value">youtube-sentiment-api</div>
                              </div>
                              <div class="metric">
                                <div class="metric-label">Version</div>
                                <div class="metric-value">3.0</div>
                              </div>
                              <div class="metric">
                                <div class="metric-label">Async Jobs</div>
                                <div class="metric-value">Enabled</div>
                              </div>
                              <div class="metric">
                                <div class="metric-label">Job Backend</div>
                                <div class="metric-value">Render + Local Async</div>
                              </div>
                            </div>
                          </section>

                          <section class="panel">
                            <div class="eyebrow">Architecture</div>
                            <h2>How the system works</h2>
                            <div class="architecture-list">
                              <div class="architecture-step">
                                <div class="architecture-step-index">1</div>
                                <div>
                                  <h3>Comment ingestion</h3>
                                  <p>The Chrome extension or homepage demo sends comments to the Flask API after client-side sanitization.</p>
                                </div>
                              </div>
                              <div class="architecture-step">
                                <div class="architecture-step-index">2</div>
                                <div>
                                  <h3>Model inference</h3>
                                  <p>The backend loads a TF-IDF vectorizer and trained classifier to predict positive, neutral, or negative sentiment.</p>
                                </div>
                              </div>
                              <div class="architecture-step">
                                <div class="architecture-step-index">3</div>
                                <div>
                                  <h3>Async analytics</h3>
                                  <p>Heavier jobs like insights, topics, trend charts, and word clouds run asynchronously so the prediction path stays fast.</p>
                                </div>
                              </div>
                              <div class="architecture-step">
                                <div class="architecture-step-index">4</div>
                                <div>
                                  <h3>Observability + ops</h3>
                                  <p>Health checks, Prometheus metrics, retries, dead-letter handling, and deployment validation make the system easier to operate.</p>
                                </div>
                              </div>
                            </div>
                          </section>

                          <section class="panel">
                            <div class="eyebrow">Explore</div>
                            <h2>Project Links</h2>
                            <div class="link-grid">
                              <a class="card-link" href="/readyz">
                                <div class="link-title">Readiness Check</div>
                                <div class="link-body">Verify model loading, async capabilities, and deployment health.</div>
                              </a>
                              <a class="card-link" href="/metrics">
                                <div class="link-title">Metrics Endpoint</div>
                                <div class="link-body">Inspect Prometheus-compatible request and runtime metrics.</div>
                              </a>
                              <a class="card-link" href="https://github.com/D-393Patel/CommentPulse" target="_blank" rel="noreferrer">
                                <div class="link-title">GitHub Repository</div>
                                <div class="link-body">Review the codebase, tests, Docker setup, CI, and ML pipeline.</div>
                              </a>
                            </div>
                          </section>
                        </div>
                      </section>
                    </main>
                    <script>
                      const sampleComments = [
                        "Great tutorial and very clear explanations.",
                        "The video was useful but the audio quality needs work.",
                        "This helped me understand the concept much faster.",
                        "Not bad, but I expected better examples in the second half."
                      ];

                      const commentsInput = document.getElementById("commentsInput");
                      const runDemoButton = document.getElementById("runDemoButton");
                      const sampleButton = document.getElementById("sampleButton");
                      const statusText = document.getElementById("statusText");
                      const statusDot = document.getElementById("statusDot");
                      const positiveCount = document.getElementById("positiveCount");
                      const neutralCount = document.getElementById("neutralCount");
                      const negativeCount = document.getElementById("negativeCount");
                      const predictionsList = document.getElementById("predictionsList");
                      const predictionsEmpty = document.getElementById("predictionsEmpty");

                      function setStatus(message, state) {
                        statusText.textContent = message;
                        statusDot.className = "status-dot";
                        if (state) {
                          statusDot.classList.add(state);
                        }
                      }

                      function sentimentMeta(sentiment) {
                        if (sentiment === 1) {
                          return { label: "Positive", className: "positive" };
                        }
                        if (sentiment === -1) {
                          return { label: "Negative", className: "negative" };
                        }
                        return { label: "Neutral", className: "neutral" };
                      }

                      function buildPayload(lines) {
                        const now = Date.now();
                        return {
                          comments: lines.map((text, index) => ({
                            text,
                            timestamp: new Date(now + index * 60000).toISOString()
                          }))
                        };
                      }

                      function renderPredictions(predictions) {
                        let positive = 0;
                        let neutral = 0;
                        let negative = 0;

                        predictionsList.innerHTML = "";

                        predictions.forEach((item) => {
                          const meta = sentimentMeta(item.sentiment);
                          if (item.sentiment === 1) positive += 1;
                          else if (item.sentiment === -1) negative += 1;
                          else neutral += 1;

                          const card = document.createElement("article");
                          card.className = "prediction-item";
                          card.innerHTML = `
                            <div class="prediction-meta">
                              <span class="sentiment-chip ${meta.className}">${meta.label}</span>
                              <span class="prediction-time">${new Date(item.timestamp).toLocaleString()}</span>
                            </div>
                            <p class="prediction-text"></p>
                          `;
                          card.querySelector(".prediction-text").textContent = item.comment;
                          predictionsList.appendChild(card);
                        });

                        positiveCount.textContent = String(positive);
                        neutralCount.textContent = String(neutral);
                        negativeCount.textContent = String(negative);
                        predictionsEmpty.style.display = predictions.length ? "none" : "block";
                      }

                      async function runDemo() {
                        const lines = commentsInput.value
                          .split(/\\r?\\n/)
                          .map((line) => line.trim())
                          .filter(Boolean);

                        if (!lines.length) {
                          setStatus("Add at least one comment to run inference.", "error");
                          return;
                        }

                        runDemoButton.disabled = true;
                        runDemoButton.textContent = "Running...";
                        setStatus("Calling live prediction API...", null);

                        try {
                          const response = await fetch("/predict_with_timestamps", {
                            method: "POST",
                            headers: { "Content-Type": "application/json" },
                            body: JSON.stringify(buildPayload(lines))
                          });

                          if (!response.ok) {
                            let message = "Inference request failed.";
                            try {
                              const errorPayload = await response.json();
                              message = errorPayload?.error?.message || message;
                            } catch (_error) {
                              // ignore JSON parse errors
                            }
                            throw new Error(message);
                          }

                          const predictions = await response.json();
                          renderPredictions(Array.isArray(predictions) ? predictions : []);
                          setStatus(`Completed live inference for ${lines.length} comment${lines.length === 1 ? "" : "s"}.`, "ready");
                        } catch (error) {
                          predictionsList.innerHTML = "";
                          positiveCount.textContent = "0";
                          neutralCount.textContent = "0";
                          negativeCount.textContent = "0";
                          predictionsEmpty.style.display = "block";
                          setStatus(error.message || "Something went wrong during inference.", "error");
                        } finally {
                          runDemoButton.disabled = false;
                          runDemoButton.textContent = "Run Sentiment Inference";
                        }
                      }

                      sampleButton.addEventListener("click", () => {
                        commentsInput.value = sampleComments.join("\\n");
                        setStatus("Loaded sample comments. Run the model when you're ready.", null);
                      });

                      runDemoButton.addEventListener("click", runDemo);
                    </script>
                  </body>
                </html>
                """
            ),
            mimetype="text/html",
        )

    @app.route("/health")
    @app.route("/livez")
    def liveness() -> Response:
        return jsonify({"status": "ok"})

    @app.route("/readyz")
    def readiness() -> tuple[Response, int]:
        dependencies = {
            "model": runtime is not None,
            "vectorizer": runtime is not None,
            "youtube_api_key": bool(config.youtube_api_key),
            "async_executor": True,
            "prometheus_enabled": PROMETHEUS_ENABLED,
            "job_backend": job_backend,
            "redis": redis_healthy if job_backend == "redis" else None,
            "max_job_attempts": config.max_job_attempts,
        }
        ready = dependencies["model"] and dependencies["vectorizer"] and dependencies["async_executor"]
        status = 200 if ready else 503
        return jsonify({"status": "ready" if ready else "degraded", "dependencies": dependencies}), status

    @app.route("/metrics")
    def metrics() -> Response:
        return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

    @app.route("/get_youtube_comments")
    def get_comments() -> Response:
        video_id = request.args.get("videoId", "").strip()
        if not video_id:
            raise ValidationError("videoId query parameter is required.")
        if not config.youtube_api_key:
            return error_response("YouTube API key not configured.", 503)

        url = "https://www.googleapis.com/youtube/v3/commentThreads"
        params = {
            "part": "snippet",
            "videoId": video_id,
            "maxResults": 100,
            "textFormat": "plainText",
            "key": config.youtube_api_key,
        }
        with requests.Session() as session:
            try:
                response = timed_external_request(
                    session,
                    "GET",
                    url,
                    provider="youtube",
                    timeout=config.request_timeout_seconds,
                    params=params,
                )
                data = response.json()
            except Exception:
                logger.exception("YouTube API fetch failed.")
                return error_response("Failed to fetch YouTube comments.", 502)

        comments = []
        for item in data.get("items", []):
            snippet = item["snippet"]["topLevelComment"]["snippet"]
            comments.append({"text": snippet["textOriginal"], "timestamp": snippet["publishedAt"]})
        return jsonify({"comments": comments, "count": len(comments)})

    @app.route("/predict_with_timestamps", methods=["POST"])
    def predict_with_timestamps() -> Response:
        payload = parse_json_body()
        comments = validate_comments(payload.get("comments"), config, require_timestamps=True)
        INFERENCE_COMMENT_COUNT.observe(len(comments))
        texts = [comment["text"] for comment in comments]
        predictions = ensure_runtime().predict_sentiments(texts)
        return jsonify(
            [
                {
                    "comment": text,
                    "sentiment": sentiment,
                    "timestamp": comment["timestamp"],
                }
                for text, sentiment, comment in zip(texts, predictions, comments)
            ]
        )

    @app.route("/generate_chart", methods=["POST"])
    def generate_chart() -> Response:
        payload = parse_json_body()
        counts = validate_sentiment_counts(payload.get("sentiment_counts"))
        image_bytes, content_type, _ = ensure_runtime().render_pie_chart(counts)
        return send_file(io.BytesIO(image_bytes), mimetype=content_type)

    @app.route("/generate_wordcloud", methods=["POST"])
    def generate_wordcloud() -> Response:
        payload = parse_json_body()
        comments = validate_comments(payload.get("comments"), config)
        image_bytes, content_type, _ = ensure_runtime().render_wordcloud(comments)
        return send_file(io.BytesIO(image_bytes), mimetype=content_type)

    @app.route("/extract_topics", methods=["POST"])
    def topics() -> Response:
        payload = parse_json_body()
        comments = validate_comments(payload.get("comments"), config)
        return jsonify({"topics": ensure_runtime().extract_topics(comments)})

    @app.route("/generate_insights", methods=["POST"])
    def insights() -> Response:
        payload = parse_json_body()
        comments = validate_comments(payload.get("comments"), config)
        return jsonify({"insights": ensure_runtime().generate_local_insights(comments)})

    @app.route("/generate_trend_graph", methods=["POST"])
    def generate_trend_graph() -> Response:
        payload = parse_json_body()
        sentiment_data = validate_sentiment_data(payload.get("sentiment_data"), config)
        image_bytes, content_type, _ = ensure_runtime().render_trend_graph(sentiment_data)
        return send_file(io.BytesIO(image_bytes), mimetype=content_type)

    @app.route("/generate_keyword_chart", methods=["POST"])
    def generate_keyword_chart() -> Response:
        payload = parse_json_body()
        comments = validate_comments(payload.get("comments"), config)
        image_bytes, content_type, _ = ensure_runtime().render_keyword_chart(comments)
        return send_file(io.BytesIO(image_bytes), mimetype=content_type)

    @app.route("/topic_sentiment", methods=["POST"])
    def topic_sentiment() -> Response:
        payload = parse_json_body()
        comments = validate_comments(payload.get("comments"), config, require_timestamps=True)
        INFERENCE_COMMENT_COUNT.observe(len(comments))
        return jsonify(ensure_runtime().compute_topic_sentiment(comments))

    @app.route("/jobs/insights", methods=["POST"])
    def create_insights_job() -> Response:
        payload = parse_json_body()
        comments = validate_comments(payload.get("comments"), config)
        return submit_json_job("insights", {"comments": comments})

    @app.route("/jobs/topics", methods=["POST"])
    def create_topics_job() -> Response:
        payload = parse_json_body()
        comments = validate_comments(payload.get("comments"), config)
        return submit_json_job("topics", {"comments": comments})

    @app.route("/jobs/topic-sentiment", methods=["POST"])
    def create_topic_sentiment_job() -> Response:
        payload = parse_json_body()
        comments = validate_comments(payload.get("comments"), config, require_timestamps=True)
        return submit_json_job("topic-sentiment", {"comments": comments})

    @app.route("/jobs/wordcloud", methods=["POST"])
    def create_wordcloud_job() -> Response:
        payload = parse_json_body()
        comments = validate_comments(payload.get("comments"), config)
        return submit_artifact_job("wordcloud", {"comments": comments})

    @app.route("/jobs/keyword-chart", methods=["POST"])
    def create_keyword_chart_job() -> Response:
        payload = parse_json_body()
        comments = validate_comments(payload.get("comments"), config)
        return submit_artifact_job("keyword-chart", {"comments": comments})

    @app.route("/jobs/trend-graph", methods=["POST"])
    def create_trend_graph_job() -> Response:
        payload = parse_json_body()
        sentiment_data = validate_sentiment_data(payload.get("sentiment_data"), config)
        return submit_artifact_job("trend-graph", {"sentiment_data": sentiment_data})

    @app.route("/jobs/<job_id>", methods=["GET"])
    def get_job(job_id: str) -> Response:
        job = jobs.get(job_id)
        if job is None:
            raise ValidationError("Job not found.", 404)
        payload = job.to_response()
        payload["status_url"] = url_for("get_job", job_id=job_id, _external=False)
        payload["artifact_url"] = (
            url_for("get_job_artifact", job_id=job_id, _external=False)
            if isinstance(job.result, bytes) or job.status in {"queued", "running"}
            else None
        )
        status_code = 200 if job.status in {"completed", "failed"} else 202
        return jsonify(payload), status_code

    @app.route("/jobs/<job_id>/artifact", methods=["GET"])
    def get_job_artifact(job_id: str) -> Response:
        artifact, content_type, filename = jobs.get_artifact(job_id)
        return send_file(
            io.BytesIO(artifact),
            mimetype=content_type,
            download_name=filename,
        )

    @app.route("/admin/jobs/dead-letter", methods=["GET"])
    def list_dead_letter_jobs() -> Response:
        backend = ensure_redis_backend()
        limit = request.args.get("limit", default=20, type=int)
        limit = max(1, min(limit, 100))
        return jsonify(
            {
                "job_backend": "redis",
                "dead_letter_jobs": backend.get_dead_letter_jobs(limit=limit),
            }
        )

    @app.route("/admin/jobs/<job_id>/replay", methods=["POST"])
    def replay_dead_letter_job(job_id: str) -> Response:
        backend = ensure_redis_backend()
        replayed = backend.replay_dead_letter_job(job_id)
        if not replayed:
            raise ValidationError("Dead-letter job not found.", 404)
        return jsonify({"job_id": job_id, "status": "requeued"})

    return app


app = create_app()


if __name__ == "__main__":
    config = app.config["APP_CONFIG"]
    app.run(host=config.host, port=config.port, debug=config.debug)
