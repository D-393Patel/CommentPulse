import base64
import json
import logging
import os
import time
from datetime import UTC, datetime
from pathlib import Path

from analytics_runtime import AnalyticsRuntime, create_job_handlers

try:
    import redis
except ImportError as exc:
    raise RuntimeError("redis package is required for worker mode.") from exc


BASE_DIR = Path(__file__).resolve().parent


def utc_now() -> datetime:
    return datetime.now(UTC)


def configure_logging() -> logging.Logger:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    return logging.getLogger("youtube_sentiment_worker")


def job_key(job_id: str) -> str:
    return f"job:{job_id}"


def write_job(
    client: redis.Redis,
    ttl_seconds: int,
    job_id: str,
    job_type: str,
    status: str,
    created_at: str,
    result=None,
    error: str | None = None,
    content_type: str | None = None,
    artifact_filename: str | None = None,
    attempts: int = 0,
    max_attempts: int = 1,
    dead_lettered: bool = False,
) -> None:
    result_is_bytes = isinstance(result, bytes)
    if result_is_bytes:
        result = base64.b64encode(result).decode("utf-8")
    payload = {
        "job_id": job_id,
        "job_type": job_type,
        "status": status,
        "created_at": created_at,
        "updated_at": utc_now().isoformat(),
        "completed_at": utc_now().isoformat() if status in {"completed", "failed"} else None,
        "result": result,
        "result_is_bytes": result_is_bytes,
        "error": error,
        "content_type": content_type,
        "artifact_filename": artifact_filename,
        "attempts": attempts,
        "max_attempts": max_attempts,
        "dead_lettered": dead_lettered,
    }
    client.setex(job_key(job_id), ttl_seconds, json.dumps(payload))


def main() -> None:
    logger = configure_logging()
    redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")
    queue_name = os.getenv("JOB_QUEUE_NAME", "analytics_jobs")
    dead_letter_queue_name = os.getenv("DEAD_LETTER_QUEUE_NAME", "analytics_jobs_dead_letter")
    ttl_seconds = int(os.getenv("JOB_TTL_SECONDS", "3600"))
    max_job_attempts = int(os.getenv("MAX_JOB_ATTEMPTS", "3"))
    model_path = BASE_DIR / "lgbm_model.pkl"
    vectorizer_path = BASE_DIR / "tfidf_vectorizer.pkl"

    client = redis.Redis.from_url(redis_url, decode_responses=True)
    runtime = AnalyticsRuntime(model_path, vectorizer_path, logger)
    handlers = create_job_handlers(runtime)

    logger.info("Worker started and waiting for jobs.")
    while True:
        try:
            item = client.blpop(queue_name, timeout=5)
            if item is None:
                continue

            _, raw_message = item
            message = json.loads(raw_message)
            job_id = message["job_id"]
            job_type = message["job_type"]
            payload = message["payload"]

            existing_raw = client.get(job_key(job_id))
            created_at = utc_now().isoformat()
            attempts = 0
            if existing_raw:
                existing_payload = json.loads(existing_raw)
                created_at = existing_payload.get("created_at", created_at)
                attempts = existing_payload.get("attempts", 0)

            attempts += 1
            write_job(
                client,
                ttl_seconds,
                job_id,
                job_type,
                "running",
                created_at,
                attempts=attempts,
                max_attempts=max_job_attempts,
            )

            try:
                started_at = time.perf_counter()
                result, content_type, artifact_filename = handlers[job_type](payload)
                duration = time.perf_counter() - started_at

                write_job(
                    client,
                    ttl_seconds,
                    job_id,
                    job_type,
                    "completed",
                    created_at,
                    result=result,
                    content_type=content_type,
                    artifact_filename=artifact_filename,
                    attempts=attempts,
                    max_attempts=max_job_attempts,
                )
                logger.info("Completed job %s (%s) in %.2fs", job_id, job_type, duration)
            except Exception as exc:
                if attempts < max_job_attempts:
                    write_job(
                        client,
                        ttl_seconds,
                        job_id,
                        job_type,
                        "queued",
                        created_at,
                        error=str(exc),
                        attempts=attempts,
                        max_attempts=max_job_attempts,
                    )
                    client.rpush(queue_name, json.dumps(message))
                    logger.warning(
                        "Retrying job %s (%s), attempt %s/%s after error: %s",
                        job_id,
                        job_type,
                        attempts,
                        max_job_attempts,
                        exc,
                    )
                else:
                    write_job(
                        client,
                        ttl_seconds,
                        job_id,
                        job_type,
                        "failed",
                        created_at,
                        error=str(exc),
                        attempts=attempts,
                        max_attempts=max_job_attempts,
                        dead_lettered=True,
                    )
                    client.rpush(dead_letter_queue_name, json.dumps(message))
                    logger.exception("Job %s (%s) failed permanently: %s", job_id, job_type, exc)
        except Exception as exc:
            logger.exception("Worker loop error: %s", exc)


if __name__ == "__main__":
    main()
