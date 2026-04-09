import json
import os
import subprocess
import sys
import time
import unittest
from pathlib import Path
from unittest import skipUnless

import redis

import app


WORKSPACE = Path(__file__).resolve().parents[1]
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
RUN_REDIS_INTEGRATION = os.getenv("RUN_REDIS_INTEGRATION", "0") == "1"


def redis_available() -> bool:
    try:
        client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
        client.ping()
        return True
    except Exception:
        return False


@skipUnless(RUN_REDIS_INTEGRATION and redis_available(), "Redis integration test requires RUN_REDIS_INTEGRATION=1 and a reachable Redis server.")
class RedisIntegrationTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client_redis = redis.Redis.from_url(REDIS_URL, decode_responses=True)
        cls.client_redis.flushdb()

        cls.worker_process = subprocess.Popen(
            [sys.executable, "worker.py"],
            cwd=str(WORKSPACE),
            env={
                **os.environ,
                "REDIS_URL": REDIS_URL,
                "JOB_QUEUE_NAME": "analytics_jobs",
                "DEAD_LETTER_QUEUE_NAME": "analytics_jobs_dead_letter",
                "MAX_JOB_ATTEMPTS": "2",
                "LOG_LEVEL": "WARNING",
            },
        )

        for _ in range(30):
            if cls.worker_process.poll() is not None:
                raise RuntimeError("Worker exited unexpectedly during startup.")
            time.sleep(0.1)

        previous_env = os.environ.copy()
        cls._previous_env = previous_env
        os.environ["REDIS_URL"] = REDIS_URL
        os.environ["JOB_QUEUE_NAME"] = "analytics_jobs"
        os.environ["DEAD_LETTER_QUEUE_NAME"] = "analytics_jobs_dead_letter"
        os.environ["MAX_JOB_ATTEMPTS"] = "2"
        os.environ["LOG_LEVEL"] = "WARNING"

        cls.test_app = app.create_app()
        cls.client = cls.test_app.test_client()

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "worker_process"):
            cls.worker_process.terminate()
            try:
                cls.worker_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                cls.worker_process.kill()

        if hasattr(cls, "client_redis"):
            cls.client_redis.flushdb()

        for key in ["REDIS_URL", "JOB_QUEUE_NAME", "DEAD_LETTER_QUEUE_NAME", "MAX_JOB_ATTEMPTS", "LOG_LEVEL"]:
            if key in cls._previous_env:
                os.environ[key] = cls._previous_env[key]
            elif key in os.environ:
                del os.environ[key]

    def wait_for_job(self, job_id: str, timeout: float = 15.0):
        deadline = time.time() + timeout
        last_payload = None
        while time.time() < deadline:
            response = self.client.get(f"/jobs/{job_id}")
            self.assertIn(response.status_code, (200, 202))
            payload = response.get_json()
            last_payload = payload
            if payload["status"] in {"completed", "failed"}:
                return payload
            time.sleep(0.2)
        self.fail(f"Timed out waiting for Redis-backed job {job_id}. Last payload: {last_payload}")

    def test_redis_backed_insights_job_completes(self):
        response = self.client.post(
            "/jobs/insights",
            json={"comments": ["great video", "very helpful", "bad audio but useful"]},
        )
        self.assertEqual(response.status_code, 202)
        payload = response.get_json()

        result = self.wait_for_job(payload["job_id"])
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["attempts"], 1)
        self.assertIn("insights", result["result"])

    def test_dead_letter_listing_and_replay_with_unknown_job(self):
        manager = app.RedisJobManager(
            REDIS_URL,
            "analytics_jobs",
            "analytics_jobs_dead_letter",
            3600,
            app.logging.getLogger("redis-test"),
            2,
        )
        submitted = manager.submit("unknown-job-type", {"comments": ["should fail"]})

        failed = self.wait_for_job(submitted.job_id)
        self.assertEqual(failed["status"], "failed")
        self.assertTrue(failed["dead_lettered"])
        self.assertEqual(failed["attempts"], 2)

        dead_letter_response = self.client.get("/admin/jobs/dead-letter")
        self.assertEqual(dead_letter_response.status_code, 200)
        dead_letter_payload = dead_letter_response.get_json()
        self.assertTrue(any(job["job_id"] == submitted.job_id for job in dead_letter_payload["dead_letter_jobs"]))

        replay_response = self.client.post(f"/admin/jobs/{submitted.job_id}/replay")
        self.assertEqual(replay_response.status_code, 200)
        self.assertEqual(replay_response.get_json()["status"], "requeued")

        replayed = self.wait_for_job(submitted.job_id)
        self.assertEqual(replayed["status"], "failed")
        self.assertTrue(replayed["dead_lettered"])


if __name__ == "__main__":
    unittest.main()
