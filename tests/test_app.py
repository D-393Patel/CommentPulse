import time
import unittest
from unittest import mock

import app


class AppTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = app.app.test_client()

    def wait_for_job(self, job_id, timeout=10.0, client=None):
        client = client or self.client
        deadline = time.time() + timeout
        last_status = None

        while time.time() < deadline:
            response = client.get(f"/jobs/{job_id}")
            self.assertIn(response.status_code, (200, 202))
            payload = response.get_json()
            last_status = payload
            if payload["status"] in {"completed", "failed"}:
                return payload
            time.sleep(0.1)

        self.fail(f"Timed out waiting for job {job_id}. Last status: {last_status}")

    def test_health_and_readiness_endpoints(self):
        health_response = self.client.get("/health")
        self.assertEqual(health_response.status_code, 200)
        self.assertEqual(health_response.get_json()["status"], "ok")

        readiness_response = self.client.get("/readyz")
        self.assertEqual(readiness_response.status_code, 200)
        readiness_payload = readiness_response.get_json()
        self.assertEqual(readiness_payload["status"], "ready")
        self.assertIn("async_executor", readiness_payload["dependencies"])

    def test_validation_error_contract(self):
        response = self.client.post("/generate_chart", json={})
        self.assertEqual(response.status_code, 400)
        payload = response.get_json()
        self.assertIn("error", payload)
        self.assertEqual(payload["error"]["status_code"], 400)
        self.assertIn("request_id", payload["error"])

    def test_insights_job_completes_with_local_summary(self):
        payload = {
            "comments": [
                "I love this breakdown and editing style.",
                "This was useful but the audio is rough.",
                "Great explanation and very clear examples.",
            ]
        }

        create_response = self.client.post("/jobs/insights", json=payload)
        self.assertEqual(create_response.status_code, 202)
        job_id = create_response.get_json()["job_id"]

        result = self.wait_for_job(job_id)
        self.assertEqual(result["status"], "completed")
        self.assertIn("insights", result["result"])
        self.assertIn("summary", result["result"]["insights"])
        self.assertIn("Analyzed", result["result"]["insights"]["summary"])

    def test_keyword_chart_job_returns_artifact(self):
        payload = {
            "comments": [
                "great tutorial",
                "useful tutorial with practical tips",
                "bad mic but great content",
            ]
        }

        create_response = self.client.post("/jobs/keyword-chart", json=payload)
        self.assertEqual(create_response.status_code, 202)
        create_payload = create_response.get_json()
        self.assertIsNotNone(create_payload["artifact_url"])

        result = self.wait_for_job(create_payload["job_id"])
        self.assertEqual(result["status"], "completed")

        artifact_response = self.client.get(create_payload["artifact_url"])
        self.assertEqual(artifact_response.status_code, 200)
        self.assertTrue(artifact_response.content_type.startswith("image/"))

    def test_topic_sentiment_job_returns_topics(self):
        payload = {
            "comments": [
                {"text": "great visuals and pacing", "timestamp": "2026-03-31T00:00:00Z"},
                {"text": "bad audio quality", "timestamp": "2026-03-31T00:01:00Z"},
                {"text": "great tutorial and examples", "timestamp": "2026-03-31T00:02:00Z"},
            ]
        }

        create_response = self.client.post("/jobs/topic-sentiment", json=payload)
        self.assertEqual(create_response.status_code, 202)
        result = self.wait_for_job(create_response.get_json()["job_id"])

        self.assertEqual(result["status"], "completed")
        self.assertIn("topics", result["result"])
        self.assertIsInstance(result["result"]["topics"], list)

    def test_local_job_retries_before_success(self):
        attempts = {"count": 0}

        def flaky_insights(_payload):
            attempts["count"] += 1
            if attempts["count"] < 2:
                raise RuntimeError("transient failure")
            return {"insights": {"summary": "Recovered after retry"}}, None, None

        with mock.patch.dict("os.environ", {"MAX_JOB_ATTEMPTS": "2"}, clear=False):
            with mock.patch.object(
                app,
                "create_job_handlers",
                return_value={
                    "insights": flaky_insights,
                    "topics": lambda payload: ({"topics": []}, None, None),
                    "topic-sentiment": lambda payload: ({"topics": []}, None, None),
                    "wordcloud": lambda payload: (b"png", "image/png", "wordcloud.png"),
                    "keyword-chart": lambda payload: (b"png", "image/png", "keyword-chart.png"),
                    "trend-graph": lambda payload: (b"png", "image/png", "trend-graph.png"),
                },
            ):
                test_app = app.create_app()
                client = test_app.test_client()
                create_response = client.post("/jobs/insights", json={"comments": ["retry me"]})
                self.assertEqual(create_response.status_code, 202)
                result = self.wait_for_job(create_response.get_json()["job_id"], client=client)

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["attempts"], 2)
        self.assertEqual(result["result"]["insights"]["summary"], "Recovered after retry")

    def test_local_job_dead_letters_after_max_attempts(self):
        def failing_insights(_payload):
            raise RuntimeError("permanent failure")

        with mock.patch.dict("os.environ", {"MAX_JOB_ATTEMPTS": "2"}, clear=False):
            with mock.patch.object(
                app,
                "create_job_handlers",
                return_value={
                    "insights": failing_insights,
                    "topics": lambda payload: ({"topics": []}, None, None),
                    "topic-sentiment": lambda payload: ({"topics": []}, None, None),
                    "wordcloud": lambda payload: (b"png", "image/png", "wordcloud.png"),
                    "keyword-chart": lambda payload: (b"png", "image/png", "keyword-chart.png"),
                    "trend-graph": lambda payload: (b"png", "image/png", "trend-graph.png"),
                },
            ):
                test_app = app.create_app()
                client = test_app.test_client()
                create_response = client.post("/jobs/insights", json={"comments": ["fail me"]})
                self.assertEqual(create_response.status_code, 202)
                job_id = create_response.get_json()["job_id"]

                deadline = time.time() + 10
                result = None
                while time.time() < deadline:
                    response = client.get(f"/jobs/{job_id}")
                    result = response.get_json()
                    if result["status"] == "failed":
                        break
                    time.sleep(0.1)
                else:
                    self.fail(f"Timed out waiting for permanent failure. Last status: {result}")

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["attempts"], 2)
        self.assertTrue(result["dead_lettered"])
        self.assertIn("permanent failure", result["error"])

    def test_dead_letter_admin_endpoints_require_redis_backend(self):
        list_response = self.client.get("/admin/jobs/dead-letter")
        self.assertEqual(list_response.status_code, 409)
        self.assertIn("Redis job backend is not active", list_response.get_json()["error"]["message"])

        replay_response = self.client.post("/admin/jobs/fake-job-id/replay")
        self.assertEqual(replay_response.status_code, 409)
        self.assertIn("Redis job backend is not active", replay_response.get_json()["error"]["message"])


if __name__ == "__main__":
    unittest.main()
