import csv
import json
import unittest
from pathlib import Path
import uuid
import shutil

from src.data import dataset_curation
from src.model import promote_baseline


WORKSPACE = Path(__file__).resolve().parents[1]
TEST_TMP_ROOT = WORKSPACE / ".tmp" / "test_ml_data_workflow"


def make_temp_workspace() -> Path:
    workspace = TEST_TMP_ROOT / str(uuid.uuid4())
    workspace.mkdir(parents=True, exist_ok=True)
    return workspace


class DatasetCurationTestCase(unittest.TestCase):
    def test_prepare_labeling_queue_deduplicates_and_shapes_output(self):
        workspace = make_temp_workspace()
        self.addCleanup(lambda: shutil.rmtree(workspace, ignore_errors=True))
        input_path = workspace / "comments.json"
        output_path = workspace / "labeling_queue.csv"

        input_path.write_text(
            json.dumps(
                {
                    "comments": [
                        "Great video",
                        "Great video",
                        "  ",
                        "Needs more examples",
                    ]
                }
            ),
            encoding="utf-8",
        )

        summary = dataset_curation.prepare_labeling_queue(
            str(input_path),
            str(output_path),
            "capture_one",
            existing_labeled_path=None,
            batch_size=0,
            add_model_suggestions=False,
        )

        self.assertEqual(summary["prepared_examples"], 2)
        self.assertTrue(output_path.exists())

        with output_path.open("r", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["sentiment_label"], "")
        self.assertEqual(rows[0]["source_dataset"], "capture_one")
        self.assertEqual(rows[0]["needs_review"], "True")
        self.assertEqual(rows[0]["suggested_label"], "")
        self.assertEqual(rows[0]["review_priority_rank"], "1")
        self.assertIn(rows[0]["review_priority_reason"], {"no_model_signal", "model_uncertain", "high_confidence_spot_check"})

    def test_prepare_labeling_queue_skips_existing_comments_and_batches_output(self):
        workspace = make_temp_workspace()
        self.addCleanup(lambda: shutil.rmtree(workspace, ignore_errors=True))
        input_path = workspace / "comments.csv"
        existing_path = workspace / "existing.csv"
        output_path = workspace / "labeling_queue.csv"

        input_path.write_text(
            "comment_text\n"
            "Great video\n"
            "Already labeled\n"
            "Needs examples\n",
            encoding="utf-8",
        )
        existing_path.write_text(
            "comment_text,sentiment_label\n"
            "Already labeled,positive\n",
            encoding="utf-8",
        )

        summary = dataset_curation.prepare_labeling_queue(
            str(input_path),
            str(output_path),
            "capture_two",
            existing_labeled_path=str(existing_path),
            batch_size=1,
            add_model_suggestions=False,
        )

        self.assertEqual(summary["prepared_examples"], 2)
        self.assertEqual(summary["already_labeled_skipped"], 1)

        with output_path.open("r", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        self.assertEqual(rows[0]["batch_id"], "capture_two-batch-01-of-02")
        self.assertEqual(rows[1]["batch_id"], "capture_two-batch-02-of-02")
        self.assertEqual(rows[0]["review_priority_rank"], "1")
        self.assertEqual(rows[1]["review_priority_rank"], "2")

    def test_merge_labeled_data_keeps_latest_unique_label(self):
        workspace = make_temp_workspace()
        self.addCleanup(lambda: shutil.rmtree(workspace, ignore_errors=True))
        base_path = workspace / "base.csv"
        new_path = workspace / "new.csv"
        output_path = workspace / "merged.csv"

        base_path.write_text(
            "comment_text,sentiment_label\n"
            "Great pacing,positive\n"
            "Bad audio,negative\n",
            encoding="utf-8",
        )
        new_path.write_text(
            "comment_text,sentiment_label\n"
            "Bad audio,neutral\n"
            "Very clear explanation,positive\n"
            "Ignore me,\n",
            encoding="utf-8",
        )

        summary = dataset_curation.merge_labeled_data(str(base_path), str(new_path), str(output_path))

        self.assertEqual(summary["base_examples"], 2)
        self.assertEqual(summary["new_valid_examples"], 2)
        self.assertEqual(summary["merged_examples"], 3)

        with output_path.open("r", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        merged = {row["comment_text"]: row["sentiment_label"] for row in rows}
        self.assertEqual(merged["Bad audio"], "neutral")
        self.assertEqual(merged["Very clear explanation"], "positive")


class PromoteBaselineTestCase(unittest.TestCase):
    def test_promote_baseline_requires_promote_recommendation(self):
        workspace = make_temp_workspace()
        self.addCleanup(lambda: shutil.rmtree(workspace, ignore_errors=True))
        comparison_path = workspace / "comparison.json"
        candidate_path = workspace / "candidate.json"
        baseline_path = workspace / "baseline.json"

        comparison_path.write_text(json.dumps({"recommendation": "hold"}), encoding="utf-8")
        candidate_path.write_text(json.dumps({"summary_metrics": {"macro_f1": 0.4}}), encoding="utf-8")

        with self.assertRaises(ValueError):
            promote_baseline.promote_baseline(
                str(comparison_path),
                str(candidate_path),
                str(baseline_path),
            )

        result = promote_baseline.promote_baseline(
            str(comparison_path),
            str(candidate_path),
            str(baseline_path),
            force=True,
        )
        self.assertEqual(result["status"], "promoted")
        self.assertTrue(baseline_path.exists())

    def test_promote_baseline_copies_candidate_report(self):
        workspace = make_temp_workspace()
        self.addCleanup(lambda: shutil.rmtree(workspace, ignore_errors=True))
        comparison_path = workspace / "comparison.json"
        candidate_path = workspace / "candidate.json"
        baseline_path = workspace / "baseline.json"

        comparison_path.write_text(json.dumps({"recommendation": "promote"}), encoding="utf-8")
        candidate_payload = {"summary_metrics": {"macro_f1": 0.72, "accuracy": 0.81}}
        candidate_path.write_text(json.dumps(candidate_payload), encoding="utf-8")

        result = promote_baseline.promote_baseline(
            str(comparison_path),
            str(candidate_path),
            str(baseline_path),
        )

        self.assertEqual(result["summary_metrics"]["macro_f1"], 0.72)
        copied_payload = json.loads(baseline_path.read_text(encoding="utf-8"))
        self.assertEqual(copied_payload, candidate_payload)


class DatasetAuditTestCase(unittest.TestCase):
    def test_audit_labeled_dataset_reports_invalid_labels_and_duplicates(self):
        workspace = make_temp_workspace()
        self.addCleanup(lambda: shutil.rmtree(workspace, ignore_errors=True))
        input_path = workspace / "labeled.csv"
        input_path.write_text(
            "comment_text,sentiment_label\n"
            "Helpful video,positive\n"
            "Helpful video,neutral\n"
            "Missing label,\n"
            "Bad audio,negative\n",
            encoding="utf-8",
        )

        report = dataset_curation.audit_labeled_dataset(str(input_path))

        self.assertEqual(report["total_rows"], 4)
        self.assertEqual(report["valid_labeled_rows"], 3)
        self.assertEqual(report["invalid_label_rows"], 1)
        self.assertEqual(report["duplicate_comment_rows"], 2)
        self.assertTrue(report["quality_checks"]["has_invalid_labels"])
        self.assertTrue(report["quality_checks"]["has_duplicate_comments"])


class DatasetBootstrapTestCase(unittest.TestCase):
    def test_bootstrap_pseudo_labels_filters_by_confidence_and_label(self):
        workspace = make_temp_workspace()
        self.addCleanup(lambda: shutil.rmtree(workspace, ignore_errors=True))
        input_path = workspace / "queue.csv"
        output_path = workspace / "pseudo.csv"
        input_path.write_text(
            "comment_text,suggested_label,suggested_confidence,source_dataset,batch_id,example_id,labeler_notes\n"
            "Great explanation,positive,0.91,capture,batch-01,id-1,\n"
            "Maybe okay,neutral,0.60,capture,batch-01,id-2,\n"
            "Bad audio,negative,0.88,capture,batch-01,id-3,\n"
            "Unknown row,,0.99,capture,batch-01,id-4,\n",
            encoding="utf-8",
        )

        summary = dataset_curation.bootstrap_pseudo_labels(
            str(input_path),
            str(output_path),
            min_confidence=0.8,
        )

        self.assertEqual(summary["candidate_examples"], 2)
        with output_path.open("r", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["label_origin"] == "pseudo_label" for row in rows))
        self.assertTrue(all(row["needs_review"] == "True" for row in rows))

    def test_prepare_review_file_preserves_priority_fields(self):
        workspace = make_temp_workspace()
        self.addCleanup(lambda: shutil.rmtree(workspace, ignore_errors=True))
        input_path = workspace / "pseudo.csv"
        output_path = workspace / "review.csv"
        input_path.write_text(
            "comment_text,sentiment_label,label_origin,suggested_confidence,source_dataset,batch_id,example_id,needs_review,labeler_notes,review_priority_rank,review_priority_score,review_priority_reason\n"
            "Great explanation,positive,pseudo_label,0.91,capture,batch-01,id-1,True,,1,9.0,high_confidence_spot_check\n",
            encoding="utf-8",
        )

        summary = dataset_curation.prepare_review_file(str(input_path), str(output_path))

        self.assertEqual(summary["review_examples"], 1)
        with output_path.open("r", encoding="utf-8") as file:
            row = next(csv.DictReader(file))
        self.assertEqual(row["review_priority_rank"], "1")
        self.assertEqual(row["review_priority_reason"], "high_confidence_spot_check")


if __name__ == "__main__":
    unittest.main()
