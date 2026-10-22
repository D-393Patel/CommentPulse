import unittest
import logging
from types import SimpleNamespace
from unittest.mock import patch

from analytics_runtime import AnalyticsRuntime, load_stop_words


class FakeVectorizer:
    def __init__(self, feature_counts=None):
        self.feature_counts = feature_counts

    def transform(self, texts):
        return FakeSparseMatrix(self.feature_counts or [1] * len(texts))


class FakeSparseMatrix:
    def __init__(self, feature_counts):
        self.feature_counts = feature_counts

    def getrow(self, index):
        return SimpleNamespace(nnz=self.feature_counts[index])


class FakeModel:
    def __init__(self, classes, predictions):
        self.classes_ = classes
        self.predictions = predictions

    def predict(self, _matrix):
        return self.predictions


def make_runtime(classes, predictions, feature_counts=None):
    runtime = AnalyticsRuntime.__new__(AnalyticsRuntime)
    runtime.model = FakeModel(classes, predictions)
    runtime.vectorizer = FakeVectorizer(feature_counts)
    runtime.sentiment_label_map = runtime._build_sentiment_label_map()
    runtime.stop_words = set()
    runtime.has_wordnet = False
    runtime.lemmatizer = None
    return runtime


class SentimentRuntimeTestCase(unittest.TestCase):
    def test_fallback_stopwords_filter_common_english_fillers_but_keep_negation(self):
        with patch("analytics_runtime.nltk_find", side_effect=LookupError("missing stopwords")):
            words = load_stop_words(logging.getLogger(__name__))

        self.assertTrue({"you", "when", "this"}.issubset(words))
        self.assertNotIn("not", words)

    def test_training_class_ids_map_to_extension_labels(self):
        runtime = make_runtime([0, 1, 2], [])
        self.assertEqual(runtime.sentiment_label_map, {0: -1, 1: 0, 2: 1})

    def test_canonical_model_class_ids_remain_unchanged(self):
        runtime = make_runtime([-1, 0, 1], [])
        self.assertEqual(runtime.sentiment_label_map, {-1: -1, 0: 0, 1: 1})

    def test_clear_praise_and_complaints_override_missed_model_labels(self):
        texts = [
            "Awesome!!!!",
            "Thank you very much sir amazing lecture",
            "Nice explanation Thanks sir",
            "not good",
            "not bad",
        ]
        runtime = make_runtime([0, 1, 2], [0] * len(texts))
        self.assertEqual(runtime.predict_sentiments(texts), [1, 1, 1, -1, 1])

    def test_out_of_vocabulary_text_defaults_to_neutral_unless_cue_is_clear(self):
        runtime = make_runtime([0, 1, 2], [0, 0], feature_counts=[0, 0])
        self.assertEqual(runtime.predict_sentiments(["foreign topic words", "Awesome!!!!"]), [0, 1])

    def test_insight_counts_and_percentages_cover_every_prediction(self):
        comments = [
            "The algebra explanation is detailed.",
            "The course schedule changed yesterday.",
            "This chapter covers matrices.",
            "The lecture audio is unclear.",
            "The examples use a useful method.",
        ]
        runtime = make_runtime([0, 1, 2], [0, 1, 2, 0, 2])
        insights = runtime.generate_local_insights(comments)
        counts = insights["counts"]
        self.assertEqual(sum(counts.values()), len(comments))
        self.assertEqual(sum(insights["percentages"].values()), 100.0)

    def test_all_boilerplate_comments_return_empty_topics_and_valid_images(self):
        runtime = make_runtime([0, 1, 2], [0, 0])
        runtime.stop_words = {"thank", "you", "sir"}
        comments = ["Thank you sir", "Thank you sir"]
        self.assertEqual(runtime.extract_topics(comments), [])
        self.assertEqual(runtime.compute_topic_sentiment([{"text": text} for text in comments]), {"topics": []})

        for image_bytes, content_type, _ in (
            runtime.render_wordcloud(comments),
            runtime.render_keyword_chart(comments),
            runtime.render_pie_chart({"1": 0, "0": 0, "-1": 0}),
        ):
            self.assertEqual(content_type, "image/png")
            self.assertTrue(image_bytes.startswith(b"\x89PNG\r\n\x1a\n"))


if __name__ == "__main__":
    unittest.main()
