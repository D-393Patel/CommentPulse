import io
import logging
import os
import pickle
import re
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any

from nltk.corpus import stopwords
from nltk.data import find as nltk_find
from nltk.stem import WordNetLemmatizer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
warnings.filterwarnings(
    "ignore",
    message="Could not find the number of physical cores",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
    category=UserWarning,
)

NEWLINE_RE = re.compile(r"\n")
NON_ALNUM_RE = re.compile(r"[^a-z0-9 ]")
NUMBER_RE = re.compile(r"\b\d+\b")
POSITIVE_CUE_RE = re.compile(
    r"\b(?:not\s+(?:bad|terrible|awful)|thank\s+you|thanks|thx|dhanyavaad|shukriya|awesome|amazing|excellent|"
    r"fantastic|wonderful|brilliant|helpful|useful|informative|understandable|clear|explained|well\s+explained|clear\s+explanation|"
    r"great|good|nice|love|loved|best|superb|zabardast|accha|achha|acha|"
    r"bahut\s+(?:accha|achha|acha)|bohot\s+(?:accha|achha|acha))\b",
    re.IGNORECASE,
)
NEGATIVE_CUE_RE = re.compile(
    r"\b(?:not\s+(?:good|great|helpful|clear|useful|working)|don['’]?t\s+like|"
    r"didn['’]?t\s+like|doesn['’]?t\s+work|isn['’]?t\s+working|"
    r"waste(?:d)?|useless|awful|terrible|horrible|worst|boring|confusing|"
    r"incorrect|wrong|poor|bad|hate|bekar|bakwas|ghatiya|samajh\s+nahi\s+aaya)\b",
    re.IGNORECASE,
)
POSITIVE_NEGATION_RE = re.compile(
    r"\b(?:not|never|no|don['’]?t|didn['’]?t|doesn['’]?t|isn['’]?t|wasn['’]?t)"
    r"(?:\s+\w+){0,2}\s+(?:awesome|amazing|excellent|fantastic|wonderful|"
    r"brilliant|helpful|great|good|nice|love|loved|best|superb|accha|achha|acha)\b",
    re.IGNORECASE,
)
NEGATIVE_NEGATION_RE = re.compile(
    r"\b(?:not|never|no|isn['’]?t|wasn['’]?t|aren['’]?t)"
    r"(?:\s+\w+){0,2}\s+(?:bad|boring|wrong|poor|useless|awful|terrible|horrible)\b",
    re.IGNORECASE,
)
TOPIC_STOP_WORDS = {
    "sir", "madam", "thanks", "thank", "thankyou", "thx", "please", "video",
    "lecture", "subscribe", "like", "comment", "awesome", "amazing", "great",
    "good", "nice", "best", "sirji", "bro", "brother", "dear", "test",
}


def _load_pandas():
    import pandas as pd

    return pd


def _load_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    return plt, mdates


def _load_wordcloud():
    from wordcloud import WordCloud

    return WordCloud


def load_stop_words(logger: logging.Logger) -> set[str]:
    fallback_words = set(ENGLISH_STOP_WORDS)
    try:
        nltk_find("corpora/stopwords")
        words = set(stopwords.words("english"))
        return (words - {"not", "no", "but"}) | TOPIC_STOP_WORDS
    except LookupError:
        logger.warning("NLTK stopwords corpus not available, using fallback stopword set.")
        return (fallback_words - {"not", "no", "but"}) | TOPIC_STOP_WORDS


def wordnet_available() -> bool:
    try:
        nltk_find("corpora/wordnet")
        return True
    except LookupError:
        return False


class AnalyticsRuntime:
    def __init__(self, model_path: Path, vectorizer_path: Path, logger: logging.Logger) -> None:
        self.logger = logger
        self.stop_words = load_stop_words(logger)
        self.lemmatizer = WordNetLemmatizer()
        self.has_wordnet = wordnet_available()
        self.model = None
        self.vectorizer = None

        with model_path.open("rb") as model_file:
            self.model = pickle.load(model_file)
        with vectorizer_path.open("rb") as vectorizer_file:
            self.vectorizer = pickle.load(vectorizer_file)
        self.sentiment_label_map = self._build_sentiment_label_map()

    def _build_sentiment_label_map(self) -> dict[int, int]:
        """Translate training labels to the public API contract (-1/0/1)."""
        classes = getattr(self.model, "classes_", None)
        try:
            class_values = {int(value) for value in classes} if classes is not None else set()
        except (TypeError, ValueError):
            class_values = set()

        if -1 in class_values:
            return {-1: -1, 0: 0, 1: 1}
        if class_values and class_values.issubset({0, 1, 2}):
            # params.yaml maps negative/neutral/positive to 0/1/2.
            return {0: -1, 1: 0, 2: 1}
        raise ValueError(
            f"Unsupported sentiment model classes {sorted(class_values)}; expected -1/0/1 or 0/1/2."
        )

    @lru_cache(maxsize=8192)
    def _preprocess_comment_cached(self, text: str) -> str:
        normalized = str(text).lower()
        normalized = NEWLINE_RE.sub(" ", normalized)
        normalized = NUMBER_RE.sub(" ", normalized)
        normalized = NON_ALNUM_RE.sub("", normalized)
        words = [word for word in normalized.split() if word not in self.stop_words]
        if self.has_wordnet:
            words = [self.lemmatizer.lemmatize(word) for word in words]
        return " ".join(words)

    def preprocess_comment(self, text: str) -> str:
        return self._preprocess_comment_cached(str(text))

    def top_terms_from_comments(self, comments: list[str], limit: int = 5) -> list[str]:
        processed = [self.preprocess_comment(comment) for comment in comments if str(comment).strip()]
        processed = [comment for comment in processed if comment]
        if not processed:
            return []
        tfidf = TfidfVectorizer(stop_words=sorted(self.stop_words), max_features=50, ngram_range=(1, 2))
        matrix = tfidf.fit_transform(processed)
        scores = matrix.sum(axis=0).A1
        terms = tfidf.get_feature_names_out()
        ranked = sorted(zip(terms, scores), key=lambda item: item[1], reverse=True)
        return [term for term, _ in ranked[:limit]]

    def predict_sentiments(self, texts: list[str]) -> list[int]:
        processed = [self.preprocess_comment(text) for text in texts]
        matrix = self.vectorizer.transform(processed)
        try:
            predictions = self.model.predict(matrix)
        except Exception:
            pd = _load_pandas()
            feature_frame = pd.DataFrame.sparse.from_spmatrix(
                matrix,
                columns=self.vectorizer.get_feature_names_out(),
            )
            predictions = self.model.predict(feature_frame)
        mapped = [self.sentiment_label_map[int(prediction)] for prediction in predictions]
        try:
            has_model_features = [matrix.getrow(index).nnz > 0 for index in range(len(texts))]
        except AttributeError:
            has_model_features = [True] * len(texts)
        return [
            self._apply_clear_sentiment_cue(text, sentiment, has_features)
            for text, sentiment, has_features in zip(texts, mapped, has_model_features)
        ]

    @staticmethod
    def _apply_clear_sentiment_cue(
        text: str,
        model_sentiment: int,
        has_model_features: bool = True,
    ) -> int:
        """Correct obvious short praise/complaints the general-purpose model misses."""
        normalized = str(text).strip()
        if POSITIVE_NEGATION_RE.search(normalized):
            return -1
        if NEGATIVE_NEGATION_RE.search(normalized):
            return 1
        positive = bool(POSITIVE_CUE_RE.search(normalized)) and not bool(
            POSITIVE_NEGATION_RE.search(normalized)
        )
        negative = bool(NEGATIVE_CUE_RE.search(normalized)) and not bool(
            NEGATIVE_NEGATION_RE.search(normalized)
        )
        if positive != negative:
            return 1 if positive else -1
        if not has_model_features:
            return 0
        return model_sentiment

    def extract_topics(self, comments: list[str]) -> list[dict[str, Any]]:
        processed = [self.preprocess_comment(comment) for comment in comments if comment.strip()]
        processed = [comment for comment in processed if comment]
        if not processed:
            return []
        tfidf = TfidfVectorizer(stop_words=sorted(self.stop_words), max_features=50, ngram_range=(1, 2))
        matrix = tfidf.fit_transform(processed)
        cluster_count = min(3, len(set(processed)))
        kmeans = KMeans(n_clusters=cluster_count, random_state=42, n_init=10)
        kmeans.fit(matrix)
        terms = tfidf.get_feature_names_out()
        topics = []
        for index, center in enumerate(kmeans.cluster_centers_):
            keywords = [terms[i] for i in center.argsort()[-5:][::-1]]
            topics.append(
                {
                    "topic": index + 1,
                    "keywords": keywords,
                    "title": ", ".join(keywords[:3]).title(),
                }
            )
        return topics

    def generate_local_insights(self, comments: list[str]) -> dict[str, Any]:
        sentiments = self.predict_sentiments(comments)
        total = len(sentiments)
        counts = {
            "positive": sentiments.count(1),
            "neutral": sentiments.count(0),
            "negative": sentiments.count(-1),
        }
        percentages = {
            key: round((value / total) * 100, 1) if total else 0.0
            for key, value in counts.items()
        }
        positive_comments = [comment for comment, sentiment in zip(comments, sentiments) if sentiment == 1]
        negative_comments = [comment for comment, sentiment in zip(comments, sentiments) if sentiment == -1]
        overall_topics = self.top_terms_from_comments(comments, limit=5)
        positive_themes = self.top_terms_from_comments(positive_comments, limit=3)
        negative_themes = self.top_terms_from_comments(negative_comments, limit=3)

        dominant_sentiment = max(("neutral", "positive", "negative"), key=counts.get)
        summary_lines = [
            f"Analyzed {total} comments.",
            (
                f"Overall sentiment is {dominant_sentiment} "
                f"({percentages['positive']}% positive, {percentages['neutral']}% neutral, "
                f"{percentages['negative']}% negative)."
            ),
            "Most discussed themes: "
            + (", ".join(overall_topics) if overall_topics else "not enough repeated terms to summarize."),
            "Positive themes: "
            + (", ".join(positive_themes) if positive_themes else "positive feedback is present but diffuse."),
            "Negative themes: "
            + (", ".join(negative_themes) if negative_themes else "no concentrated negative theme detected."),
        ]

        recommendations = []
        if counts["negative"] > counts["positive"]:
            recommendations.append("Address the top negative themes directly in the next video or pinned comment.")
        elif counts["positive"] > 0:
            recommendations.append("Double down on the topics driving the strongest positive reactions.")
        else:
            recommendations.append("Review a sample of comments manually; the model found no clear positive reactions.")
        if negative_themes:
            recommendations.append(
                "Investigate whether these recurring negative topics are tied to content clarity, pacing, or audience expectation gaps."
            )
        if counts["neutral"] > counts["positive"]:
            recommendations.append("Prompt viewers with clearer calls to action to convert neutral engagement into stronger sentiment.")

        summary_lines.append("Suggested actions: " + " ".join(recommendations))
        return {
            "counts": counts,
            "percentages": percentages,
            "dominant_sentiment": dominant_sentiment,
            "top_themes": overall_topics,
            "positive_themes": positive_themes,
            "negative_themes": negative_themes,
            "summary": "\n".join(summary_lines),
        }

    def render_pie_chart(self, counts: dict[str, int]) -> tuple[bytes, str | None, str | None]:
        plt, _ = _load_matplotlib()
        total = sum(counts.values())
        if total <= 0:
            return self._render_empty_chart("No sentiment results yet.", (5, 5))
        plt.figure(figsize=(5, 5))
        plt.pie(
            [counts["1"], counts["0"], counts["-1"]],
            labels=["Positive", "Neutral", "Negative"],
            autopct="%1.1f%%",
        )
        image = io.BytesIO()
        plt.savefig(image, format="png")
        image.seek(0)
        plt.close()
        return image.getvalue(), "image/png", "sentiment-chart.png"

    def render_wordcloud(self, comments: list[str]) -> tuple[bytes, str | None, str | None]:
        WordCloud = _load_wordcloud()
        text = " ".join(self.preprocess_comment(comment) for comment in comments)
        if not text.strip():
            return self._render_empty_chart("No recurring keywords found.", (8, 4))
        wordcloud = WordCloud(width=800, height=400).generate(text)
        image = io.BytesIO()
        wordcloud.to_image().save(image, format="PNG")
        image.seek(0)
        return image.getvalue(), "image/png", "wordcloud.png"

    def render_keyword_chart(self, comments: list[str]) -> tuple[bytes, str | None, str | None]:
        plt, _ = _load_matplotlib()
        processed = [self.preprocess_comment(comment) for comment in comments]
        processed = [comment for comment in processed if comment]
        if not processed:
            return self._render_empty_chart("No recurring keywords found.", (8, 4))
        tfidf = TfidfVectorizer(stop_words=sorted(self.stop_words), max_features=10)
        matrix = tfidf.fit_transform(processed)
        scores = matrix.sum(axis=0).A1
        words = tfidf.get_feature_names_out()
        keyword_scores = sorted(zip(words, scores), key=lambda item: item[1], reverse=True)[:10]
        plt.figure(figsize=(8, 4))
        plt.bar([label for label, _ in keyword_scores], [score for _, score in keyword_scores])
        plt.title("Top Keywords")
        plt.xticks(rotation=45)
        plt.tight_layout()
        image = io.BytesIO()
        plt.savefig(image, format="PNG")
        image.seek(0)
        plt.close()
        return image.getvalue(), "image/png", "keyword-chart.png"

    def render_trend_graph(self, sentiment_data: list[dict[str, Any]]) -> tuple[bytes, str | None, str | None]:
        pd = _load_pandas()
        plt, mdates = _load_matplotlib()
        dataframe = pd.DataFrame(sentiment_data)
        dataframe["timestamp"] = pd.to_datetime(dataframe["timestamp"])
        dataframe.set_index("timestamp", inplace=True)
        dataframe["sentiment"] = dataframe["sentiment"].astype(int)
        monthly_counts = dataframe.resample("ME")["sentiment"].value_counts().unstack(fill_value=0)
        monthly_totals = monthly_counts.sum(axis=1)
        monthly_percent = (monthly_counts.T / monthly_totals).T * 100
        plt.figure(figsize=(12, 6))
        colors = {-1: "red", 0: "gray", 1: "green"}
        labels = {-1: "Negative", 0: "Neutral", 1: "Positive"}
        for sentiment in [-1, 0, 1]:
            if sentiment in monthly_percent.columns:
                plt.plot(
                    monthly_percent.index,
                    monthly_percent[sentiment],
                    marker="o",
                    label=labels[sentiment],
                    color=colors[sentiment],
                )
        plt.title("Monthly Sentiment Trend")
        plt.xlabel("Month")
        plt.ylabel("Percentage")
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        image = io.BytesIO()
        plt.savefig(image, format="PNG")
        image.seek(0)
        plt.close()
        return image.getvalue(), "image/png", "trend-graph.png"

    def compute_topic_sentiment(self, comments: list[dict[str, str]]) -> dict[str, Any]:
        texts = [comment["text"] for comment in comments]
        processed = [self.preprocess_comment(text) for text in texts]
        active_indices = [index for index, text in enumerate(processed) if text]
        if not active_indices:
            return {"topics": []}
        active_texts = [processed[index] for index in active_indices]
        tfidf = TfidfVectorizer(stop_words=sorted(self.stop_words), max_features=100)
        matrix = tfidf.fit_transform(active_texts)
        cluster_count = min(3, len(set(active_texts)))
        clusters = KMeans(n_clusters=cluster_count, random_state=42, n_init=10).fit_predict(matrix)
        terms = tfidf.get_feature_names_out()
        topics: list[dict[str, Any]] = []
        predictions = self.predict_sentiments(texts)
        for cluster_index in range(cluster_count):
            local_indices = [index for index, cluster in enumerate(clusters) if cluster == cluster_index]
            if not local_indices:
                continue
            indices = [active_indices[index] for index in local_indices]
            cluster_matrix = tfidf.transform([processed[index] for index in indices])
            mean_scores = cluster_matrix.mean(axis=0).A1
            keywords = [terms[index] for index in mean_scores.argsort()[-3:][::-1]]
            sentiments = [predictions[index] for index in indices]
            positive = sentiments.count(1)
            neutral = sentiments.count(0)
            negative = sentiments.count(-1)
            dominant_sentiment = "Neutral"
            if positive >= neutral and positive >= negative:
                dominant_sentiment = "Positive"
            elif negative > positive and negative > neutral:
                dominant_sentiment = "Negative"
            topics.append(
                {
                    "topic": " ".join(keywords).title(),
                    "positive": positive,
                    "neutral": neutral,
                    "negative": negative,
                    "total": len(sentiments),
                    "dominant_sentiment": dominant_sentiment,
                }
            )
        return {"topics": topics}

    @staticmethod
    def _render_empty_chart(message: str, figsize: tuple[int, int]) -> tuple[bytes, str, str]:
        plt, _ = _load_matplotlib()
        figure, axis = plt.subplots(figsize=figsize)
        axis.axis("off")
        axis.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
        image = io.BytesIO()
        figure.savefig(image, format="png", bbox_inches="tight")
        image.seek(0)
        plt.close(figure)
        return image.getvalue(), "image/png", "empty-chart.png"


def create_job_handlers(runtime: AnalyticsRuntime) -> dict[str, Any]:
    return {
        "insights": lambda payload: ({"insights": runtime.generate_local_insights(payload["comments"])}, None, None),
        "topics": lambda payload: ({"topics": runtime.extract_topics(payload["comments"])}, None, None),
        "topic-sentiment": lambda payload: (runtime.compute_topic_sentiment(payload["comments"]), None, None),
        "wordcloud": lambda payload: runtime.render_wordcloud(payload["comments"]),
        "keyword-chart": lambda payload: runtime.render_keyword_chart(payload["comments"]),
        "trend-graph": lambda payload: runtime.render_trend_graph(payload["sentiment_data"]),
    }
