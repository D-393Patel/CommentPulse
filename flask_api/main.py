import os
import io
import re
import pickle
import requests
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from dotenv import load_dotenv
from wordcloud import WordCloud

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

from keybert import KeyBERT


# ------------------------------------------------
# Setup
# ------------------------------------------------

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)

load_dotenv(os.path.join(root_dir, ".env"))

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

app = Flask(__name__)
CORS(app)


# ------------------------------------------------
# API KEYS
# ------------------------------------------------

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"


# ------------------------------------------------
# Load ML Model
# ------------------------------------------------

model = None
vectorizer = None

try:

    model_path = os.path.join(root_dir, "lgbm_model.pkl")
    vec_path = os.path.join(root_dir, "tfidf_vectorizer.pkl")

    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            model = pickle.load(f)

    if os.path.exists(vec_path):
        with open(vec_path, "rb") as f:
            vectorizer = pickle.load(f)

    print("Sentiment model loaded")

except Exception as e:
    print("MODEL LOAD ERROR:", e)


# ------------------------------------------------
# Semantic Keyword Model
# ------------------------------------------------

try:

    kw_model = KeyBERT()

    print("KeyBERT model loaded")

except Exception as e:

    print("KeyBERT load error:", e)
    kw_model = None


# ------------------------------------------------
# NLP Setup
# ------------------------------------------------

STOP_WORDS = set(stopwords.words("english")) - {"not", "no", "but"}
LEMMATIZER = WordNetLemmatizer()


def preprocess_comment(text):

    text = str(text).lower()
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"[^a-z0-9 ]", "", text)

    words = text.split()
    words = [w for w in words if w not in STOP_WORDS]
    words = [LEMMATIZER.lemmatize(w) for w in words]

    return " ".join(words)


# ------------------------------------------------
# Topic Extraction
# ------------------------------------------------

def extract_topics(comments):

    try:

        if not comments:
            return []

        text = " ".join(comments)

        # Semantic keyword extraction
        if kw_model:

            keywords = kw_model.extract_keywords(
                text,
                keyphrase_ngram_range=(1, 2),
                stop_words="english",
                top_n=5
            )

            topics = []

            for i, (kw, score) in enumerate(keywords):

                topics.append({
                    "topic": i + 1,
                    "title": kw.title(),
                    "score": round(float(score), 3)
                })

            return topics

        # Fallback to TF-IDF clustering
        processed = [preprocess_comment(c) for c in comments if c.strip()]

        tfidf = TfidfVectorizer(
            stop_words="english",
            max_features=50,
            ngram_range=(1, 2)
        )

        X = tfidf.fit_transform(processed)

        n_clusters = min(3, len(processed))

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(X)

        terms = tfidf.get_feature_names_out()

        topics = []

        for i, center in enumerate(kmeans.cluster_centers_):

            words = [terms[j] for j in center.argsort()[-5:]]

            topics.append({
                "topic": i + 1,
                "keywords": words
            })

        return topics

    except Exception as e:

        print("Topic extraction error:", e)

        return [{
            "topic": 1,
            "title": "AI Discussion"
        }]


# ------------------------------------------------
# Gemini AI Insights
# ------------------------------------------------

def generate_ai_insights(comments):

    if not GEMINI_API_KEY:
        return "Gemini API key not configured."

    sample = "\n".join(comments[:40])

    prompt = f"""
Analyze these YouTube comments.

Provide:
1. Overall sentiment summary
2. Top positive themes
3. Top negative themes
4. Suggestions for the creator

Comments:
{sample}
"""

    payload = {
        "contents": [
            {"parts": [{"text": prompt}]}
        ]
    }

    try:

        response = requests.post(GEMINI_URL, json=payload, timeout=60)

        data = response.json()

        return data["candidates"][0]["content"]["parts"][0]["text"]

    except Exception as e:

        print("Gemini error:", e)

        return "AI insights unavailable"


# ------------------------------------------------
# Root
# ------------------------------------------------

@app.route("/")
def home():
    return jsonify({
        "message": "YouTube Sentiment API Running"
    })


# ------------------------------------------------
# Health
# ------------------------------------------------

@app.route("/health")
def health():
    return jsonify({"status": "ok"})


# ------------------------------------------------
# Fetch YouTube Comments
# ------------------------------------------------

@app.route("/get_youtube_comments")
def get_comments():

    video_id = request.args.get("videoId")

    if not video_id:
        return jsonify({"comments": []})

    comments = []

    url = "https://www.googleapis.com/youtube/v3/commentThreads"

    params = {
        "part": "snippet",
        "videoId": video_id,
        "maxResults": 100,
        "textFormat": "plainText",
        "key": YOUTUBE_API_KEY
    }

    try:

        r = requests.get(url, params=params)
        data = r.json()

        for item in data.get("items", []):

            snippet = item["snippet"]["topLevelComment"]["snippet"]

            comments.append({
                "text": snippet["textOriginal"],
                "timestamp": snippet["publishedAt"]
            })

    except Exception as e:
        print("YouTube API error:", e)

    return jsonify({"comments": comments})


# ------------------------------------------------
# Sentiment Prediction
# ------------------------------------------------

@app.route("/predict_with_timestamps", methods=["POST"])
def predict():

    data = request.json
    comments = data.get("comments", [])

    if not comments or model is None or vectorizer is None:
        return jsonify([])

    texts = [c["text"] for c in comments]
    timestamps = [c["timestamp"] for c in comments]

    processed = [preprocess_comment(t) for t in texts]
    X = vectorizer.transform(processed)

    preds = model.predict(X)

    results = []

    for t, s, ts in zip(texts, preds, timestamps):

        results.append({
            "comment": t,
            "sentiment": int(s),
            "timestamp": ts
        })

    return jsonify(results)


# ------------------------------------------------
# Sentiment Pie Chart
# ------------------------------------------------

@app.route("/generate_chart", methods=["POST"])
def chart():

    data = request.json
    counts = data.get("sentiment_counts", {})

    labels = ["Positive", "Neutral", "Negative"]

    sizes = [
        counts.get("1", 0),
        counts.get("0", 0),
        counts.get("-1", 0)
    ]

    plt.figure(figsize=(5, 5))
    plt.pie(sizes, labels=labels, autopct="%1.1f%%")

    img = io.BytesIO()
    plt.savefig(img, format="png")
    img.seek(0)
    plt.close()

    return send_file(img, mimetype="image/png")


# ------------------------------------------------
# Wordcloud
# ------------------------------------------------

@app.route("/generate_wordcloud", methods=["POST"])
def wordcloud():

    data = request.json
    comments = data.get("comments", [])

    processed = [preprocess_comment(c) for c in comments]

    text = " ".join(processed)

    wc = WordCloud(width=800, height=400).generate(text)

    img = io.BytesIO()
    wc.to_image().save(img, format="PNG")

    img.seek(0)

    return send_file(img, mimetype="image/png")


# ------------------------------------------------
# Topic Extraction API
# ------------------------------------------------

@app.route("/extract_topics", methods=["POST"])
def topics():

    data = request.json
    comments = data.get("comments", [])

    topics = extract_topics(comments)

    return jsonify({"topics": topics})


# ------------------------------------------------
# Gemini Insights
# ------------------------------------------------

@app.route("/generate_insights", methods=["POST"])
def insights():

    data = request.json
    comments = data.get("comments", [])

    insights = generate_ai_insights(comments)

    return jsonify({"insights": insights})


# ------------------------------------------------
# Sentiment Trend Graph
# ------------------------------------------------

@app.route("/generate_trend_graph", methods=["POST"])
def trend_graph():

    data = request.json
    sentiment_data = data.get("sentiment_data", [])

    if not sentiment_data:
        return jsonify({"error": "No data provided"}), 400

    df = pd.DataFrame(sentiment_data)

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)
    df["sentiment"] = df["sentiment"].astype(int)

    monthly_counts = df.resample("M")["sentiment"].value_counts().unstack(fill_value=0)

    monthly_totals = monthly_counts.sum(axis=1)

    monthly_percent = (monthly_counts.T / monthly_totals).T * 100

    plt.figure(figsize=(12, 6))

    colors = {-1: "red", 0: "gray", 1: "green"}
    labels = {-1: "Negative", 0: "Neutral", 1: "Positive"}

    for s in [-1, 0, 1]:

        if s in monthly_percent.columns:

            plt.plot(
                monthly_percent.index,
                monthly_percent[s],
                marker="o",
                label=labels[s],
                color=colors[s]
            )

    plt.title("Monthly Sentiment Trend")
    plt.xlabel("Month")
    plt.ylabel("Percentage")

    plt.legend()
    plt.grid(True)

    plt.xticks(rotation=45)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    img = io.BytesIO()

    plt.savefig(img, format="PNG")

    img.seek(0)
    plt.close()

    return send_file(img, mimetype="image/png")

@app.route("/generate_keyword_chart", methods=["POST"])
def generate_keyword_chart():

    data = request.json
    comments = data.get("comments", [])

    if not comments:
        return jsonify({"error": "No comments provided"}), 400

    processed = [preprocess_comment(c) for c in comments]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=10
    )

    X = vectorizer.fit_transform(processed)

    scores = X.sum(axis=0).A1
    words = vectorizer.get_feature_names_out()

    keyword_scores = sorted(
        zip(words, scores),
        key=lambda x: x[1],
        reverse=True
    )[:10]

    labels = [k[0] for k in keyword_scores]
    values = [k[1] for k in keyword_scores]

    plt.figure(figsize=(8,4))
    plt.bar(labels, values)

    plt.title("Top Keywords")
    plt.xticks(rotation=45)

    img = io.BytesIO()
    plt.tight_layout()
    plt.savefig(img, format="PNG")
    img.seek(0)
    plt.close()

    return send_file(img, mimetype="image/png")

@app.route("/topic_sentiment", methods=["POST"])
def topic_sentiment():

    data = request.json
    comments = data.get("comments", [])

    if not comments:
        return jsonify({"topics": []})

    texts = [c["text"] for c in comments]

    processed = [preprocess_comment(t) for t in texts]

    tfidf = TfidfVectorizer(
        stop_words="english",
        max_features=100
    )

    X = tfidf.fit_transform(processed)

    n_clusters = min(3, len(processed))

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    clusters = kmeans.fit_predict(X)

    terms = tfidf.get_feature_names_out()

    topics = []

    for i in range(n_clusters):

        idx = [j for j, c in enumerate(clusters) if c == i]

        if not idx:
            continue

        cluster_comments = [texts[j] for j in idx]

        cluster_processed = [processed[j] for j in idx]

        X_cluster = tfidf.transform(cluster_processed)

        mean_scores = X_cluster.mean(axis=0).A1

        keywords = [
            terms[j] for j in mean_scores.argsort()[-3:]
        ]

        topic_name = " ".join(keywords)

        sentiments = []

        for j in idx:

            vec = vectorizer.transform(
                [processed[j]]
            )

            pred = model.predict(vec)[0]

            sentiments.append(pred)

        pos = sentiments.count(1)
        neu = sentiments.count(0)
        neg = sentiments.count(-1)

        total = len(sentiments)

        sentiment_label = "Neutral"

        if pos >= neu and pos >= neg:
            sentiment_label = "Positive"

        if neg > pos and neg > neu:
            sentiment_label = "Negative"

        topics.append({
            "topic": topic_name.title(),
            "positive": pos,
            "neutral": neu,
            "negative": neg,
            "total": total,
            "dominant_sentiment": sentiment_label
        })

    return jsonify({"topics": topics})
# ------------------------------------------------
# Run
# ------------------------------------------------

if __name__ == "__main__":

    import os

    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("DEBUG", "false").lower() == "true"

    app.run(host="0.0.0.0", port=port, debug=debug)