document.addEventListener("DOMContentLoaded", init);

const API = "http://localhost:5000";

let comments = [];
let predictions = [];

async function init() {

  const videoId = await getVideoId();

  if (!videoId) {
    document.getElementById("loading").innerText =
      "Open a YouTube video";
    return;
  }

  document.getElementById("loading").innerText =
    "Loading comments...";

  comments = await fetchComments(videoId);

  if (!comments.length) {
    document.getElementById("loading").innerText =
      "No comments found";
    return;
  }

  predictions = await fetchPredictions(comments);

  renderStats();
  renderComments();

  await loadCharts();

  await loadWordCloud();   // ⭐ ADD THIS
  await loadKeywordChart();
  await loadTopics();
  await loadTopicSentiment();
  await loadInsights();

  document.getElementById("loading").style.display = "none";
}


function getVideoId() {

  return new Promise(resolve => {

    chrome.tabs.query({ active: true, currentWindow: true }, tabs => {

      const url = tabs[0]?.url || "";
      resolve(extractVideoId(url));

    });

  });

}


function extractVideoId(url) {

  const regExp =
    /(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/shorts\/)([^&?/]+)/;

  const match = url.match(regExp);

  return match ? match[1] : null;

}


async function fetchComments(videoId) {

  try {

    const res = await fetch(
      `${API}/get_youtube_comments?videoId=${videoId}`
    );

    const data = await res.json();

    return data.comments || [];

  } catch (err) {

    console.error("Comment fetch error:", err);

    return [];

  }

}


async function fetchPredictions(comments) {

  try {

    const res = await fetch(`${API}/predict_with_timestamps`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ comments })
    });

    return await res.json();

  } catch (err) {

    console.error("Prediction error:", err);

    return [];

  }

}


function renderStats() {

  let pos = 0;
  let neu = 0;
  let neg = 0;

  predictions.forEach(p => {

    if (p.sentiment === 1) pos++;
    else if (p.sentiment === 0) neu++;
    else if (p.sentiment === -1) neg++;

  });

  document.getElementById("positiveCount").innerText = pos;
  document.getElementById("neutralCount").innerText = neu;
  document.getElementById("negativeCount").innerText = neg;
  document.getElementById("totalCount").innerText = predictions.length;

}


function renderComments() {

  const list = document.getElementById("commentList");

  list.innerHTML = "";

  predictions.slice(0, 10).forEach(c => {

    const li = document.createElement("li");

    let sentimentText = "Neutral";
    let sentimentClass = "neutral";

    if (c.sentiment === 1) {
      sentimentText = "Positive";
      sentimentClass = "positive";
    }

    if (c.sentiment === -1) {
      sentimentText = "Negative";
      sentimentClass = "negative";
    }

    li.innerHTML = `
      ${c.comment.substring(0, 120)}...
      <div class="sentiment ${sentimentClass}">
        ${sentimentText}
      </div>
    `;

    list.appendChild(li);

  });

}


async function loadCharts() {

  const counts = { "1": 0, "0": 0, "-1": 0 };

  predictions.forEach(p => {

    const key = String(p.sentiment);

    if (counts[key] !== undefined) counts[key]++;

  });

  try {

    const res = await fetch(`${API}/generate_chart`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sentiment_counts: counts })
    });

    const blob = await res.blob();

    document.getElementById("sentimentChart").src =
      URL.createObjectURL(blob);

  } catch (err) {

    console.error("Chart error:", err);

  }

}


async function loadTopics() {

  try {

    const res = await fetch(`${API}/extract_topics`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        comments: comments.map(c => c.text)
      })
    });

    const data = await res.json();

    const list = document.getElementById("topicList");

    list.innerHTML = "";

    if (!data.topics || data.topics.length === 0) {

      list.innerHTML = "<li>No discussion topics found</li>";
      return;

    }

    data.topics.forEach(t => {

      const li = document.createElement("li");

      // Semantic keyword result
      if (t.title) {

        li.innerHTML = `
          <strong>${t.title}</strong>
        `;

      }

      // TF-IDF fallback result
      else if (t.keywords) {

        li.innerHTML = `
          <strong>Topic ${t.topic}</strong><br>
          ${t.keywords.join(", ")}
        `;

      }

      list.appendChild(li);

    });

  } catch (err) {

    console.error("Topic error:", err);

  }

}




async function loadInsights() {

  try {

    const res = await fetch(`${API}/generate_insights`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        comments: comments.map(c => c.text)
      })
    });

    const data = await res.json();

    document.getElementById("aiInsights").innerText =
      data.insights || "No insights generated.";

  } catch (err) {

    console.error("Insights error:", err);

    document.getElementById("aiInsights").innerText =
      "Failed to generate insights.";

  }

}


async function loadWordCloud() {

  try {

    const res = await fetch(`${API}/generate_wordcloud`, {

      method: "POST",

      headers: {
        "Content-Type": "application/json"
      },

      body: JSON.stringify({
        comments: comments.map(c => c.text)
      })

    });

    const blob = await res.blob();

    document.getElementById("wordCloud").src =
      URL.createObjectURL(blob);

  } catch (err) {

    console.error("WordCloud error:", err);

  }

}

async function loadKeywordChart() {

  try {

    const res = await fetch(`${API}/generate_keyword_chart`, {

      method: "POST",

      headers: {
        "Content-Type": "application/json"
      },

      body: JSON.stringify({
        comments: comments.map(c => c.text)
      })

    });

    const blob = await res.blob();

    document.getElementById("keywordChart").src =
      URL.createObjectURL(blob);

  } catch (err) {

    console.error("Keyword chart error:", err);

  }

}


async function loadTopicSentiment() {

  try {

    const res = await fetch(
      `${API}/topic_sentiment`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          comments: comments
        })
      }
    );

    const data = await res.json();

    const list =
      document.getElementById("topicSentimentList");

    list.innerHTML = "";

    if (!data.topics || data.topics.length === 0) {
      list.innerHTML = "<li>No topic data</li>";
      return;
    }

    data.topics.forEach(t => {

      const li = document.createElement("li");

      li.innerHTML = `
        <strong>${t.topic}</strong> → 
        ${t.dominant_sentiment}
        <br>
        👍 ${t.positive} 
        😐 ${t.neutral} 
        👎 ${t.negative}
      `;

      list.appendChild(li);

    });

  } catch (err) {

    console.error("Topic sentiment error:", err);

  }

}