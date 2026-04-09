console.log("Comment Insights popup.js v1.0.1 loaded");
document.addEventListener("DOMContentLoaded", init);

const API_CANDIDATES = [
  "https://featured-circles-advertiser-gig.trycloudflare.com",
  "http://127.0.0.1:5000",
  "http://localhost:5000",
];
const JOB_POLL_MS = 600;
const JOB_TIMEOUT_MS = 30000;
const MAX_COMMENT_LENGTH = 1000;
const STORAGE_KEY_API_BASE_URL = "apiBaseUrl";

let comments = [];
let predictions = [];
let apiBaseUrl = API_CANDIDATES[0];

async function init() {
  try {
    setupSettingsForm();
    await hydrateSavedApiBaseUrl();
    await resolveApiBaseUrl();
    syncApiInput();

    const videoId = await getVideoId();

    if (!videoId) {
      setLoadingMessage("Open a YouTube video");
      return;
    }

    setLoadingMessage("Loading comments...");
    comments = sanitizeComments(await fetchComments());

    if (!Array.isArray(comments) || comments.length === 0) {
      setLoadingMessage("No comments found");
      return;
    }

    setLoadingMessage("Scoring sentiment...");
    predictions = normalizePredictions(await fetchPredictions(comments));

    if (predictions.length === 0) {
      setLoadingMessage("No prediction results returned");
      return;
    }

    renderStats();
    renderComments();

    setLoadingMessage("Loading analytics...");
    await loadCharts();
    await loadWordCloud();
    await loadKeywordChart();
    await loadTopics();
    await loadTopicSentiment();
    await loadInsights();

    document.getElementById("loading").style.display = "none";
  } catch (err) {
    console.error("Popup init error:", err);
    setLoadingMessage(err?.message || "Failed to load insights");
  }
}

function setLoadingMessage(message) {
  const loading = document.getElementById("loading");
  loading.style.display = "block";
  loading.innerText = message;
}

function setMutedMessage(elementId, message) {
  const element = document.getElementById(elementId);

  if (!element) {
    return;
  }

  element.innerHTML = `<div class="muted">${message}</div>`;
}

function setSettingsHelp(message, isError = false) {
  const help = document.getElementById("apiBaseUrlHelp");

  if (!help) {
    return;
  }

  help.innerText = message;
  help.style.color = isError ? "#fca5a5" : "#94a3b8";
}

function normalizeApiUrl(url) {
  if (typeof url !== "string") {
    return "";
  }

  return url.trim().replace(/\/+$/, "");
}

function normalizePredictions(data) {
  if (Array.isArray(data)) {
    return data;
  }

  if (Array.isArray(data?.predictions)) {
    return data.predictions;
  }

  return [];
}

function getPredictionList() {
  return normalizePredictions(predictions);
}

function sanitizeCommentText(text) {
  if (typeof text !== "string") {
    return "";
  }

  const normalizedText = text.replace(/\s+/g, " ").trim();

  if (!normalizedText) {
    return "";
  }

  if (normalizedText.length <= MAX_COMMENT_LENGTH) {
    return normalizedText;
  }

  return normalizedText.slice(0, MAX_COMMENT_LENGTH);
}

function sanitizeComments(commentItems) {
  if (!Array.isArray(commentItems)) {
    return [];
  }

  return commentItems
    .map((comment) => ({
      ...comment,
      text: sanitizeCommentText(comment?.text),
    }))
    .filter((comment) => comment.text);
}

function syncApiInput() {
  const input = document.getElementById("apiBaseUrlInput");

  if (!input) {
    return;
  }

  input.value = apiBaseUrl || "";
}

async function hydrateSavedApiBaseUrl() {
  if (!chrome.storage?.local) {
    return;
  }

  const stored = await chrome.storage.local.get(STORAGE_KEY_API_BASE_URL);
  const savedApiBaseUrl = normalizeApiUrl(stored?.[STORAGE_KEY_API_BASE_URL]);

  if (savedApiBaseUrl) {
    apiBaseUrl = savedApiBaseUrl;
  }
}

function setupSettingsForm() {
  const button = document.getElementById("saveApiBaseUrlButton");
  const input = document.getElementById("apiBaseUrlInput");

  if (!button || !input) {
    return;
  }

  button.addEventListener("click", async () => {
    const candidateUrl = normalizeApiUrl(input.value);

    if (!candidateUrl) {
      if (chrome.storage?.local) {
        await chrome.storage.local.remove(STORAGE_KEY_API_BASE_URL);
      }
      apiBaseUrl = API_CANDIDATES[0];
      syncApiInput();
      setSettingsHelp("Saved backend cleared. The extension will auto-detect the API again.");
      return;
    }

    if (!/^https?:\/\//i.test(candidateUrl)) {
      setSettingsHelp("Enter a full URL starting with http:// or https://", true);
      return;
    }

    try {
      const response = await fetch(`${candidateUrl}/readyz`);

      if (!response.ok) {
        throw new Error("Ready check failed");
      }

      if (chrome.storage?.local) {
        await chrome.storage.local.set({ [STORAGE_KEY_API_BASE_URL]: candidateUrl });
      }

      apiBaseUrl = candidateUrl;
      syncApiInput();
      setSettingsHelp(`Saved backend URL: ${candidateUrl}`);
    } catch (err) {
      console.error("API settings save error:", err);
      setSettingsHelp("Could not reach that backend URL. Check the tunnel and try again.", true);
    }
  });
}

async function resolveApiBaseUrl() {
  const candidates = [apiBaseUrl, ...API_CANDIDATES].filter(
    (candidate, index, array) => candidate && array.indexOf(candidate) === index,
  );

  for (const candidate of candidates) {
    try {
      const response = await fetch(`${candidate}/readyz`);

      if (response.ok) {
        apiBaseUrl = candidate;
        return apiBaseUrl;
      }
    } catch (err) {
      console.warn(`API probe failed for ${candidate}:`, err);
    }
  }

  throw new Error("Backend is not reachable. Update the API Settings URL or start the local service.");
}

async function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function waitForJob(jobPath, payload) {
  const createRes = await fetch(`${apiBaseUrl}${jobPath}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  const job = await createRes.json();

  if (!createRes.ok) {
    throw new Error(job?.error?.message || "Failed to create job");
  }

  const startedAt = Date.now();

  while (Date.now() - startedAt < JOB_TIMEOUT_MS) {
    const statusRes = await fetch(`${apiBaseUrl}${job.status_url}`);
    const statusData = await statusRes.json();

    if (statusData.status === "completed") {
      return statusData;
    }

    if (statusData.status === "failed") {
      throw new Error(statusData.error || "Background job failed");
    }

    await sleep(JOB_POLL_MS);
  }

  throw new Error("Background job timed out");
}

function getVideoId() {
  return new Promise((resolve) => {
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
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

async function fetchComments() {
  try {
    return await extractCommentsFromPage();
  } catch (err) {
    console.error("Comment fetch error:", err);
    return [];
  }
}

async function extractCommentsFromPage() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

  const results = await chrome.scripting.executeScript({
    target: { tabId: tab.id },
    func: async () => {
      const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
      const extractedComments = new Map();

      const sanitizeCommentText = (text) => {
        if (typeof text !== "string") {
          return "";
        }

        const normalizedText = text.replace(/\s+/g, " ").trim();

        if (!normalizedText) {
          return "";
        }

        if (normalizedText.length <= 1000) {
          return normalizedText;
        }

        return normalizedText.slice(0, 1000);
      };

      const collect = () => {
        const nodes = document.querySelectorAll(
          "ytd-comment-thread-renderer #content-text",
        );

        nodes.forEach((node, index) => {
          const text = sanitizeCommentText(node.innerText);

          if (!text) {
            return;
          }

          const key = `${text}-${index}`;
          extractedComments.set(key, {
            text,
            timestamp: new Date().toISOString(),
          });
        });
      };

      window.scrollTo({ top: document.body.scrollHeight * 0.35, behavior: "smooth" });
      await sleep(1200);

      for (let i = 0; i < 5; i += 1) {
        collect();
        window.scrollBy({ top: 1200, behavior: "smooth" });
        await sleep(900);
      }

      collect();
      return Array.from(extractedComments.values()).slice(0, 100);
    },
  });

  return sanitizeComments(results?.[0]?.result || []);
}

async function fetchPredictions(commentItems) {
  const response = await fetch(`${apiBaseUrl}/predict_with_timestamps`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ comments: sanitizeComments(commentItems) }),
  });

  const data = await response.json();

  if (!response.ok) {
    throw new Error(data?.error?.message || "Prediction request failed");
  }

  const normalizedPredictions = normalizePredictions(data);

  if (normalizedPredictions.length === 0) {
    throw new Error("Prediction response was not a list");
  }

  return normalizedPredictions;
}

function renderStats() {
  const predictionList = getPredictionList();

  if (predictionList.length === 0) {
    return;
  }

  let pos = 0;
  let neu = 0;
  let neg = 0;

  predictionList.forEach((prediction) => {
    if (prediction.sentiment === 1) {
      pos += 1;
    } else if (prediction.sentiment === 0) {
      neu += 1;
    } else if (prediction.sentiment === -1) {
      neg += 1;
    }
  });

  document.getElementById("positiveCount").innerText = pos;
  document.getElementById("neutralCount").innerText = neu;
  document.getElementById("negativeCount").innerText = neg;
  document.getElementById("totalCount").innerText = predictionList.length;
}

function renderComments() {
  const predictionList = getPredictionList();

  if (predictionList.length === 0) {
    return;
  }

  const list = document.getElementById("commentList");
  list.innerHTML = "";

  predictionList.slice(0, 10).forEach((comment) => {
    const li = document.createElement("li");
    li.className = "comment-item";

    let sentimentText = "Neutral";
    let sentimentClass = "neutral";

    if (comment.sentiment === 1) {
      sentimentText = "Positive";
      sentimentClass = "positive";
    } else if (comment.sentiment === -1) {
      sentimentText = "Negative";
      sentimentClass = "negative";
    }

    const commentText = typeof comment.comment === "string" ? comment.comment : "";
    const truncatedComment = commentText.length > 120 ? `${commentText.slice(0, 120)}...` : commentText;

    li.innerHTML = `
      ${truncatedComment}
      <div class="sentiment ${sentimentClass}">
        ${sentimentText}
      </div>
    `;

    list.appendChild(li);
  });
}

function formatTopicLabel(topic) {
  if (topic.title) {
    return topic.title;
  }

  if (Array.isArray(topic.keywords) && topic.keywords.length > 0) {
    return topic.keywords.slice(0, 3).join(", ");
  }

  return `Topic ${topic.topic ?? ""}`.trim();
}

async function loadCharts() {
  const predictionList = getPredictionList();

  if (predictionList.length === 0) {
    return;
  }

  const counts = { "1": 0, "0": 0, "-1": 0 };

  predictionList.forEach((prediction) => {
    const key = String(prediction.sentiment);
    if (counts[key] !== undefined) {
      counts[key] += 1;
    }
  });

  try {
    const response = await fetch(`${apiBaseUrl}/generate_chart`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sentiment_counts: counts }),
    });

    if (!response.ok) {
      throw new Error("Chart request failed");
    }

    const blob = await response.blob();
    document.getElementById("sentimentChart").src = URL.createObjectURL(blob);
  } catch (err) {
    console.error("Chart error:", err);
  }
}

async function loadTopics() {
  try {
    const data = await waitForJob("/jobs/topics", {
      comments: comments.map((comment) => comment.text),
    });

    const list = document.getElementById("topicList");
    list.innerHTML = "";

    if (!data.result?.topics || data.result.topics.length === 0) {
      setMutedMessage("topicList", "No discussion topics found yet.");
      return;
    }

    data.result.topics.forEach((topic) => {
      const li = document.createElement("li");
      li.className = "topic-pill";
      li.innerText = formatTopicLabel(topic);
      list.appendChild(li);
    });
  } catch (err) {
    console.error("Topic error:", err);
    setMutedMessage("topicList", "Failed to load discussion topics.");
  }
}

async function loadInsights() {
  try {
    const data = await waitForJob("/jobs/insights", {
      comments: comments.map((comment) => comment.text),
    });

    document.getElementById("aiInsights").innerText =
      data.result?.insights?.summary || "No insights generated.";
  } catch (err) {
    console.error("Insights error:", err);
    document.getElementById("aiInsights").innerText = "Failed to generate insights.";
  }
}

async function loadWordCloud() {
  try {
    const data = await waitForJob("/jobs/wordcloud", {
      comments: comments.map((comment) => comment.text),
    });

    document.getElementById("wordCloud").src = `${apiBaseUrl}${data.artifact_url}`;
  } catch (err) {
    console.error("WordCloud error:", err);
  }
}

async function loadKeywordChart() {
  try {
    const data = await waitForJob("/jobs/keyword-chart", {
      comments: comments.map((comment) => comment.text),
    });

    document.getElementById("keywordChart").src = `${apiBaseUrl}${data.artifact_url}`;
  } catch (err) {
    console.error("Keyword chart error:", err);
  }
}

async function loadTopicSentiment() {
  try {
    const data = await waitForJob("/jobs/topic-sentiment", {
      comments,
    });

    const list = document.getElementById("topicSentimentList");
    list.innerHTML = "";

    if (!data.result?.topics || data.result.topics.length === 0) {
      setMutedMessage("topicSentimentList", "No topic sentiment breakdown available yet.");
      return;
    }

    data.result.topics.forEach((topic) => {
      const li = document.createElement("li");
      li.className = "topic-sentiment-item";
      li.innerHTML = `
        <strong>${topic.topic}</strong> - ${topic.dominant_sentiment}
        <div class="topic-sentiment-meta">
          Positive ${topic.positive} - Neutral ${topic.neutral} - Negative ${topic.negative}
        </div>
      `;
      list.appendChild(li);
    });
  } catch (err) {
    console.error("Topic sentiment error:", err);
    setMutedMessage("topicSentimentList", "Failed to load topic sentiment breakdown.");
  }
}
