console.log("Comment Insights popup.js v1.0.4 loaded");
document.addEventListener("DOMContentLoaded", init);

const API_CANDIDATES = [
  "https://commentpulse-api.onrender.com",
  "http://127.0.0.1:5000",
  "http://localhost:5000",
];
const JOB_POLL_MS = 1000;
const JOB_TIMEOUT_MS = 120000;
const READY_TIMEOUT_MS = 90000;
const REQUEST_TIMEOUT_MS = 20000;
const MAX_COMMENT_LENGTH = 1000;
const STORAGE_KEY_API_BASE_URL = "apiBaseUrl";

let comments = [];
let predictions = [];
let apiBaseUrl = API_CANDIDATES[0];
let analyticsBusy = false;
let settingsFormBound = false;

async function init() {
  document.getElementById("retryAnalysisButton").hidden = true;
  setRetryVisible(false);
  try {
    setupSettingsForm();
    await hydrateSavedApiBaseUrl();
    setLoadingMessage("Connecting to CommentPulse. The free server may need up to a minute to wake up...");
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

    await loadAnalytics();
    document.getElementById("retryAnalysisButton").hidden = true;
  } catch (err) {
    console.error("Popup init error:", err);
    setLoadingMessage(err?.message || "Failed to load insights");
    document.getElementById("retryAnalysisButton").hidden = false;
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

  const notice = document.createElement("div");
  notice.className = "muted";
  notice.textContent = message;
  element.replaceChildren(notice);
}

function setImageMessage(elementId, message) {
  const image = document.getElementById(elementId);
  if (!image) return;
  let status = document.getElementById(`${elementId}Status`);
  if (!status) {
    status = document.createElement("div");
    status.id = `${elementId}Status`;
    status.className = "muted";
    status.setAttribute("role", "status");
    image.insertAdjacentElement("afterend", status);
  }
  status.textContent = message;
  status.hidden = !message;
  image.alt = message;
  if (message) image.removeAttribute("src");
}

function setRetryVisible(visible) {
  const button = document.getElementById("retryAnalyticsButton");
  if (button) button.hidden = !visible;
}

async function fetchWithTimeout(url, options = {}, timeoutMs = REQUEST_TIMEOUT_MS) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { ...options, signal: controller.signal });
  } catch (error) {
    if (error?.name === "AbortError") {
      throw new Error("The server took too long to respond. Please retry in a moment.");
    }
    throw error;
  } finally {
    clearTimeout(timeout);
  }
}

async function readJsonResponse(response, fallbackMessage) {
  try {
    return await response.json();
  } catch {
    throw new Error(fallbackMessage);
  }
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
  const rows = Array.isArray(data) ? data : data?.predictions;
  if (!Array.isArray(rows)) return [];

  return rows.map((row, index) => {
    const sentiment = Number(row?.sentiment);
    if (!Number.isInteger(sentiment) || ![-1, 0, 1].includes(sentiment)) {
      throw new Error(`The analysis service returned an invalid sentiment for comment ${index + 1}. Please retry.`);
    }
    return { ...row, sentiment };
  });
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
  if (settingsFormBound) return;
  const button = document.getElementById("saveApiBaseUrlButton");
  const input = document.getElementById("apiBaseUrlInput");

  if (!button || !input) {
    return;
  }
  settingsFormBound = true;

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
      const response = await fetchWithTimeout(`${candidateUrl}/readyz`, {}, READY_TIMEOUT_MS);

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
      const response = await fetchWithTimeout(`${candidate}/readyz`, {}, READY_TIMEOUT_MS);

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
  const createRes = await fetchWithTimeout(`${apiBaseUrl}${jobPath}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  }, READY_TIMEOUT_MS);

  const job = await readJsonResponse(createRes, "The server returned an unreadable response. Please retry.");

  if (!createRes.ok) {
    throw new Error(job?.error?.message || "Failed to create job");
  }
  if (typeof job.status_url !== "string" || !job.status_url.startsWith("/jobs/")) {
    throw new Error("The server returned an invalid job status link.");
  }

  const startedAt = Date.now();

  while (Date.now() - startedAt < JOB_TIMEOUT_MS) {
    const statusRes = await fetchWithTimeout(`${apiBaseUrl}${job.status_url}`);
    const statusData = await readJsonResponse(statusRes, "Could not read the analysis status. Please retry.");
    if (!statusRes.ok) {
      throw new Error(statusData?.error?.message || "Could not check the analysis status.");
    }

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
    throw new Error("Could not read comments from this YouTube tab. Reload the video and try again.");
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
  const response = await fetchWithTimeout(`${apiBaseUrl}/predict_with_timestamps`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ comments: sanitizeComments(commentItems) }),
  }, READY_TIMEOUT_MS);

  const data = await readJsonResponse(response, "The server returned an unreadable analysis response.");

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

  const counts = { positive: 0, neutral: 0, negative: 0 };

  predictionList.forEach((prediction) => {
    if (prediction.sentiment === 1) {
      counts.positive += 1;
    } else if (prediction.sentiment === 0) {
      counts.neutral += 1;
    } else {
      counts.negative += 1;
    }
  });

  document.getElementById("positiveCount").innerText = counts.positive;
  document.getElementById("neutralCount").innerText = counts.neutral;
  document.getElementById("negativeCount").innerText = counts.negative;
  document.getElementById("totalCount").innerText = Object.values(counts).reduce((sum, count) => sum + count, 0);
}

function renderComments() {
  const predictionList = getPredictionList();

  if (predictionList.length === 0) {
    return;
  }

  const list = document.getElementById("commentList");
  list.replaceChildren();

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

    const text = document.createElement("div");
    text.className = "comment-text";
    text.textContent = truncatedComment;
    const sentimentLabel = document.createElement("div");
    sentimentLabel.className = `sentiment ${sentimentClass}`;
    sentimentLabel.textContent = sentimentText;
    li.append(text, sentimentLabel);

    list.appendChild(li);
  });
}

function formatTopicLabel(topic) {
  if (topic.title) {
    return String(topic.title);
  }

  if (Array.isArray(topic.keywords) && topic.keywords.length > 0) {
    return topic.keywords.slice(0, 3).map(String).join(", ");
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
    const response = await fetchWithTimeout(`${apiBaseUrl}/generate_chart`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sentiment_counts: counts }),
    });

    if (!response.ok) {
      throw new Error("Chart request failed");
    }

    const blob = await response.blob();
    const chart = document.getElementById("sentimentChart");
    if (chart.src.startsWith("blob:")) URL.revokeObjectURL(chart.src);
    chart.src = URL.createObjectURL(blob);
    setImageMessage("sentimentChart", "");
    return true;
  } catch (err) {
    console.error("Chart error:", err);
    setImageMessage("sentimentChart", "Sentiment chart unavailable. Use Retry failed insights to try again.");
    return false;
  }
}

async function loadTopics() {
  try {
    const data = await waitForJob("/jobs/topics", {
      comments: comments.map((comment) => comment.text),
    });

    const list = document.getElementById("topicList");
    list.replaceChildren();

    if (!data.result?.topics || data.result.topics.length === 0) {
      setMutedMessage("topicList", "No discussion topics found yet.");
      return true;
    }

    data.result.topics.forEach((topic) => {
      const li = document.createElement("li");
      li.className = "topic-pill";
      li.innerText = formatTopicLabel(topic);
      list.appendChild(li);
    });
    return true;
  } catch (err) {
    console.error("Topic error:", err);
    setMutedMessage("topicList", "Topics unavailable. Use Retry failed insights to try again.");
    return false;
  }
}

async function loadInsights() {
  try {
    const data = await waitForJob("/jobs/insights", {
      comments: comments.map((comment) => comment.text),
    });

    document.getElementById("aiInsights").textContent =
      data.result?.insights?.summary || "No insights generated.";
    return true;
  } catch (err) {
    console.error("Insights error:", err);
    document.getElementById("aiInsights").textContent = "Summary unavailable. Use Retry failed insights to try again.";
    return false;
  }
}

async function loadWordCloud() {
  try {
    const data = await waitForJob("/jobs/wordcloud", {
      comments: comments.map((comment) => comment.text),
    });

    document.getElementById("wordCloud").src = getApiArtifactUrl(data.artifact_url);
    setImageMessage("wordCloud", "");
    return true;
  } catch (err) {
    console.error("WordCloud error:", err);
    setImageMessage("wordCloud", "Word cloud unavailable. Use Retry failed insights to try again.");
    return false;
  }
}

async function loadKeywordChart() {
  try {
    const data = await waitForJob("/jobs/keyword-chart", {
      comments: comments.map((comment) => comment.text),
    });

    document.getElementById("keywordChart").src = getApiArtifactUrl(data.artifact_url);
    setImageMessage("keywordChart", "");
    return true;
  } catch (err) {
    console.error("Keyword chart error:", err);
    setImageMessage("keywordChart", "Keyword chart unavailable. Use Retry failed insights to try again.");
    return false;
  }
}

async function loadTopicSentiment() {
  try {
    const data = await waitForJob("/jobs/topic-sentiment", {
      comments,
    });

    const list = document.getElementById("topicSentimentList");
    list.replaceChildren();

    if (!data.result?.topics || data.result.topics.length === 0) {
      setMutedMessage("topicSentimentList", "No topic sentiment breakdown available yet.");
      return true;
    }

    data.result.topics.forEach((topic) => {
      const li = document.createElement("li");
      li.className = "topic-sentiment-item";
      const label = document.createElement("strong");
      label.textContent = `${String(topic.topic ?? "Topic")} - ${String(topic.dominant_sentiment ?? "Unknown")}`;
      const meta = document.createElement("div");
      meta.className = "topic-sentiment-meta";
      meta.textContent = `Positive ${Number(topic.positive) || 0} - Neutral ${Number(topic.neutral) || 0} - Negative ${Number(topic.negative) || 0}`;
      li.append(label, meta);
      list.appendChild(li);
    });
    return true;
  } catch (err) {
    console.error("Topic sentiment error:", err);
    setMutedMessage("topicSentimentList", "Topic sentiment unavailable. Use Retry failed insights to try again.");
    return false;
  }
}

function getApiArtifactUrl(path) {
  if (typeof path !== "string" || !path.startsWith("/jobs/")) {
    throw new Error("The server returned an invalid chart link.");
  }
  const base = new URL(apiBaseUrl);
  const artifact = new URL(path, base);
  if (artifact.origin !== base.origin) {
    throw new Error("The server returned an invalid chart link.");
  }
  return artifact.href;
}

async function loadAnalytics() {
  if (analyticsBusy || comments.length === 0) return;
  analyticsBusy = true;
  setRetryVisible(false);
  setLoadingMessage("Generating charts and insights. This can take a little while on the free server...");

  try {
    const results = await Promise.all([
      loadCharts(),
      loadWordCloud(),
      loadKeywordChart(),
      loadTopics(),
      loadTopicSentiment(),
      loadInsights(),
    ]);
    const hasFailures = results.some((success) => !success);
    document.getElementById("loading").style.display = hasFailures ? "block" : "none";
    if (hasFailures) {
      setLoadingMessage("Some insights could not load. Retry the failed insights below.");
      setRetryVisible(true);
    }
  } finally {
    analyticsBusy = false;
  }
}

document.addEventListener("DOMContentLoaded", () => {
  document.getElementById("retryAnalyticsButton")?.addEventListener("click", loadAnalytics);
  document.getElementById("retryAnalysisButton")?.addEventListener("click", init);
});
