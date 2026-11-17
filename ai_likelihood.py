"""Explainable, text-based AI-content likelihood heuristics.

This module intentionally does not try to prove whether a video was made with
AI. It produces a conservative estimate from public page text and comments so
the product can explain every signal it shows to a viewer.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any


AI_MENTION_PATTERN = re.compile(
    r"\b(?:ai[ -]?(?:generated|made|voice|content|video)|"
    r"artificial intelligence|chatgpt|deepfake|synthetic voice|robot voice|"
    r"text[ -]to[ -]speech|tts)\b",
    re.IGNORECASE,
)
GENERIC_PHRASE_PATTERN = re.compile(
    r"\b(?:must watch|you won't believe|game changer|unlock the secret|"
    r"revolutionary|in this video we will|let's dive in)\b",
    re.IGNORECASE,
)
WORD_PATTERN = re.compile(r"[a-zA-Z][a-zA-Z']+")


def _clean_text(value: Any, limit: int = 5000) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.split())[:limit]


def _repetition_ratio(text: str) -> float:
    words = [word.lower() for word in WORD_PATTERN.findall(text)]
    if len(words) < 12:
        return 0.0
    bigrams = [f"{words[index]} {words[index + 1]}" for index in range(len(words) - 1)]
    repeated = sum(count - 1 for count in Counter(bigrams).values() if count > 1)
    return repeated / max(1, len(bigrams))


def analyze_ai_likelihood(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a bounded estimate with reasons, never a factual determination."""
    title = _clean_text(payload.get("title"), 300)
    description = _clean_text(payload.get("description"), 5000)
    channel_name = _clean_text(payload.get("channel_name"), 300)
    comments = [_clean_text(item, 1000) for item in payload.get("comments", [])]
    comments = [comment for comment in comments if comment]
    page_text = " ".join(part for part in (title, description) if part)
    score = 0
    reasons: list[str] = []
    evidence: list[str] = []
    explicit_mentions = [comment for comment in comments if AI_MENTION_PATTERN.search(comment)]
    if explicit_mentions:
        score += min(35, 12 + len(explicit_mentions) * 6)
        reasons.append(f"{len(explicit_mentions)} viewer comment(s) explicitly mention AI or synthetic media.")
        evidence.extend(explicit_mentions[:2])
    if AI_MENTION_PATTERN.search(page_text):
        score += 15
        reasons.append("The title or description refers to AI, synthetic media, or text-to-speech.")
    generic_matches = GENERIC_PHRASE_PATTERN.findall(page_text)
    if generic_matches:
        score += min(10, len(generic_matches) * 3)
        reasons.append("The public metadata contains highly templated promotional phrasing.")
    repetition = _repetition_ratio(description)
    if repetition >= 0.10:
        score += 12
        reasons.append("The description has repeated phrase patterns; this can occur in templated content.")
    elif repetition >= 0.05:
        score += 6
        reasons.append("The description contains some repeated phrase patterns.")
    available_sources = sum(bool(value) for value in (title, description, channel_name)) + bool(comments)
    if available_sources <= 1:
        confidence = "Low"
        reasons.append("Limited public information was available, so this is a low-confidence estimate.")
    elif len(comments) < 5:
        confidence = "Low"
        reasons.append("Only a small number of public comments were available.")
    elif title and description:
        confidence = "Medium"
    else:
        confidence = "Low"
    score = min(100, score)
    likelihood = "High" if score >= 45 else "Medium" if score >= 20 else "Low"
    if not reasons:
        reasons.append("No strong public text or comment signals associated with AI-generated media were found.")
    return {
        "likelihood": likelihood,
        "score": score,
        "confidence": confidence,
        "reasons": reasons,
        "evidence": evidence,
        "disclaimer": "Estimate based on public text and comments only. It is not proof that a video is AI-generated.",
        "sources": {"title": bool(title), "description": bool(description), "channel_name": bool(channel_name), "comments_analyzed": len(comments)},
    }
