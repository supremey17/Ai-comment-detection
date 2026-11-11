"""Prompt construction and output cleanup for AI comment generation.

Each human tweet gets one AI counterpart written about the same subject at a
similar length. Which instruction, persona and writing style is used varies
per row (chosen deterministically from config), so the detector sees many
kinds of AI text instead of one model's default voice.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Optional

SYSTEM_PROMPT = (
    "You write short social media posts for a research dataset used to train "
    "detectors of AI-written text. Follow the instructions exactly. Reply with "
    "only the post itself: no quotation marks, no preamble, no explanation, no "
    "hashtags unless asked, and no emoji."
)

MODE_TEMPLATES = {
    # A new, independent post on the same subject as the human tweet.
    "rewrite_topic": (
        "Here is a post someone wrote:\n<post>{seed}</post>\n\n"
        "Write a different post, by a different person, about the same subject or "
        "situation. Do not reuse its wording."
    ),
    # A reply to the human tweet: this is what AI comment bots actually do.
    "reply": (
        "Here is a post someone wrote:\n<post>{seed}</post>\n\n"
        "Write a reply to it, as another user commenting underneath."
    ),
    # Only the gist is given, so wording can't be copied at all.
    "from_gist": (
        "Write a post about this everyday situation: {gist}"
    ),
}

STYLE_NOTES = {
    "default": "",
    "lowercase_casual": "Write casually, mostly in lowercase, with light punctuation.",
    "typos_slang": "Write casually, with an abbreviation or two (like u, ur, lol, gonna, idk) and maybe a small typo.",
    "standard": "Use normal capitalization and punctuation.",
    "enthusiastic": "Sound excited; repeated letters or exclamation marks are fine.",
    "sarcastic": "Use a dry, sarcastic tone.",
    "blunt": "Be short and blunt.",
    "rambling": "Ramble a little, like someone thinking out loud.",
}

SENTIMENT_NOTES = {
    "positive": "The overall mood is positive.",
    "negative": "The overall mood is negative.",
    "neutral": "The overall mood is neutral or matter-of-fact.",
}

COMMON_RULES = (
    "Keep it to about {n_words} words and under 280 characters. Plain text on a "
    "single line. Stay on the same everyday subject; don't introduce brands, apps, "
    "news or events that aren't implied by it. Don't mention that you are an AI."
)


def build_prompt(*, seed: str, mode: str, persona: str, style: str,
                 sentiment: Optional[str], n_words: int) -> str:
    if mode == "from_gist":
        body = MODE_TEMPLATES[mode].format(gist=_gist(seed))
    else:
        body = MODE_TEMPLATES[mode].format(seed=seed)
    parts = [body]
    if persona:
        parts.append(f"Write it as {persona}.")
    if STYLE_NOTES.get(style):
        parts.append(STYLE_NOTES[style])
    if sentiment in SENTIMENT_NOTES and mode != "reply":
        parts.append(SENTIMENT_NOTES[sentiment])
    parts.append(COMMON_RULES.format(n_words=max(3, n_words)))
    return "\n\n".join(parts)


def _gist(seed: str) -> str:
    # A crude topic summary that keeps content words and drops the original
    # phrasing, so the model has to write from scratch.
    words = re.findall(r"[A-Za-z']{4,}", seed)
    keep = [w.lower() for w in words if w.lower() not in _STOP][:6]
    return ("someone posting about " + ", ".join(keep)) if keep else "someone's day"


_STOP = {
    "that", "this", "with", "have", "just", "from", "what", "when", "your", "about",
    "would", "there", "their", "they", "will", "been", "were", "know", "like", "really",
    "dont", "don't", "it's", "i'm", "can't", "cant", "im", "going", "want", "some",
}

_PREAMBLE = re.compile(
    r"^(sure|okay|ok|here(?:'s| is| are)|certainly|of course)\b[^:\n]*:\s*", re.IGNORECASE
)
_LABEL = re.compile(r"^(post|reply|tweet|comment)\s*:\s*", re.IGNORECASE)


def clean_output(text: Optional[str]) -> Optional[str]:
    """Strip wrappers models add around the post (quotes, 'Here's a tweet:')."""
    if not text:
        return None
    s = text.strip()
    s = re.sub(r"</?post>", "", s).strip()
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    if len(lines) > 1 and _PREAMBLE.match(lines[0] + " ") and lines[0].rstrip().endswith(":"):
        lines = lines[1:]
    s = " ".join(lines)
    s = _PREAMBLE.sub("", s)
    s = _LABEL.sub("", s)
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'“”":
        s = s[1:-1].strip()
    if len(s) >= 2 and s[0] in "“\"" and s[-1] in "”\"":
        s = s[1:-1].strip()
    return s or None


def too_similar(a: str, b: str, threshold: float = 0.75) -> bool:
    """True when the AI text is a near-copy of the human seed."""
    a2, b2 = a.lower(), b.lower()
    if SequenceMatcher(None, a2, b2).ratio() >= threshold:
        return True
    ta, tb = set(a2.split()), set(b2.split())
    if ta and tb and len(ta & tb) / len(ta | tb) >= threshold:
        return True
    return False
