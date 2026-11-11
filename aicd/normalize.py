"""Text cleanup applied identically to human and AI text.

The 2009 tweet dump has collection artifacts that never appear in AI output
(backtick apostrophes, mojibake, stripped @usernames, censored profanity).
If only one side carries an artifact, a classifier learns the artifact
instead of "human vs AI". Every rule here is applied to BOTH classes.
"""

from __future__ import annotations

import re
import unicodedata

# Mojibake for a curly apostrophe that went through several bad re-encodings.
_MOJIBAKE = ["Ã¯Â¿Â½", "ï¿½", "�", "ÃÂ´", "ÃÂ"]

_URL = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
_MENTION = re.compile(r"(?<!\w)@\w+")
# Leftover username fragments: the dataset stripped "@" plus part of the name,
# leaving tokens like "_elj" or "_04071991" at the start of the tweet.
_LEADING_USER_FRAGMENT = re.compile(r"^_\w+\s*")
_WS = re.compile(r"\s+")

# The tweet dataset masks strong profanity as "****"; mask the same words in
# AI text so "****" is not a human-only token.
_PROFANITY = re.compile(
    r"\b(fuck\w*|shit\w*|bitch\w*|cunt\w*|motherfuck\w*|bullshit)\b", re.IGNORECASE
)

_CHAR_MAP = {
    "`": "'",
    "‘": "'",
    "’": "'",
    "‚": "'",
    "‛": "'",
    "“": '"',
    "”": '"',
    "„": '"',
    "–": "-",
    "—": " - ",
    "―": " - ",
    "…": "...",
    " ": " ",
}


def _is_emoji_or_symbol(ch: str) -> bool:
    cat = unicodedata.category(ch)
    cp = ord(ch)
    if cat == "So":  # "Symbol, other" covers almost all emoji
        return True
    # Variation selectors, zero-width joiner, skin-tone modifiers.
    return cp in (0x200D, 0xFE0E, 0xFE0F) or 0x1F3FB <= cp <= 0x1F3FF


def normalize(text: str, *, strip_emoji: bool = True) -> str:
    """Return a cleaned copy of ``text``. Safe to run more than once."""
    if text is None:
        return ""
    s = str(text)
    for bad in _MOJIBAKE:
        s = s.replace(bad, "'")
    s = unicodedata.normalize("NFKC", s)
    s = "".join(_CHAR_MAP.get(ch, ch) for ch in s)
    if strip_emoji:
        s = "".join(ch for ch in s if not _is_emoji_or_symbol(ch))
    s = _URL.sub("URL", s)
    s = _MENTION.sub("", s)
    s = _WS.sub(" ", s).strip()
    s = _LEADING_USER_FRAGMENT.sub("", s)
    s = _PROFANITY.sub("****", s)
    s = _WS.sub(" ", s).strip()
    return s


def word_count(text: str) -> int:
    return len(text.split())
