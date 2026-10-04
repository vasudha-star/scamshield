"""Text cleaning and normalization module for ScamShield (Phase 3).

Applies:
- URL extraction prior to text cleaning (preserves original URL strings).
- Unicode NFKC normalization (preserving Hindi, Telugu, and English scripts).
- Whitespace regularization.
- Safe PII handling without removing scam indicators (preserves OTP, account numbers context).
"""

from __future__ import annotations

import re
import unicodedata

# Strict URL regex that captures HTTP/HTTPS and www links before cleaning
URL_PATTERN = re.compile(
    r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
)

# Email address pattern
EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b")

# Phone number pattern (Indian 10-digit mobile & international format)
PHONE_PATTERN = re.compile(r"(?:\+91[\-\s]?)?[6789]\d{9}\b")

# Multi-whitespace and newline normalizer
MULTISPACE_PATTERN = re.compile(r"[^\S\r\n]+")
MULTINEWLINE_PATTERN = re.compile(r"\n{3,}")


def extract_urls(text: str) -> list[str]:
    """Extracts all URL strings from raw text before any modifications."""
    if not isinstance(text, str) or not text.strip():
        return []
    matches = URL_PATTERN.findall(text)
    urls = []
    for match in matches:
        url_str = match[0] if isinstance(match, tuple) else match
        if url_str:
            urls.append(url_str.strip())
    return urls


def normalize_unicode(text: str) -> str:
    """Normalizes text using Unicode NFKC form.

    Crucially preserves Devanagari (Hindi) and Telugu unicode blocks
    while standardizing lookalike characters and ligatures.
    """
    if not isinstance(text, str):
        return ""
    # NFKC normalizes compatibility characters while preserving script semantics
    normalized = unicodedata.normalize("NFKC", text)
    # Strip null bytes and non-printable control characters except tabs and newlines
    cleaned = "".join(ch for ch in normalized if ch in ("\n", "\r", "\t") or unicodedata.category(ch)[0] != "C")
    return cleaned


def clean_text(text: str, mask_pii: bool = False) -> tuple[str, list[str], bool]:
    """Full standardization pipeline for message text.

    Args:
        text: Raw text string.
        mask_pii: If True, replaces specific phone numbers and personal emails with tokens.

    Returns:
        tuple of (cleaned_text, extracted_urls, url_present)
    """
    if not isinstance(text, str):
        return "", [], False

    # 1. Extract URLs BEFORE text cleaning
    urls = extract_urls(text)
    url_present = len(urls) > 0

    # 2. Unicode NFKC normalization
    text = normalize_unicode(text)

    # 3. Optional PII masking (preserves surrounding urgency words)
    if mask_pii:
        text = EMAIL_PATTERN.sub("<EMAIL>", text)
        text = PHONE_PATTERN.sub("<PHONE>", text)

    # 4. Whitespace cleanup
    text = MULTISPACE_PATTERN.sub(" ", text)
    text = MULTINEWLINE_PATTERN.sub("\n\n", text)
    text = text.strip()

    return text, urls, url_present
