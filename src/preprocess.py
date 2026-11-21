"""Text preprocessing shared across validation, retrieval and sentiment."""
import html
import re
import unicodedata

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def normalise_text(value) -> str:
    """Display-safe normalisation: NFKC, strip HTML tags/entities, collapse whitespace.

    None becomes ''. Used by validation; the raw value is never overwritten.
    """
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    text = _TAG_RE.sub(" ", text)
    text = html.unescape(text)
    return _WS_RE.sub(" ", text).strip()


def clean(text) -> str:
    """Matching normalisation for TF-IDF: lowercase, drop URLs and punctuation."""
    text = str(text).lower()
    text = re.sub(r"http\S+", " ", text)      # strip urls
    text = re.sub(r"[^a-z0-9\s]", " ", text)  # keep letters/digits
    return re.sub(r"\s+", " ", text).strip()
