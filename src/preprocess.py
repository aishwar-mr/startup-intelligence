"""Text preprocessing shared across retrieval and sentiment."""
import re


def clean(text: str) -> str:
    """Lowercase, strip URLs and punctuation, collapse whitespace."""
    text = str(text).lower()
    text = re.sub(r"http\S+", " ", text)      # strip urls
    text = re.sub(r"[^a-z0-9\s]", " ", text)  # keep letters/digits
    text = re.sub(r"\s+", " ", text).strip()
    return text
