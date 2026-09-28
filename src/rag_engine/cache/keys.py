import hashlib

EMBEDDING_TIER = "embedding"
EMBEDDING_TTL = 7 * 24 * 3600


def _digest(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


def embedding_key(model: str, text: str) -> str:
    return f"emb:{_digest(model, text)}"
