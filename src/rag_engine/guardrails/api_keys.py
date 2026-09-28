"""API key auth: every MCP request carries a key; the key's name in ENGINE_API_KEYS is the caller."""

import hmac

from rag_engine.config import settings
from rag_engine.models import AuthError


def authenticate(api_key: str | None) -> str:
    """Return the caller for a valid key. Fails closed: no key, an unknown key, or no keys
    configured at all are all rejected."""
    if api_key:
        for known_key, caller in settings.api_key_callers.items():
            if hmac.compare_digest(api_key.encode(), known_key.encode()):
                return caller
    raise AuthError("Missing or invalid API key")
