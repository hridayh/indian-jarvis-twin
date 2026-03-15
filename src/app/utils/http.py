from __future__ import annotations

import requests


def download_bytes(
    url: str,
    *,
    auth: tuple[str, str] | None = None,
    timeout_s: int = 30,
) -> bytes:
    """
    Download content from URL and return raw bytes.

    Twilio MediaUrl requires HTTP Basic Auth: (AccountSid, AuthToken).
    Many deployments instead proxy this via Twilio SDK; this keeps it simple.
    """
    resp = requests.get(url, auth=auth, timeout=timeout_s)
    resp.raise_for_status()
    return resp.content

