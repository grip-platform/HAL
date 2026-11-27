"""Local Ollama verifier for the semantic response cache (HAL#965).

Two calls to the local Ollama daemon, never to a cloud model:

  * ``embed(text)`` returns a nomic-embed-text vector. The cache uses it only to pick the one
    stored prompt worth checking. Embeddings place opposites close together, so a vector
    never decides a hit on its own.
  * ``same_request(a, b)`` asks a local chat model, at temperature 0 with a fixed seed,
    whether one answer is correct for both prompts. The cache reuses an answer only when the
    whole reply is YES; "YES, but ..." is a NO.

Every failure fails closed: Ollama down, a model missing, a timeout, an HTTP protocol error,
an unreadable reply or a vector that is not all finite numbers gives no vector, or NO, so the
cache misses rather than guesses. After a failure the verifier rests for ``rest_seconds``, so
a dead daemon costs one timeout, not one per lookup.

Cold models heal themselves. Ollama cancels a model load when the client gives up (measured
for HAL#965: qwen3:8b was still not loaded 40 s after a 3 s request timed out, and a load
took 6.1 s), so a short-timeout call can never warm a cold judge. ``start_warm()`` runs one
background ``warm()``, which loads both models with a long timeout and keeps them resident
for 30 minutes. The cache factory calls it when it builds the cache, and every failure calls
it again.

A warm-up never ends a rest. It sends an empty prompt, which shows that the models load but
not that a real call finishes inside the timeout, so a host whose judge calls run past the
timeout pays one timeout per rest, not one per lookup (HAL#987). The price: a judge that timed
out while cold stays unused until the rest ends, although the warm-up loads it within seconds.

Each prompt reaches the judge as plain text, so a prompt can argue with the judge, and a line
break inside one can imitate the instruction's layout. Quoting each prompt as a JSON string
closed that gap but was measured for HAL#965 to cost recall with no gain in safety: held-out
hits fell from 18 to 14 of 30 (from 14 to 8 with the prompts swapped), while near-misses and
unrelated pairs stayed at 0 either way. So prompts stay plain. The risk stays inside one
conversation, because each ``Conversation`` holds its own cache, and a judged pair must first
clear the embedding floor and the deterministic veto. A prompt longer than 2,000 characters is
never judged.

Measured for HAL#965 with qwen3:8b and a candidate floor of 0.70, through the cache end to
end: 18 of 20 and 17 of 30 paraphrase pairs hit (17 and 12 with the two prompts swapped), with
no wrong answer on 50 near-miss pairs or 100 unrelated pairs, and a repeated pass made the same
decision on every pair. A warm judge call took about 0.35 seconds.
"""
from __future__ import annotations

import http.client
import json
import math
import re
import threading
import time
import urllib.request
from typing import Callable

_JUDGE_PROMPT = (
    "You check whether a cached answer can be reused. Two requests were sent to a developer "
    "assistant.\nRequest A: {a}\nRequest B: {b}\nWould exactly the same answer be correct and "
    "complete for both? Answer NO if they differ in any object, file, name, number, direction, "
    "order, negation, quantity, scope, time or action, even slightly. Answer only YES or NO."
)
# A longer prompt is never judged: it misses rather than being truncated, because a cut could
# hide the very difference that matters.
_MAX_JUDGED_CHARS = 2000
_YES_RE = re.compile(r"\s*yes\s*[.!]?\s*", re.IGNORECASE)
_KEEP_ALIVE = "30m"


class OllamaVerifier:
    """Embeddings and a yes-or-no judge from the local Ollama daemon, failing closed."""

    def __init__(
        self,
        judge_model: str,
        *,
        embed_model: str = "nomic-embed-text",
        base_url: str = "http://127.0.0.1:11434",
        timeout_s: float = 5.0,
        warm_timeout_s: float = 300.0,
        rest_seconds: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        if not judge_model.strip():
            raise ValueError("judge_model must name an Ollama model")
        self.name = f"ollama:{judge_model}"
        self._judge_model = judge_model
        self._embed_model = embed_model
        self._base = base_url.rstrip("/")
        self._timeout = timeout_s
        self._warm_timeout = warm_timeout_s
        self._rest = rest_seconds
        self._clock = clock
        self._resting_until = 0.0
        self._warm_lock = threading.Lock()
        self.warm_thread: threading.Thread | None = None

    def embed(self, text: str) -> list[float] | None:
        """A unit-length embedding of ``text``, or None when Ollama cannot give a numeric one."""
        reply = self._post("/api/embeddings", {"model": self._embed_model, "prompt": text})
        raw = reply.get("embedding") if isinstance(reply, dict) else None
        if not isinstance(raw, list) or not raw:
            return None
        try:
            vector = [float(v) for v in raw]
        except (TypeError, ValueError):
            return None
        if not all(math.isfinite(v) for v in vector):
            return None
        norm = math.sqrt(sum(v * v for v in vector))
        return [v / norm for v in vector] if norm else None

    def same_request(self, a: str, b: str) -> bool:
        """True only when the judge's whole reply is YES; every other reply or failure is a NO.

        A prompt longer than ``_MAX_JUDGED_CHARS`` is never sent: the answer is NO at once.
        """
        if max(len(a), len(b)) > _MAX_JUDGED_CHARS:
            return False
        reply = self._post("/api/generate", {
            "model": self._judge_model,
            "prompt": _JUDGE_PROMPT.format(a=a, b=b),
            "stream": False,
            "think": False,
            "keep_alive": _KEEP_ALIVE,
            "options": {"temperature": 0, "seed": 1},
        })
        text = reply.get("response") if isinstance(reply, dict) else None
        return isinstance(text, str) and _YES_RE.fullmatch(text) is not None

    def warm(self) -> bool:
        """Load the judge and embedding models and keep them resident; True when both loaded.

        A success leaves any rest running: an empty prompt shows that the models load, not that
        a real call finishes inside the timeout, so only the end of the rest lets real calls
        through again.
        """
        judge = self._request("/api/generate", {"model": self._judge_model, "prompt": "", "stream": False,
                                                "keep_alive": _KEEP_ALIVE}, self._warm_timeout)
        embedder = self._request("/api/embeddings", {"model": self._embed_model, "prompt": "",
                                                     "keep_alive": _KEEP_ALIVE}, self._warm_timeout)
        return judge is not None and embedder is not None

    def start_warm(self) -> None:
        """Run ``warm()`` in one background thread, unless one is already running."""
        with self._warm_lock:
            if self.warm_thread is not None and self.warm_thread.is_alive():
                return
            self.warm_thread = threading.Thread(target=self.warm, daemon=True, name="cache-verifier-warm")
            self.warm_thread.start()

    def _post(self, path: str, payload: dict) -> object:
        """The decoded reply, or None while resting or after a failure (which rests and warms)."""
        if self._clock() < self._resting_until:
            return None
        reply = self._request(path, payload, self._timeout)
        if reply is None:
            self._resting_until = self._clock() + self._rest
            self.start_warm()
        return reply

    def _request(self, path: str, payload: dict, timeout: float) -> object:
        """One POST to Ollama: the decoded JSON reply, or None on any failure."""
        request = urllib.request.Request(
            self._base + path, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read())
        except (OSError, ValueError, http.client.HTTPException):
            return None
