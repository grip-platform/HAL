"""H338 — HAL Sentinel must signal fallback to caller (visible, not silent).

Before this fix: when LimitSentinel.dispatch() catches a provider failure
and tries the next provider in chain, the returned ChatResponse carries
NO indication that a fallback occurred. The CLI footer shows the
actually-answering provider, but the user has no signal that their
explicitly-selected provider was bypassed.

After this fix: when fallback fires, response.raw['_hal_fallback'] holds
{requested_provider, requested_model, actual_provider, actual_model,
errors_along_way} so the CLI (or any caller) can render a warning.

Goodhart-proof: tests assert BOTH the positive (fallback metadata exists
with correct values) AND the negative (no metadata when first provider
succeeds — silent path stays silent, no false positives).
"""
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lib.hal.providers.openai_compat import OpenAICompatError
from lib.hal.sentinel import LimitSentinel
from lib.hal.types import ChatResponse, TokenUsage


def _make_response(provider: str, model: str, text: str = "ok") -> ChatResponse:
    """Build a minimal ChatResponse with mutable raw dict."""
    return ChatResponse(
        text=text,
        model=model,
        provider=provider,
        finish_reason="stop",
        usage=TokenUsage(),
        cost_usd=None,
        latency_ms=10.0,
        raw={},
    )


def _make_registry(provider_to_behavior: dict, configs: dict | None = None):
    """Build a mock registry where each provider name maps to either:
       - a ChatResponse (success), or
       - an Exception instance (raised on .chat())
    """
    registry = MagicMock()

    providers = {}
    for name, behavior in provider_to_behavior.items():
        p = MagicMock()
        p.name = name
        if isinstance(behavior, Exception):
            p.chat.side_effect = behavior
        else:
            p.chat.return_value = behavior
        providers[name] = p

    registry.get.side_effect = lambda n: providers.get(n)
    registry.list_providers.return_value = [
        MagicMock(name=n) for n in provider_to_behavior
    ]
    registry.routing.fallback_chain = []
    registry._configs = configs or {
        n: MagicMock(models=[f"{n}-default-model"]) for n in provider_to_behavior
    }
    return registry


class TestSentinelVisibleFallback(unittest.TestCase):
    """H338: silent fallback must become visible via response metadata."""

    def test_fallback_metadata_present_when_first_provider_fails(self) -> None:
        """When moonshot raises and groq succeeds, response.raw must carry
        _hal_fallback with both providers identified.

        This is the load-bearing assertion — without it the CLI has no
        signal to show the user that their selected provider was bypassed.
        """
        registry = _make_registry({
            "moonshot": OpenAICompatError("moonshot: HTTP 401 — Invalid Authentication"),
            "groq": _make_response("groq", "llama-3.3-70b", "fallback answer"),
        })
        sentinel = LimitSentinel(registry)

        response = sentinel.dispatch(
            "moonshot/kimi-k2.5",
            "test prompt",
            fallback_chain=["groq"],
        )

        self.assertIsNotNone(response, "response must not be None on successful fallback")
        self.assertEqual(response.provider, "groq", "actual provider should be groq")

        # The load-bearing assertion: fallback metadata must be present.
        self.assertIsInstance(response.raw, dict, "response.raw must be a dict")
        self.assertIn(
            "_hal_fallback",
            response.raw,
            "H338 regression: response.raw missing '_hal_fallback' key — "
            "the CLI cannot warn the user about silent provider swap",
        )

        fb = response.raw["_hal_fallback"]
        self.assertEqual(fb["requested_provider"], "moonshot")
        self.assertEqual(fb["requested_model"], "kimi-k2.5")
        self.assertEqual(fb["actual_provider"], "groq")
        self.assertIn("Invalid Authentication", str(fb["errors_along_way"]))

    def test_no_fallback_metadata_when_first_provider_succeeds(self) -> None:
        """Negative case: when moonshot succeeds on first try, response.raw
        must NOT carry _hal_fallback (false-positive guard).

        Goodhart-proof: a fix that ALWAYS sets _hal_fallback would pass the
        positive test but fail this one. Both must hold.
        """
        registry = _make_registry({
            "moonshot": _make_response("moonshot", "kimi-k2.5", "primary answer"),
            "groq": _make_response("groq", "llama-3.3-70b", "fallback answer"),
        })
        sentinel = LimitSentinel(registry)

        response = sentinel.dispatch(
            "moonshot/kimi-k2.5",
            "test prompt",
            fallback_chain=["groq"],
        )

        self.assertEqual(response.provider, "moonshot")
        self.assertIsInstance(response.raw, dict)
        self.assertNotIn(
            "_hal_fallback",
            response.raw,
            "H338 false-positive: _hal_fallback should NOT be set when "
            "the requested provider succeeded on first attempt",
        )

    def test_fallback_metadata_records_multiple_failures(self) -> None:
        """When moonshot AND groq both fail before deepseek succeeds,
        errors_along_way must accumulate both failures so the CLI can
        report the full ladder, not just the first failure.
        """
        registry = _make_registry({
            "moonshot": OpenAICompatError("moonshot: HTTP 401"),
            "groq": OpenAICompatError("groq: HTTP 503"),
            "deepseek": _make_response("deepseek", "deepseek-v4-flash", "third-time answer"),
        })
        sentinel = LimitSentinel(registry)

        response = sentinel.dispatch(
            "moonshot/kimi-k2.5",
            "test prompt",
            fallback_chain=["groq", "deepseek"],
        )

        self.assertEqual(response.provider, "deepseek")
        fb = response.raw["_hal_fallback"]
        self.assertEqual(fb["requested_provider"], "moonshot")
        self.assertEqual(fb["actual_provider"], "deepseek")
        errs = str(fb["errors_along_way"])
        self.assertIn("moonshot", errs, "first failure must be in errors_along_way")
        self.assertIn("groq", errs, "second failure must be in errors_along_way")


class TestTypedFallbackInfo(unittest.TestCase):
    """H347: AGORA Council #311 concern #1 — typed ChatResponse.fallback_info
    must be set on every fallback (not just the magic raw dict key)."""

    def test_typed_fallback_info_set_when_fallback_fires(self) -> None:
        """Sentinel must populate response.fallback_info (typed FallbackInfo)
        in addition to the legacy response.raw['_hal_fallback'] mirror."""
        registry = _make_registry({
            "moonshot": OpenAICompatError("moonshot: HTTP 401"),
            "groq": _make_response("groq", "llama-3.3-70b", "fallback"),
        })
        sentinel = LimitSentinel(registry)
        response = sentinel.dispatch(
            "moonshot/kimi-k2.5", "test", fallback_chain=["groq"],
        )
        self.assertIsNotNone(response.fallback_info,
            "H347 regression: typed fallback_info field not set")
        self.assertEqual(response.fallback_info.requested_provider, "moonshot")
        self.assertEqual(response.fallback_info.actual_provider, "groq")
        self.assertIsInstance(response.fallback_info.errors_along_way, tuple,
            "errors_along_way must be tuple (immutable, hashable)")

    def test_typed_field_works_when_raw_is_mappingproxy(self) -> None:
        """AGORA #311 concern #4: when response.raw is MappingProxyType
        (read-only), the in-place raw mutation skips silently — but the
        typed fallback_info field MUST still be set so the CLI can warn."""
        from types import MappingProxyType

        readonly_raw = MappingProxyType({"original": "value"})
        primary_response = _make_response("moonshot", "kimi-k2.5")
        # Replace raw with a MappingProxyType (frozen-style)
        from dataclasses import replace
        primary_response = replace(primary_response, raw=readonly_raw)

        registry = _make_registry({
            "moonshot": OpenAICompatError("moonshot: HTTP 401"),
            "groq": replace(_make_response("groq", "llama-3.3-70b"), raw=readonly_raw),
        })
        sentinel = LimitSentinel(registry)
        # Must NOT raise even though raw is read-only.
        response = sentinel.dispatch(
            "moonshot/kimi-k2.5", "test", fallback_chain=["groq"],
        )
        # Typed field is the canonical signal — it MUST be set.
        self.assertIsNotNone(response.fallback_info,
            "H347 regression: typed field not set when raw is read-only")
        self.assertEqual(response.fallback_info.actual_provider, "groq")

    def test_sanitised_for_audit_strips_legacy_raw_key(self) -> None:
        """AGORA #311 concern #5: response.sanitised_for_audit() must strip
        the legacy _hal_fallback key from raw so internal state doesn't
        leak into KONO embeddings or audit logs. Typed field is preserved."""
        registry = _make_registry({
            "moonshot": OpenAICompatError("moonshot: HTTP 401"),
            "groq": _make_response("groq", "llama-3.3-70b", "ok"),
        })
        sentinel = LimitSentinel(registry)
        response = sentinel.dispatch(
            "moonshot/kimi-k2.5", "test", fallback_chain=["groq"],
        )
        # Confirm the legacy key is present pre-sanitisation
        self.assertIn("_hal_fallback", response.raw)

        clean = response.sanitised_for_audit()
        self.assertNotIn("_hal_fallback", clean.raw or {},
            "sanitised_for_audit must strip _hal_fallback from raw")
        # But the typed contract survives — auditors can still see the swap.
        self.assertEqual(clean.fallback_info, response.fallback_info)


class TestCLIFallbackRendering(unittest.TestCase):
    """H338 CLI side: _maybe_print_fallback_warning must emit a visible
    warning when fallback metadata is present, and stay silent otherwise.
    """

    def _capture(self, fn, *args, **kwargs) -> str:
        """Run fn and return its stdout."""
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            fn(*args, **kwargs)
        return buf.getvalue()

    def test_warning_printed_when_metadata_present(self) -> None:
        """When response.raw['_hal_fallback'] is set, the helper must print
        a warning naming both the requested and actual providers."""
        from lib.hal.cli import _maybe_print_fallback_warning

        response = _make_response("groq", "llama-3.3-70b")
        response.raw["_hal_fallback"] = {
            "requested_provider": "moonshot",
            "requested_model": "kimi-k2.5",
            "actual_provider": "groq",
            "actual_model": "llama-3.3-70b",
            "errors_along_way": ["moonshot: HTTP 401"],
        }

        output = self._capture(_maybe_print_fallback_warning, response)
        self.assertIn("moonshot/kimi-k2.5", output, "requested provider/model must appear in warning")
        self.assertIn("groq/llama-3.3-70b", output, "actual provider/model must appear in warning")
        self.assertIn("fell back", output, "warning must indicate a fallback occurred")

    def test_silent_when_no_metadata(self) -> None:
        """No _hal_fallback key → no output (false-positive guard)."""
        from lib.hal.cli import _maybe_print_fallback_warning

        response = _make_response("moonshot", "kimi-k2.5")  # raw is empty {}
        output = self._capture(_maybe_print_fallback_warning, response)
        self.assertEqual(
            output,
            "",
            f"H338 false-positive: helper printed something when no fallback occurred: {output!r}",
        )

    def test_silent_when_raw_is_not_a_dict(self) -> None:
        """Defensive: if response.raw is None or non-dict, helper must not crash
        or print anything. Goodhart-proof against API changes."""
        from lib.hal.cli import _maybe_print_fallback_warning

        response = _make_response("openai", "gpt-4o")
        # Mutate to unusual shapes the type system might not catch.
        for unusual_raw in (None, "not a dict", 42, []):
            response_with_unusual = ChatResponse(
                text="ok", model=response.model, provider=response.provider,
                finish_reason="stop", usage=TokenUsage(), cost_usd=None,
                latency_ms=0.0, raw=unusual_raw,  # type: ignore[arg-type]
            )
            output = self._capture(_maybe_print_fallback_warning, response_with_unusual)
            self.assertEqual(output, "", f"helper crashed/printed on raw={unusual_raw!r}")


if __name__ == "__main__":
    unittest.main()
