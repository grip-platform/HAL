"""Tests for the cost-aware /api/infer route resolver (resolve_infer_model).

Goodhart-resistance: every test asserts the RESOLVED model string (or None),
never a call count. Each fails under a plausible mutation:
  * if "classify" mapped to opus-4-7 -> test_classify_routes_to_cheap fails
    (the whole cost win is gone);
  * if the no-field default returned a model instead of None ->
    test_default_unchanged fails (the byte-identical-default safety contract);
  * if route silently won over an explicit model -> test_explicit_model_wins
    fails (caller-override regression).
"""
from __future__ import annotations

from lib.hal.server import _INFER_ROUTES, resolve_infer_model


class TestResolveInferModel:
    def test_classify_routes_to_cheap(self):
        """The benchmarked cost win: classify -> groq llama-3.3-70b (free,
        10/10 verdict-match vs opus). This is the reason the route exists."""
        assert resolve_infer_model({"route": "classify"}) == \
            "groq/llama-3.3-70b-versatile"

    def test_default_unchanged_when_no_route_or_model(self):
        """Safety contract: a body with neither field resolves to None, so
        grip_infer uses its default primary (opus-4-7) exactly as before the
        route mechanism existed. Merging this changes nothing for old callers."""
        assert resolve_infer_model({"prompt": "hi"}) is None
        assert resolve_infer_model({}) is None

    def test_explicit_model_wins_over_route(self):
        """Caller override preserved: an explicit model beats a route, even a
        known one. The route is a default, never an override of the caller."""
        assert resolve_infer_model(
            {"model": "gemini/gemini-2.5-flash", "route": "classify"}) == \
            "gemini/gemini-2.5-flash"

    def test_explicit_model_alone_is_passed_through(self):
        assert resolve_infer_model({"model": "xai/grok-3"}) == "xai/grok-3"

    def test_unknown_route_falls_through_to_none(self):
        """Fail-safe: a typo'd / unconfigured route never fails the call — it
        degrades to the default primary, the route's whole reason for being."""
        assert resolve_infer_model({"route": "no-such-route"}) is None

    def test_non_string_route_ignored(self):
        """A malformed route value (e.g. a dict from a buggy caller) must not
        raise — it degrades to None like any unresolvable route."""
        assert resolve_infer_model({"route": {"oops": 1}}) is None
        assert resolve_infer_model({"route": 42}) is None

    def test_empty_model_string_does_not_win(self):
        """An empty-string model is falsy and must NOT pin an empty primary;
        the route (or None) takes over. Guards the `if explicit:` truthiness."""
        assert resolve_infer_model({"model": "", "route": "classify"}) == \
            "groq/llama-3.3-70b-versatile"
        assert resolve_infer_model({"model": ""}) is None

    def test_classify_route_is_registered(self):
        """The classify route must exist in the operator-editable map — guards
        an accidental deletion of the only shipped route."""
        assert "classify" in _INFER_ROUTES
        assert _INFER_ROUTES["classify"] == "groq/llama-3.3-70b-versatile"
