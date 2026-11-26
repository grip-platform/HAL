"""Discovery-visible aliases: getting HAL's models into Claude Code's picker.

Why this exists, measured rather than assumed. Claude Code 2.1.267 pointed at
a private HAL gateway (127.0.0.1:49841, 2026-09-10) with
``CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=1`` fetched ``GET
/v1/models?limit=1000``, received 68 ids, and wrote exactly ONE of them to its
own picker cache::

    {"baseUrl":"http://127.0.0.1:49841", ...,
     "models":[{"id":"soruxgpt/claude-3-5-sonnet-20241022"}]}

The other 67 -- ``openai/gpt-6-astra`` and ``moonshot/kimi-k3`` among them --
were dropped. Anthropic's gateway-protocol reference states the rule verbatim:
"Claude Code keeps an entry when its ``id`` contains ``claude`` or
``anthropic`` anywhere in the string, matched case-insensitively, and ignores
the rest." HAL addresses its models ``<provider>/<model>``, so essentially none
of them contain either substring and essentially none of them can be selected
from the picker -- even though every one of them ROUTES when its id is typed
(measured the same night: ``--model openai/gpt-6-astra`` answered PONG, witness
row provider openai status 200 7218ms; ``--model moonshot/kimi-k3`` answered
PONG, provider moonshot status 200 12253ms).

The substring is a PROXY for "speaks the Anthropic wire", and for this gateway
the proxy is simply wrong: the cross-wire translates every one of these models
onto the Anthropic Messages format, which is what makes them routable at all.
So the alias supplies the fact the substring stands in for, and says so in the
id -- ``anthropic-wire/openai/gpt-6-astra`` -- while ``display_name`` keeps the
picker row honest about who actually serves it. Provider-prefixed ids are the
documented idiom (``vertex_ai/claude-sonnet-4-6`` passes the same filter).

Off by default: turning it on doubles a list other harnesses read.
"""
from __future__ import annotations

import os
import unittest
import unittest.mock

from lib.hal.gateway_picker_alias import (
    ALIAS_PREFIX,
    alias_entries,
    alias_for,
    canonical_for,
    is_alias,
    is_discoverable,
    picker_aliases_enabled,
)

# The three ids below are verbatim from the 2026-09-10 measurement.
ASTRA = "openai/gpt-6-astra"
KIMI = "moonshot/kimi-k3"
ALREADY_VISIBLE = "soruxgpt/claude-3-5-sonnet-20241022"


def _declared(*ids):
    """The shape ``declared_model_entries`` produces."""
    return [{"id": i, "object": "model", "owned_by": i.split("/", 1)[0]} for i in ids]


class TestTheClaudeCodeFilter(unittest.TestCase):
    """``is_discoverable`` reimplements Claude Code's documented rule."""

    def test_hal_provider_addressed_ids_are_dropped(self):
        self.assertFalse(is_discoverable(ASTRA))
        self.assertFalse(is_discoverable(KIMI))

    def test_an_id_containing_claude_survives(self):
        # The one of 68 that reached the picker cache in the live run.
        self.assertTrue(is_discoverable(ALREADY_VISIBLE))

    def test_the_match_is_case_insensitive_and_anywhere_in_the_string(self):
        self.assertTrue(is_discoverable("vertex_ai/CLAUDE-sonnet-4-6"))
        self.assertTrue(is_discoverable("bedrock/Anthropic.claude-sonnet-4-5"))

    def test_a_near_miss_does_not_survive(self):
        # Guards the predicate against being loosened into "any /" or similar.
        self.assertFalse(is_discoverable("openai/gpt-6"))
        self.assertFalse(is_discoverable("clause/sonnet"))  # "clause" is not "claude"


class TestAliasRoundTrip(unittest.TestCase):

    def test_an_alias_survives_the_filter(self):
        self.assertTrue(is_discoverable(alias_for(ASTRA)))
        self.assertTrue(is_discoverable(alias_for(KIMI)))

    def test_the_alias_carries_the_canonical_id_intact(self):
        self.assertEqual(alias_for(ASTRA), ALIAS_PREFIX + ASTRA)

    def test_an_alias_maps_back_to_the_model_that_routes(self):
        for canonical in (ASTRA, KIMI):
            self.assertEqual(canonical_for(alias_for(canonical)), canonical)

    def test_a_plain_id_passes_through_canonical_for_untouched(self):
        # The routing hot path calls this on EVERY request; a non-alias must
        # come back byte-identical or every existing model breaks.
        for plain in (ASTRA, KIMI, ALREADY_VISIBLE, "claude-opus-4-8", "", "hal-fanout"):
            self.assertEqual(canonical_for(plain), plain)

    def test_canonical_for_strips_one_prefix_not_all_of_them(self):
        # A model whose own name happened to contain the prefix must not be
        # over-stripped: exactly one leading prefix comes off.
        doubled = ALIAS_PREFIX + ALIAS_PREFIX + ASTRA
        self.assertEqual(canonical_for(doubled), ALIAS_PREFIX + ASTRA)

    def test_an_alias_typed_with_the_window_suffix_keeps_the_suffix(self):
        # THE SEAM BETWEEN TWO STRIPPERS, pinned so a change to either side is
        # visible. An operator has to type ``[1m]`` for the id to be usable at
        # all, so ``anthropic-wire/<id>[1m]`` is the form that actually arrives
        # when a human pins a model by hand rather than picking it from /model.
        #
        # Two layers strip two different things and neither should take the
        # other's job: ``canonical_for`` owns the alias prefix,
        # ``gateway_egress._strip_bracket_suffix`` owns the window suffix.
        # Leaving the bracket here is deliberate, not an oversight -- measured
        # end to end on a private gateway 2026-09-10, where
        # ``anthropic-wire/openai/gpt-6-astra[1m]`` returned 200 and routed to
        # provider ``openai``.
        for canonical in (ASTRA, KIMI):
            typed = alias_for(canonical) + "[1m]"
            self.assertEqual(canonical_for(typed), canonical + "[1m]")

    def test_the_window_suffix_alone_is_not_an_alias(self):
        # A bracket-suffixed id that was never aliased comes back
        # byte-identical, because it reaches the router on the hot path.
        self.assertEqual(canonical_for(ASTRA + "[1m]"), ASTRA + "[1m]")


class TestAliasEntries(unittest.TestCase):

    def test_every_emitted_entry_survives_the_filter(self):
        # The single property the whole feature exists for.
        entries = alias_entries(_declared(ASTRA, KIMI))
        self.assertTrue(entries)
        for e in entries:
            self.assertTrue(is_discoverable(e["id"]), e["id"])

    def test_an_already_visible_id_gets_no_duplicate_row(self):
        entries = alias_entries(_declared(ALREADY_VISIBLE))
        self.assertEqual(entries, [])

    def test_the_row_count_is_exactly_the_dropped_count(self):
        declared = _declared(ASTRA, KIMI, ALREADY_VISIBLE)
        dropped = [d for d in declared if not is_discoverable(d["id"])]
        self.assertEqual(len(alias_entries(declared)), len(dropped))

    def test_the_display_name_names_the_real_provider_not_anthropic(self):
        # The id has to say "anthropic" to pass the filter; the row the human
        # reads must not inherit that and must not imply a Claude model.
        (entry,) = alias_entries(_declared(ASTRA))
        name = entry["display_name"]
        self.assertIn("gpt-6-astra", name)
        self.assertIn("openai", name)
        self.assertNotIn("claude", name.lower())
        self.assertNotIn("anthropic", name.lower())

    def test_the_display_name_differs_from_the_id(self):
        # Claude Code uses display_name only when it differs from the id.
        (entry,) = alias_entries(_declared(KIMI))
        self.assertNotEqual(entry["display_name"], entry["id"])

    def test_the_description_says_who_serves_it(self):
        (entry,) = alias_entries(_declared(KIMI))
        self.assertIn("moonshot", entry["description"])

    def test_a_malformed_declared_row_is_skipped_not_raised(self):
        # /v1/models is a live route: a bad row must never 500 the endpoint.
        self.assertEqual(alias_entries([{"no_id": 1}, "not-a-dict", None]), [])


class TestFlag(unittest.TestCase):

    def test_absent_env_is_off(self):
        self.assertFalse(picker_aliases_enabled({}))

    def test_explicit_truthy_turns_it_on(self):
        for v in ("1", "true", "yes", "on", "TRUE"):
            self.assertTrue(picker_aliases_enabled({"HAL_GATEWAY_PICKER_ALIASES": v}), v)

    def test_explicit_falsey_stays_off(self):
        for v in ("0", "false", "no", "off", ""):
            self.assertFalse(picker_aliases_enabled({"HAL_GATEWAY_PICKER_ALIASES": v}), v)


if __name__ == "__main__":
    unittest.main()


class _FakeInfo:
    def __init__(self, name, models):
        self.name = name
        self.models = models


class _FakeRegistry:
    def __init__(self, infos):
        self._infos = infos

    def list_providers(self):
        return self._infos


class TestDeclaredEntriesWiring(unittest.TestCase):
    """``GET /v1/models`` gains the alias rows only when the flag is on."""

    def setUp(self):
        self._registry = _FakeRegistry([_FakeInfo("openai", ["gpt-6-astra"]),
                                        _FakeInfo("moonshot", ["kimi-k3"])])
        self._saved = os.environ.pop("HAL_GATEWAY_PICKER_ALIASES", None)

    def tearDown(self):
        os.environ.pop("HAL_GATEWAY_PICKER_ALIASES", None)
        if self._saved is not None:
            os.environ["HAL_GATEWAY_PICKER_ALIASES"] = self._saved

    def test_flag_off_serves_the_canonical_list_unchanged(self):
        from lib.hal.gateway import declared_model_entries
        ids = [e["id"] for e in declared_model_entries(self._registry)]
        self.assertEqual(ids, [ASTRA, KIMI])

    def test_flag_on_adds_one_discoverable_row_per_dropped_model(self):
        os.environ["HAL_GATEWAY_PICKER_ALIASES"] = "1"
        from lib.hal.gateway import declared_model_entries
        entries = declared_model_entries(self._registry)
        ids = [e["id"] for e in entries]
        # The canonical ids stay, first and in order -- an OpenAI-wire client
        # that already picks from this list must see no change at its head.
        self.assertEqual(ids[:2], [ASTRA, KIMI])
        added = ids[2:]
        self.assertEqual(added, [alias_for(ASTRA), alias_for(KIMI)])
        for i in added:
            self.assertTrue(is_discoverable(i), i)

    def test_flag_on_makes_astra_and_kimi_reachable_from_the_picker(self):
        # The end-to-end property, stated as the picker would see it: filter
        # the served list the way Claude Code does and both models survive.
        os.environ["HAL_GATEWAY_PICKER_ALIASES"] = "1"
        from lib.hal.gateway import declared_model_entries
        kept = [e for e in declared_model_entries(self._registry)
                if is_discoverable(e["id"])]
        self.assertEqual(len(kept), 2)
        self.assertEqual({canonical_for(e["id"]) for e in kept}, {ASTRA, KIMI})


class TestAliasRouting(unittest.TestCase):
    """An alias picked from the picker must route to the real provider."""

    def test_an_alias_routes_to_the_provider_the_canonical_id_names(self):
        import json as _json
        from lib.hal import gateway as gw
        seen = {}

        def _fake_cross_wire(payload, *, provider, want_stream):
            seen["provider"] = provider
            seen["model"] = payload.get("model")
            return "ROUTED"

        with unittest.mock.patch("lib.hal.gateway_egress.cross_wire_forward",
                                 _fake_cross_wire):
            body = _json.dumps({"model": alias_for(ASTRA), "messages": []}).encode()
            out = gw._maybe_cross_wire("/v1/messages", "POST", body)
        self.assertEqual(out, "ROUTED")
        self.assertEqual(seen["provider"], "openai")
        # The translator downstream must never see the alias.
        self.assertEqual(seen["model"], ASTRA)

    def test_a_same_wire_alias_forwards_the_canonical_id_in_the_bytes(self):
        # Moonshot serves the Anthropic wire, so its branch forwards the
        # client's BODY verbatim -- the rewritten model has to be in there,
        # not just in the parsed payload.
        import json as _json
        from lib.hal import gateway as gw
        seen = {}

        def _fake_same_wire(provider, path, body, client_headers):
            seen["provider"] = provider
            seen["model"] = _json.loads(body)["model"]
            return "SAME_WIRE"

        with unittest.mock.patch("lib.hal.gateway_egress.anthropic_same_wire_forward",
                                 _fake_same_wire):
            body = _json.dumps({"model": alias_for(KIMI), "messages": []}).encode()
            out = gw._maybe_cross_wire("/v1/messages", "POST", body)
        self.assertEqual(out, "SAME_WIRE")
        self.assertEqual(seen["provider"], "moonshot")
        self.assertEqual(seen["model"], KIMI)

    def test_a_plain_model_still_reaches_the_verbatim_relay(self):
        # claude-* must keep returning None (the #3274 verbatim path).
        import json as _json
        from lib.hal import gateway as gw
        body = _json.dumps({"model": "claude-opus-4-8", "messages": []}).encode()
        self.assertIsNone(gw._maybe_cross_wire("/v1/messages", "POST", body))


class TestWitnessLabelsTheRealProvider(unittest.TestCase):
    """The audit row must name who actually served the request.

    Regression anchor for a defect the alias itself introduced and a live run
    caught: ``_witness_messages_rail`` re-derives the provider from the
    caller's body, which still carries the alias, so
    ``same_wire_provider("anthropic-wire/openai/gpt-6-astra")`` returned None
    and the row was labelled ``provider=anthropic`` for a request OpenAI
    served. A witness that cannot disagree with reality is worse than none.
    """

    def _row_for(self, model_id):
        from lib.hal import gateway as gw
        import json as _json
        captured = {}

        def _fake_witness_chat(**kw):
            captured.update(kw)

        class _Resp:
            status = 200

        with unittest.mock.patch("lib.hal.gateway_witness.witness_chat",
                                 _fake_witness_chat):
            body = _json.dumps({"model": model_id, "messages": []}).encode()
            gw._witness_messages_rail(body, _Resp(), 0.0)
        return captured

    def test_an_alias_is_witnessed_against_its_real_provider(self):
        row = self._row_for(alias_for(ASTRA))
        self.assertEqual(row["provider"], "openai")
        # Inverted 2026-10-05: this asserted True, which demanded the defect.
        # Serving the model the caller asked for is routing, not a failover.
        self.assertFalse(row["failed_over"])

    def test_the_witnessed_model_is_the_one_that_ran(self):
        row = self._row_for(alias_for(KIMI))
        self.assertEqual(row["provider"], "moonshot")
        self.assertEqual(row["model"], KIMI)

    def test_a_plain_model_is_witnessed_exactly_as_before(self):
        row = self._row_for(ASTRA)
        self.assertEqual(row["provider"], "openai")
        self.assertEqual(row["model"], ASTRA)

    def test_a_claude_model_is_still_witnessed_as_anthropic(self):
        row = self._row_for("claude-opus-4-8")
        self.assertEqual(row["provider"], "anthropic")
        self.assertFalse(row["failed_over"])


class TestAgoraFalsifiers574(unittest.TestCase):
    """Concerns raised by the AGORA council on PR #574, each R0-tested.

    Speaker C named four. Three reproduced and are fixed here; the fourth --
    that ``_alias_entry`` would IndexError on an id containing no "/" -- did
    NOT reproduce, because the owner is taken from ``split("/", 1)[0]``, which
    always exists. Its anchor test is kept anyway: the speaker's "what I would
    falsify if I had time" becomes a forward anchor whether or not it was
    already true.
    """

    def test_a_bare_id_with_no_slash_yields_a_sane_row(self):
        # Speaker C's falsifier. Refuted -- the code already handles it.
        (entry,) = alias_entries([{"id": "gpt-6-astra"}])
        self.assertEqual(entry["owned_by"], "gpt-6-astra")
        self.assertIn("gpt-6-astra", entry["display_name"])
        self.assertTrue(is_discoverable(entry["id"]))

    def test_an_alias_on_the_openai_wire_reaches_the_real_provider(self):
        """Speaker C, concern 1. CONFIRMED before the fix.

        The alias rows are in the same ``/v1/models`` body an OpenAI-wire
        client reads (Codex consumes the injected ``models`` array), so such a
        client can pick one -- and that wire resolved the model separately,
        forwarding the alias verbatim to the provider.
        """
        import json as _json
        from lib.hal import gateway_openai as go
        seen = {}

        class _Handler:
            command = "POST"
            headers = {}

        def _fake_dispatch(handler, proxy, raw, payload, model, started):
            seen["model"] = model
            seen["raw_model"] = _json.loads(raw)["model"]

        body = _json.dumps({"model": alias_for(ASTRA), "messages": []}).encode()
        with unittest.mock.patch.object(go, "read_request_body", lambda h: body), \
             unittest.mock.patch.object(go, "_dispatch_openai_chat", _fake_dispatch):
            go.handle_chat_completions(_Handler(), object())
        self.assertEqual(seen["model"], ASTRA)
        # raw is forwarded VERBATIM on the same-wire relay, so the rewrite has
        # to reach the bytes, not just the parsed payload.
        self.assertEqual(seen["raw_model"], ASTRA)

    def test_gpt6_locked_sampling_fields_never_reach_the_responses_wire(self):
        """Speaker C, concern 2. CONFIRMED before the fix.

        The chat wire drops the sampling fields gpt-6 refuses
        (``_fix_gpt6_request_params``), but the Responses branch returns before
        that runs, and the Responses translator copied temperature/top_p
        straight through -- a guaranteed provider 400.
        """
        from lib.hal.gateway_crosswire_responses import anthropic_to_responses_request as _translate_request
        from lib.hal.model_quirks import locked_sampling_params
        payload = {"model": "gpt-6-astra", "messages": [],
                   "temperature": 0.7, "top_p": 0.9, "max_tokens": 16}
        out = _translate_request(payload)
        locked = locked_sampling_params("gpt-6-astra")
        self.assertTrue(locked, "precondition: gpt-6 must lock some params")
        for field in locked:
            self.assertNotIn(field, out, field)

    def test_a_model_that_locks_nothing_keeps_its_sampling_fields(self):
        """Control: the drop is keyed on the model, not applied to everything.

        gemini locks no sampling params (``locked_sampling_params`` returns
        ``()``), unlike kimi, which locks five -- so kimi is the wrong control
        and reading this assertion as "any non-gpt-6 model" would be wrong.
        """
        from lib.hal.gateway_crosswire_responses import anthropic_to_responses_request as _translate_request
        from lib.hal.model_quirks import locked_sampling_params
        model = "gemini/gemini-3.1-pro-preview"
        self.assertEqual(locked_sampling_params(model), ())
        out = _translate_request({"model": model, "messages": [],
                                  "top_p": 0.9, "temperature": 0.3, "max_tokens": 16})
        self.assertEqual(out.get("top_p"), 0.9)
        self.assertEqual(out.get("temperature"), 0.3)

    def test_a_non_dict_tool_call_does_not_raise(self):
        """Speaker C, concern 3. CONFIRMED before the fix.

        ``_assistant_items`` tested ``isinstance(call, dict)`` and then called
        ``call.get("id")`` unguarded on the very next line, so the guard was
        defeated by its own next statement. The earlier test only covered a
        malformed ``function`` INSIDE a well-formed call.
        """
        from lib.hal.gateway_crosswire_responses import _assistant_items
        for bad in ("not-a-dict", ["a"], 7, None):
            items = _assistant_items({"role": "assistant", "content": None,
                                      "tool_calls": [bad]})
            self.assertEqual(items, [], bad)


class TestTheResponsesRailResolvesTheAlias(unittest.TestCase):
    """``POST /v1/responses`` is the third ingress rail, and it must strip too.

    Regression anchor for H6, MEASURED on a private gateway (127.0.0.1:52931,
    HAL origin/main bb815496) with ONE key held constant across four cells so
    the rail is the only variable::

        /v1/messages   openai/gpt-6-astra                    -> 200 "PONG"
        /v1/messages   anthropic-wire/openai/gpt-6-astra     -> 200 "PONG"
        /v1/responses  openai/gpt-6-astra                    -> 200 (OpenAI)
        /v1/responses  anthropic-wire/openai/gpt-6-astra     -> 401   <-- defect

    The alias worked on ``/v1/messages`` because ``_maybe_cross_wire``
    canonicalises at ``gateway.py:501``; it failed on ``/v1/responses``
    because ``handle_responses`` read ``payload["model"]`` raw.
    ``same_wire_provider`` then returned ``None``, so the request fell through
    to the verbatim Anthropic relay -- an OpenAI-intended request served by
    the wrong vendor, exactly the H2 shape on a rail H2's fix did not cover.

    It is reachable, not theoretical: ``declared_model_entries`` appends the
    alias rows and ``_inject_models_list`` mirrors every one of them into the
    Codex-compat ``models`` array (measured on the same gateway: 67 of 67),
    and Codex posts to ``/v1/responses``.
    """

    def _dispatch(self, model_id):
        """Drive ``handle_responses`` far enough to see which provider it picked."""
        import io
        import json as _json
        from lib.hal import gateway_responses as gr

        seen = {}

        def _fake_forward_same_wire(provider, path, body, *a, **kw):
            seen["provider"] = provider
            seen["path"] = path
            seen["model"] = _json.loads(body)["model"]
            raise RuntimeError("stop-after-dispatch")

        raw = _json.dumps({"model": model_id, "input": []}).encode()

        class _H:
            command = "POST"
            path = "/v1/responses"
            headers = {"Content-Length": str(len(raw)),
                       "Content-Type": "application/json"}
            rfile = io.BytesIO(raw)
            wfile = io.BytesIO()

            def send_response(self, code):
                seen["status"] = code

            def send_header(self, *a):
                pass

            def end_headers(self):
                pass

        class _Proxy:
            def forward(self, path, body, client_headers, *, method="POST", **kw):
                # The verbatim-Anthropic fallback. Reaching it IS the defect.
                seen["provider"] = "anthropic"
                seen["path"] = path
                seen["model"] = _json.loads(body)["model"]
                raise RuntimeError("stop-after-dispatch")

        with unittest.mock.patch("lib.hal.gateway_egress.forward_same_wire",
                                 _fake_forward_same_wire):
            gr.handle_responses(_H(), _Proxy())
        return seen

    def test_an_alias_on_the_responses_rail_reaches_the_real_provider(self):
        seen = self._dispatch(alias_for(ASTRA))
        self.assertEqual(seen["provider"], "openai",
                         "the alias fell through to the Anthropic relay (H6)")
        self.assertEqual(seen["path"], "/v1/responses")

    def test_the_alias_never_reaches_the_provider_in_the_forwarded_bytes(self):
        # forward_same_wire relays `raw` VERBATIM, so canonicalising the parsed
        # payload alone would still send OpenAI an id it does not have.
        seen = self._dispatch(alias_for(ASTRA))
        self.assertEqual(seen["model"], ASTRA)

    def test_the_canonical_id_is_unchanged_on_that_rail(self):
        # The control cell: this one already passed before the fix, and must
        # keep passing -- the fix is identity for every non-alias request.
        seen = self._dispatch(ASTRA)
        self.assertEqual(seen["provider"], "openai")
        self.assertEqual(seen["model"], ASTRA)

    def test_a_body_naming_no_model_is_not_treated_as_an_alias(self):
        # Raised by the AGORA judge on PR #575 against the guard this rail
        # used to use: it compared ``canonical_for(payload.get("model", ""))``
        # against ``payload.get("model")``, defaulting on one side only, so a
        # body carrying no model compared "" against None, never matched, and
        # was rewritten to carry ``"model": ""``.
        #
        # HONEST SCOPE, because the first version of this test asserted the
        # forwarded bytes and could not fail: with no model there is no
        # same-wire provider, so the request goes down the Responses-to-
        # Messages translation path, which rewrites the body and supplies its
        # own ``model`` key either way. The mutation was therefore real in the
        # code and unobservable at the handler's output. What IS worth pinning
        # is the predicate the rail now asks, so assert that directly.
        self.assertFalse(is_alias(""))
        self.assertFalse(is_alias(ASTRA))
        self.assertTrue(is_alias(alias_for(ASTRA)))
