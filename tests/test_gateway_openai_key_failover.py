"""OpenAI's funded-key failover is GONE from the gateway egress, and stays gone.

INVERTED 2026-10-05 (InfiniteMachine Step 5h, the spend guard). This file used
to require that a 401 or 429 on the primary OpenAI key made the gateway retry
the same request once with a SECOND funded account (``hal-openai-failover``,
``lib/hal/key_failover.py``). Both accounts are funded API keys, so that retry
spent money HAL chose on the operator's behalf: a paid rung, which the
InfiniteMachine contract says is removed rather than merely tested for. Every
test below keeps its original scenario and inverts the expectation, so a change
that brings the rotation back turns them red.

Hermetic + secret-free: opener + key_resolver are injected (no network, no
keychain), and all keys are obvious FAKES (``sk-test-fake-*``).
"""
import importlib
import unittest

from lib.hal.gateway_egress import forward_same_wire

# Fake keys ONLY — never a real secret. The two distinct values let a test prove
# WHICH account a given upstream call authenticated with.
_PRIMARY_KEY = "sk-test-fake-primary"
_FAILOVER_KEY = "sk-test-fake-failover"
_PRIMARY_SVC = "hal-openai"
_FAILOVER_SVC = "hal-openai-failover"


class _RecordingResolver:
    """Maps keychain service -> fake key, and records every service asked for."""

    def __init__(self, keys=None):
        self.keys = {_PRIMARY_SVC: _PRIMARY_KEY, _FAILOVER_SVC: _FAILOVER_KEY} if keys is None else keys
        self.asked = []

    def __call__(self, svc):
        self.asked.append(svc)
        return self.keys.get(svc, "")


class _FakeResp:
    def __init__(self, status=200, body=b"{}"):
        self.status = status
        self.headers = [("Content-Type", "application/json")]
        self._body = body
        self._read = False
        self.closed = False

    def read(self, size=-1):
        if self._read:
            return b""
        self._read = True
        return self._body

    def close(self):
        self.closed = True


class _ScriptedOpener:
    """Returns a scripted status per successive call; records every call's
    Bearer token + url + body. The Bearer is the Goodhart anchor — it proves
    which ACCOUNT authenticated each attempt (the two keys are distinct)."""

    def __init__(self, statuses):
        self._statuses = list(statuses)
        self.calls = []  # [{"url", "body", "bearer"}]

    def __call__(self, method, url, headers, body):
        hdrs = dict(headers)
        auth = hdrs.get("Authorization", "")
        bearer = auth[len("Bearer "):] if auth.startswith("Bearer ") else None
        self.calls.append({"url": url, "body": body, "bearer": bearer})
        idx = len(self.calls) - 1
        status = self._statuses[idx] if idx < len(self._statuses) else self._statuses[-1]
        return _FakeResp(status=status)


def _forward(opener, resolver):
    return forward_same_wire(
        "openai", "/v1/chat/completions", b'{"model":"gpt-5.5"}',
        opener=opener, key_resolver=resolver,
    )


class TestKeyErrorsAreRelayedNeverRetried(unittest.TestCase):
    """The scenarios that used to fail over now relay the upstream's answer."""

    def _assert_relayed_once(self, statuses, expected_status):
        op, resolver = _ScriptedOpener(statuses), _RecordingResolver()
        resp = _forward(op, resolver)
        self.assertEqual(resp.status, expected_status)
        self.assertEqual(len(op.calls), 1, "a second upstream call is the removed rung")
        self.assertEqual(op.calls[0]["bearer"], _PRIMARY_KEY)
        self.assertNotIn(_FAILOVER_SVC, resolver.asked,
                         "the second funded account was read: the paid rung is back")

    def test_401_primary_is_relayed_not_failed_over(self):
        self._assert_relayed_once([401, 200], 401)

    def test_429_primary_is_relayed_not_failed_over(self):
        self._assert_relayed_once([429, 200], 429)

    def test_the_failover_key_is_never_presented(self):
        # Whatever the upstream says, the second account's key never leaves.
        for statuses in ([401, 200], [429, 401, 200], [429, 429]):
            op = _ScriptedOpener(statuses)
            _forward(op, _RecordingResolver())
            self.assertNotIn(_FAILOVER_KEY, [c["bearer"] for c in op.calls], statuses)


class TestUnchangedPaths(unittest.TestCase):
    """What the removal must NOT change: success and non-key errors, one call."""

    def test_200_first_call_does_not_retry(self):
        op = _ScriptedOpener([200])
        resp = _forward(op, _RecordingResolver())
        self.assertEqual(resp.status, 200)
        self.assertEqual(len(op.calls), 1)
        self.assertEqual(op.calls[0]["bearer"], _PRIMARY_KEY)

    def test_non_key_error_does_not_retry(self):
        for status in (400, 500):
            op = _ScriptedOpener([status])
            resp = _forward(op, _RecordingResolver())
            self.assertEqual(resp.status, status, status)
            self.assertEqual(len(op.calls), 1, status)

    def test_groq_keeps_single_key_path(self):
        op = _ScriptedOpener([429])
        resp = forward_same_wire(
            "groq", "/v1/chat/completions", b"{}",
            opener=op, key_resolver=lambda svc: "sk-test-fake-groq",
        )
        self.assertEqual(resp.status, 429)
        self.assertEqual(len(op.calls), 1)


class TestEmptyKeyContract(unittest.TestCase):
    def test_no_account_configured_fails_loud(self):
        op = _ScriptedOpener([200])
        with self.assertRaises(RuntimeError):
            _forward(op, _RecordingResolver(keys={}))
        self.assertEqual(len(op.calls), 0)  # never reached the opener

    def test_a_funded_failover_key_alone_is_not_used(self):
        """Inverted: this used to use the failover account when the primary was
        unconfigured. Now the primary's absence fails loud, and the second funded
        account is never consulted."""
        op = _ScriptedOpener([200])
        resolver = _RecordingResolver(keys={_FAILOVER_SVC: _FAILOVER_KEY})
        with self.assertRaises(RuntimeError):
            _forward(op, resolver)
        self.assertEqual(len(op.calls), 0)
        self.assertEqual(resolver.asked, [_PRIMARY_SVC])


class TestTheRotationModuleIsGone(unittest.TestCase):
    def test_key_failover_module_no_longer_exists(self):
        """Inverted from the cooldown test: the module that ordered the two funded
        accounts is deleted, so nothing can import and re-wire it by accident."""
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("lib.hal.key_failover")


if __name__ == "__main__":
    unittest.main()
