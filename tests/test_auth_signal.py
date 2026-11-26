"""auth_signal: what a 401 or 403 says about its account. Pure; made-up bodies only.

No real Anthropic OAuth 401 or 403 body is known to HAL (the module docstring says
why), so each body here is written to exercise one class of the classifier, not to
stand for a real answer.
"""
import gzip
import json

import pytest

from lib.hal import auth_signal as sig


def body(message, error_type="permission_error"):
    return json.dumps({"type": "error",
                       "error": {"type": error_type, "message": message}}).encode()


@pytest.mark.parametrize("message, word", [
    ("OAuth token has been revoked.", "revoked"),
    ("This organization has been disabled.", "disabled"),
    ("Your account has been suspended pending review.", "suspended"),
    ("This account is banned.", "banned"),
    ("Access terminated.", "terminated"),
    ("Usage Policy violation.", "violation"),
    ("ACCOUNT SUSPENDED", "suspended"),
])
def test_a_message_naming_an_enforcement_action_is_enforcement(message, word):
    signal = sig.classify(403, body(message))
    assert (signal.kind, signal.word, signal.enforcement) == (sig.ENFORCEMENT, word, True)
    assert signal.error_type == "permission_error"


def test_a_401_naming_a_revocation_is_enforcement_too():
    signal = sig.classify(401, body("OAuth token revoked", "authentication_error"))
    assert (signal.status, signal.kind, signal.error_type) == (
        401, sig.ENFORCEMENT, "authentication_error")


def test_an_expired_token_is_an_expiry_not_enforcement():
    signal = sig.classify(401, body("OAuth token has expired. Please obtain a new token.",
                                    "authentication_error"))
    assert (signal.kind, signal.word, signal.enforcement) == (sig.EXPIRY, "expired", False)


def test_a_message_naming_both_reads_as_enforcement():
    """A stop that was not needed costs one command; a missed one is the failure."""
    signal = sig.classify(401, body("token expired or revoked", "authentication_error"))
    assert (signal.kind, signal.word) == (sig.ENFORCEMENT, "revoked")


@pytest.mark.parametrize("status, payload", [
    (401, body("invalid x-api-key", "authentication_error")),
    (401, body("Invalid bearer token", "authentication_error")),
    (403, body("Your API key does not have permission to use the specified resource.")),
    (403, body("")),
    (403, b"account disabled"),            # not JSON: only error.message is read
    (403, b""),
    (403, b'{"error": "suspended"}'),      # error is not an object
    (403, b'{"error": {"message": ["suspended"]}}'),  # message is not a string
    (403, b'{"type": "error", "message": "suspended"}'),  # no error object at all
])
def test_anything_else_is_unknown(status, payload):
    signal = sig.classify(status, payload)
    assert (signal.kind, signal.word, signal.enforcement) == (sig.UNKNOWN, None, False)


def test_words_are_matched_whole_never_inside_another_word():
    """'disabledness' is not 'disabled'; a word list is a list of words."""
    assert sig.classify(403, body("the disabledness flag")).kind == sig.UNKNOWN


def test_a_gzipped_body_is_read_through_its_content_encoding():
    packed = gzip.compress(body("Your account has been suspended."))
    signal = sig.classify(403, packed, [("Content-Encoding", "gzip")])
    assert signal.kind == sig.ENFORCEMENT


@pytest.mark.parametrize("error_type, kept", [
    ("permission_error", "permission_error"),
    ("authentication_error", "authentication_error"),
    ("made_up_error_type_with_text_in_it", "other"),
    (None, "none"),
    (["permission_error"], "other"),
])
def test_only_a_documented_error_type_is_kept(error_type, kept):
    payload = json.dumps({"error": {"type": error_type, "message": "suspended"}}).encode()
    assert sig.classify(403, payload).error_type == kept


def test_nothing_from_the_body_reaches_the_judgement_but_a_listed_word():
    """Custody: the message is never kept. Every field the caller gets comes from
    a closed list, so a row built from it cannot carry text from the body."""
    secret = "private-detail-1234"
    signal = sig.classify(403, body(f"Account {secret} suspended", f"{secret}_error"))
    assert secret not in json.dumps(signal.fields())
    assert signal.fields() == {"status": 403, "signal": sig.ENFORCEMENT,
                               "error_type": "other", "word": "suspended"}
