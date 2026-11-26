"""Regression tests for the spec-example version check (HAL#584).

The prose layer is the one layer `_check_version_consistency` never reads. It
receives `02-markdown-body.md` in `regions` and inspects only the bash prelude
and the Python runtime, so nothing has ever verified what the documentation
says about versions.

That blind spot is not hypothetical. The spec sat two releases stale: it told
readers the runtime accepted only `happi/1.0` through `happi/1.3` and always
emitted `happi/1.3`, while the envelope table fourteen lines below listed all
six versions correctly. The worked example showed `happi/1.4` events with a
delta text of `happi/1.3`; the `idr`, `context` and `continuity` wire shapes all
stamped `happi/1.4`. Every version test was green throughout, because the check
that could have noticed was reading two layers out of five.

A stale example is not cosmetic: a reader who copies one produces events this
runtime never emits, and a conformance suite written from it tests a protocol
that does not exist.

THE GOODHART PROPERTY IS `test_inbound_envelopes_are_exempt`.

The check must fire ONLY on emitted-event examples. The spec deliberately
contains INBOUND envelope examples naming older versions -- the `cite.verify`
example reads `happi/1.4`, the dogfood envelope reads `happi/1.0` -- and both
are CORRECT, because demonstrating back-compat is the entire reason they are
there. A check keyed on "any happi/N literal in the body" would report them and
invite someone to "fix" the very thing they illustrate. Widening the regex to
match inbound envelopes makes that test fail; removing the check makes the
staleness tests fail.

WHY EXAMPLES RATHER THAN PROSE. The obvious check reads the prose sentence that
was wrong. That check would be an oracle: the fix removes the duplicated
sentence, so there would be nothing left to disagree with and it would pass
vacuously ever after (tests-must-be-able-to-fail.md shape 3). Examples must keep
existing and must keep being correct, so a check on them has real work to do at
every version bump.
"""
import importlib.util
from pathlib import Path

import pytest

from tests._public_tree import folder_absent, reason

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "build-happi.py"
# Public HAL leaves scripts/ out, so on a public checkout every test here is
# skipped with that reason (HAL#859); in the full repository they all run.
pytestmark = pytest.mark.skipif(folder_absent("scripts"), reason=reason("scripts"))


def _load():
    spec = importlib.util.spec_from_file_location("build_happi", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _regions(body, py_version="happi/1.5"):
    """Minimal layer set carrying only what the example check reads."""
    return {
        "02-markdown-body.md": body,
        "04-python-runtime.py": f'HAPPI_VERSION = "{py_version}"\n',
    }


def test_matching_examples_pass():
    """No complaint when the emitted-event examples stamp the runtime version."""
    mod = _load()
    body = '`{"v":"happi/1.5","id":...,"type":"idr","ts":...,"idr_ref":{...}}`.'
    assert mod._check_spec_examples(_regions(body)) == []


def test_catches_the_real_pre_fix_staleness():
    """The actual HAL#584 regression: wire shapes left at happi/1.4."""
    mod = _load()
    body = '`{"v":"happi/1.4","id":...,"type":"continuity","ts":...,"c_ref":{...}}`.'
    failures = mod._check_spec_examples(_regions(body))
    assert len(failures) == 1
    assert "happi/1.4" in failures[0]
    assert "happi/1.5" in failures[0]


def test_inbound_envelopes_are_exempt():
    """THE GOODHART ANCHOR. An older version on an INBOUND envelope is correct.

    Both of these live in the real spec and both must stay: the `cite.verify`
    example shows a v1.4 client, and the runtime accepting it is the point.
    """
    mod = _load()
    body = (
        'echo \'{"v":"happi/1.4","id":"cv-1","cmd":"cite.verify","flags":{}}\'\n'
        'stdin:  {"v":"happi/1.0","id":"req-001","cmd":"version"}\n'
    )
    assert mod._check_spec_examples(_regions(body)) == []


def test_multiline_dogfood_envelope_is_exempt():
    """The document declares ITSELF a happi/1.0 envelope. That is deliberate."""
    mod = _load()
    body = '{\n  "v": "happi/1.0",\n  "id": "happi-md-canonical",\n  "cmd": "spec.describe"\n}\n'
    assert mod._check_spec_examples(_regions(body)) == []


def test_a_mixed_body_reports_only_the_stale_emitted_event():
    """Closest to the real file: correct inbound examples beside a stale event.

    Guards the case a naive implementation gets wrong in the damaging direction
    -- reporting the inbound envelopes alongside the genuine fault, so whoever
    acts on the report breaks two correct examples while fixing one.
    """
    mod = _load()
    body = (
        'stdin:  {"v":"happi/1.0","id":"r","cmd":"version"}\n'
        'stdout: {"v":"happi/1.4","id":"r","type":"started","ts":0}\n'
        '`{"v":"happi/1.5","id":...,"type":"reasoning","ts":...,"r_ref":{...}}`.\n'
    )
    failures = mod._check_spec_examples(_regions(body))
    assert len(failures) == 1
    assert "happi/1.4" in failures[0]
    assert "happi/1.0" not in failures[0]


def test_absent_body_is_not_a_pass_by_accident():
    """No prose layer means nothing to check -- and nothing claimed either."""
    mod = _load()
    assert mod._check_spec_examples({"04-python-runtime.py": 'HAPPI_VERSION = "happi/1.5"\n'}) == []


def test_missing_declaration_defers_to_the_version_check():
    """With no HAPPI_VERSION there is no truth to compare against.

    Reporting here would duplicate `_check_version_consistency`, which already
    returns "no HAPPI_VERSION declaration found in any layer" -- two checks
    reporting one fault reads as two faults.
    """
    mod = _load()
    body = '`{"v":"happi/1.4","id":...,"type":"idr","ts":...}`.'
    assert mod._check_spec_examples({"02-markdown-body.md": body}) == []
