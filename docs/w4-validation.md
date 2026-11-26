# W4 validation record

Run locally on 2026-10-07 with the operator-specified Python interpreter.
`TEST_PYTHON` below denotes that interpreter; no real provider or credential was
used. The wire-matrix measurement used loopback servers on OS-assigned ports;
unit tests used fake in-process transports. No live service was started/stopped.

## Passing checks

The exact gateway workflow file list, including the new test file:

```sh
"$TEST_PYTHON" -m pytest tests/test_audit_log_isolation.py \
  tests/test_hal_audit_contamination.py \
  tests/test_hal_audit_rotate.py \
  tests/test_gateway_failover.py \
  tests/test_gateway_cascade.py \
  tests/test_gateway_any_harness_failover.py \
  tests/test_gateway_messages_witness.py \
  tests/test_gateway_picker_alias.py \
  tests/test_gateway_crosswire_wiring.py \
  tests/test_gateway_error_source.py \
  tests/test_gateway_cli.py \
  tests/test_gateway_refusal_drains_body.py \
  tests/test_gateway_idempotent_bind.py \
  tests/test_gateway.py \
  tests/test_gateway_openai.py \
  tests/test_gateway_responses.py \
  tests/test_gateway_broken_pipe.py \
  tests/test_gateway_health_brake.py \
  tests/test_gateway_egress.py \
  tests/test_gateway_guard.py \
  tests/test_gateway_guard_enhanced.py -q --disable-warnings
```

Result: **469 passed, 2 subtests passed** (19.25 s). The run reported 174
pre-existing pytest temporary-directory cleanup warnings; the warnings concern
old temporary files, not assertions. The full run's stdout is retained locally.

Focused cascade suite: `"$TEST_PYTHON" -m pytest tests/test_gateway_cascade.py -q
--disable-warnings` -- **41 passed**, after restoration of every mutation.

Final spend-guard path and RAILLM composition check: `"$TEST_PYTHON" -m pytest
tests/test_spend_guard_paths.py tests/test_raillm.py tests/test_raillm_kimi_rung.py
tests/test_raillm_pin_primary.py -q --disable-warnings` -- **97 passed** (5.94 s).

Earlier core regression set (cascade, failover, OpenAI, Responses, spend-guard
paths) -- **172 passed** before the seven additional preservation/measurement
cases were added. RAILLM core, Kimi-rung and pin-primary tests -- **46 passed**.

## Falsification

`docs/w4-cascade-red-runs.json` maps each temporary production-code mutation to
its failing test cases. **23 mutations; all 41 new test cases failed at least
once.** Each mutated source file was restored byte-for-byte in a `finally`
block. The restored cascade suite passed, then the full gateway workflow passed.
These are observed pytest failures, not tests that assert a mutation flag.

The mutations remove the all-accounts-spent marker, import boundary, billing
check, window check, conservative byte bound, uncertain-input refusal, tool and
additional-tools checks, cooldown, request/reply normalization, audit provider,
non-2xx decline, safe-connect classification, credential retry, last-rung return,
503 prohibition, hidden-history/non-text-output refusals, streamed tool-call
mapping, measurement reserve and no-replay stream boundary.

## Baseline failures kept separate

`tests/test_raillm_ladder.py` has four stale model-list assertions: the fixture
expects the old Opus rung and no Fable rung. All four failed both with this
branch's type refactor and with the original HEAD versions of `raillm.py` and
`raillm_ladder.py` temporarily restored. The other test in that file passed.
No claim is made that this pre-existing test file passes; its expectations were
not changed by W4. The passing 469-test command does not include it.

## Scope and external dependencies

The real Codex CLI 0.156.1 completed one isolated GPT-6 Astra turn through the
gateway and the S0 stub. `docs/w4-codex-measurement.json` records the counts.
Its custom/namespace `additional_tools` are deliberately unsupported, so this
measured request cannot cascade. GRIP preload and production long-context turns
were not measured. The shipped target list is empty.

HAL#678 owns gateway account-switch integration; HAL#676 remains open for the
Step 7 cascade end-to-end work. GRIP#7720 tracks the fresh-thread Claude Code
launcher/store move. No review or merge is part of this run.

## Concurrent PRs

The current diffs of HAL#634 and HAL#617 were read with `gh pr diff`. #634
changes the same gateway file but different functions/lines (same-wire usage
accounting); it also changes gateway_witness and pytest-happi, untouched here.
A raw `git apply --check` fails in those three files on both this branch and
pristine origin/main copies, demonstrating existing context drift. A conflict
is therefore possible and #634 must be reconciled against current main; this
is not a claim of merge compatibility. #617 changes only cch_retry/subscription,
has no edited-file overlap, and its raw diff passes `git apply --check` here.

The proof renderer checked four bound citations against captured test summaries
and the Codex measurement: 4 exact matches, 0 missing.
`docs/w4-proof-provenance.json` retains their source hashes and offsets. This
is citation grounding, not an independent assertion of runtime correctness.
