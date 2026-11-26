# Gateway account switch: measured evidence for HAL #678

The gateway now uses `account_switch.choose` before an Anthropic inference turn,
then `account_switch.observe` and one replacement attempt after a usage-limit 429.
All three inference rails share this boundary. The replacement changes only the
bearer, passes the spend guard, and records its label and billing rank. Auxiliary
and account-bound routes do not enter the switch. The existing off marker applies.

The original gaps are recorded in [HAL #678](https://github.com/CodeTonight-SA/HAL/issues/678):
one production primary (`gateway_failover.py:303`), fixed retries including usage
limits (`gateway.py:595-635`), successful-overage-only spend observation
(`spend_guard.py:386`), and 429-to-503 translation (`gateway.py:687`). These are the
orchestrator's baseline references; line numbers describe the pre-change code.

## Retry and replay boundaries

Plain throttles honor seconds or HTTP-date Retry-After, with a 10-second total
retry-sleep budget. This permits two of the plan's five-second waits without an
unbounded pause. A delay exceeding the remaining budget is not shortened: the
429 is returned instead. Missing/invalid Retry-After uses the existing backoff.
The existing retry count still applies; 5xx and 529 retain their original policy.
Usage-limit 429s use the exact predicate shared with `account_switch.observe`
and are never retried on the rejected account.

The switch runs before a response reader reaches the handler. Reading or failing
a returned stream cannot re-enter it. Incomplete client bodies are rejected
before OpenAI translation can discard their original length. An unbuffered body
never enters the account switch. A replacement gets one attempt; its transport
failure or 400 restores the original rejection without modifying thinking blocks.
A third available account is not falsely declared exhausted when that single
replay allowance runs out: its capacity remains for the next client request.

PR #684 (Step 5f) was already merged at the initial fetch. The existing
`FailoverProxy` promotes the terminal spend refusal to `AccountsSpent`; the
unchanged chat/Responses handlers try their provider cascade, then the existing
HAL #667 final notice. The cascade remains governed by its existing opt-in and
spend policy. No cascade transport or handler code was edited. Open PR #634's
same-wire reply lines and #617's subscription retry files were left untouched.

## Real interactive Codex, pinned 0.156.1

Installed with:

```sh
npm install --prefix .pty-tools --ignore-scripts @anthropic-ai/claude-code@2.1.288 @openai/codex@0.156.1
WIRE_MATRIX_CLAUDE_BIN="$PWD/.pty-tools/node_modules/.bin/claude" \
WIRE_MATRIX_CODEX_BIN="$PWD/.pty-tools/node_modules/.bin/codex" \
"$PY" tools/wire-matrix/pty_driver.py --harness codex --out .pty-runs/c15
```

`PY` denotes the run's mandated venv interpreter. Every server used an isolated
scratch port. The PTY gateway now calls `build_failover_proxy` with one primary
and the fake pool through `account_switch`. It does not seed S8's spend ledger.
The only retry override is three for S3's overload scenario.

| Scenario | HAL verdict | Direct control | HAL elapsed |
|---|---|---|---|
| C10: ten turns, tool use, A rejects turn ten | PASS | FAIL as required: usage-limit notice, no final answer | 13.46 s |
| S1: A limited, B healthy | PASS | FAIL as required: usage-limit notice, no answer | 3.06 s |
| S3: three overloads then success | PASS | FAIL as required: visible 529 | 6.55 s |
| S8: both accounts spent | PASS: one final message naming B's earlier reset | FAIL as required: only A's later reset named | 2.97 s |

All eight expected outcomes passed. The sanitized machine verdicts are in
[the PTY record](2026-10-07-gateway-account-switch-pty.json).
C10/S1/S8 reported clean byte equality across the switch, no paid credential,
and no request in cooldown. S3 reported no paid credential; switch equality and
cooldown were not measured because no account switch or cooldown occurred.
A second full run (`.pty-runs/c15-final`) produced the same eight outcomes.

The direct controls are not the only negative evidence: temporarily disabling
`account_switch.choose` made the actual HAL S1 PTY run FAIL (`expected=False`,
3.14 s). Restoring it returned S1 to PASS in the second full run.

## Unit tests and deliberate failures

The new file is explicitly bound into `.github/workflows/pytest-gateway-audit.yml`,
including its trigger paths. The loopback tests use fake credentials only.
The gateway workflow's complete test list was run together with
`tests/test_account_switch.py` and `tests/test_wire_matrix_pty.py`. The exact final
summary was **529 passed, 2 subtests passed in 28.04s**. The captured summary and PTY output are embedded in
[the citation evidence artifact](2026-10-07-gateway-account-switch-proof.html).

```sh
"$PY" -m pytest tests/test_gateway_account_switch.py tests/test_gateway_cascade.py -q
# 59 passed (after the final third-account change)
```

Fourteen temporary code mutations produced nineteen failing unit cases, covering
every new case and the changed cascade fixture. Files were restored byte-for-byte
before the green reruns:

| Broken behavior | Failing cases |
|---|---:|
| Retry a usage-limit response on the same account | 1 |
| Stop observing the shared quota-header predicate | 2 |
| Replace Retry-After with fixed backoff | 1 |
| Remove the throttle wait bound | 1 |
| Disable account replacement | 1 |
| Permit the paid credential in the spend guard | 1 |
| Remove the inference-route gate | 4 |
| Remove the buffered-body gate | 1 |
| Ignore the switch off marker | 1 |
| Duplicate an already successful stream request | 1 |
| Return the final limit response instead of the last rung | 2 |
| Ignore Retry-After in the existing cascade fixture | 1 |
| Accept a truncated client body | 1 |
| Treat the one-replay limit as full pool exhaustion | 1 |

Removing only the adapter's duplicate paid check did not turn its test red,
because the existing transport guard still refused the credential. Disabling the
canonical guard did turn it red; this is the reported paid-guard mutation.
The stream test guards the absence of replay from a returned response reader;
its mutation duplicates that successful request. The PTY run did not exercise
S7/S10; this report makes no new claim about their interactive behavior.

The live gateway on port 4010 runs this code only after the operator restarts it.
This run did not restart, signal, bind or reconfigure any live service.
