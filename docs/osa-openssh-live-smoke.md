# osa OpenSSHTransport — live-smoke runbook (H143)

Executes a trivial AppleScript on a paired Mac via the OpenSSH transport
shipped in γ-1b. Use this to validate that the argv shape + auth + remote
osascript pipeline still works after any transport-layer change.

## When to run

- After editing `lib/hal/osa/transport.py` (any change to argv, auth flags, or timeout logic)
- Before merging a PR that touches the transport abstraction
- Monthly regression check against the paired Mac fleet

The offline suite (`tests/test_osa_transport.py::OpenSSHArgvShapeH143`)
pins the synthetic argv shape — if those 9 tests pass and this
live-smoke passes against one Mac, H163 is CONFIRMED for that host.

## Prerequisites

- **Tailscale up** on both ends (the live-smoke uses direct OpenSSH; Tailscale is for connectivity).
- **SSH key registered** on the remote — the key referenced by `HAL_LIVE_SMOKE_KEY` must appear in `~/.ssh/authorized_keys` on the peer.
- **Accessibility + Full Disk Access** granted to `osascript` on the remote Mac — required for any System Events / UI-scripting probe (trivial `return "live-ok"` does not need Accessibility, but follow-on tests that probe UI will).
- **Remote user can run `osascript`** — the default user on macOS can; verify with `ssh $HOST osascript -e 'return 42'` before running the suite.

## Run

```bash
export HAL_LIVE_SMOKE_HOST=100.64.0.10      # Tailscale IP or hostname
export HAL_LIVE_SMOKE_USER=toopie               # remote login user
export HAL_LIVE_SMOKE_KEY=~/.ssh/grip-remote    # private key path

PYTHONPATH=$HAL_HOME python3.11 -m pytest \
    tests/test_osa_transport_live.py -v
```

Without those env vars, the test silently skips (no false positives in
CI — the suite is opt-in).

## Expected output

```
tests/test_osa_transport_live.py::OpenSSHLiveSmoke::test_echo_applescript_returns_live_ok PASSED
tests/test_osa_transport_live.py::OpenSSHLiveSmoke::test_timeout_propagates_on_slow_script PASSED
```

Two tests:

1. **test_echo_applescript_returns_live_ok** — canonical one-liner.
   Pipes `return "live-ok"` over stdin, asserts stdout contains
   `live-ok`. If this fails, the shape of the error tells you which
   layer broke:
   - `SSH authentication failed` → key not authorised on peer
   - `Connection refused` → peer's ssh daemon not running / firewall
   - `osascript: command not found` → remote is not a Mac
   - `execution error` in stderr → AppleScript engine ran but choked

2. **test_timeout_propagates_on_slow_script** — sends a 10-second
   `delay` with `timeout_s=2` and asserts `OsaTimeout` propagates. If
   this fails with a late success, the timeout pipeline is broken and
   long-running dispatches will hang the caller.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Skipped: `HAL_LIVE_SMOKE_HOST not set` | No env vars | Export the three HAL_LIVE_SMOKE_* vars |
| `identity_file not found` | Key path doesn't exist | Verify `ls -l $HAL_LIVE_SMOKE_KEY` |
| `rc=255, stderr='Host key verification failed'` | Remote host key not in known_hosts | `ssh-keyscan $HAL_LIVE_SMOKE_HOST >> ~/.ssh/known_hosts` |
| `rc=255, stderr='Permission denied (publickey)'` | Key not authorised on remote | Add pubkey to peer's `~/.ssh/authorized_keys` |
| `rc=127, stderr='osascript: command not found'` | Remote is not a Mac | Point HAL_LIVE_SMOKE_HOST at a macOS peer |

## Hypothesis linkage

**H163** — "OpenSSHTransport.execute correctly composes the argv shape
(identity + user + BatchMode + IdentitiesOnly + ConnectTimeout +
ControlMaster mux + stdin-piped osascript body) against a mocked
subprocess, and the same command is runnable by the operator against a
paired Mac." Deadline: 2026-05-19.

- **Offline confirmation**: `tests/test_osa_transport.py::OpenSSHArgvShapeH143` (9 tests)
- **Live confirmation**: this runbook's two tests passing against a paired Mac
- **Falsified if**: the live-smoke fails with an argv-level error that the offline suite didn't catch (missing flag, wrong order, incorrect stdin handling)

Log successful live runs somewhere durable (operator notebook or a
GitHub issue thread on the HAL repo) so the 30-day H163 window has
evidence at verification time.
