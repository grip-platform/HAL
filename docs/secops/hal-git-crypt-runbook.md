# HAL git-crypt runbook — encrypting exposed PII (task #522)

## Why this exists

During the 2026-07-10 → 2026-07-17 window, a Guild member held V>>-granted
write access to the private HAL repository. The forensic investigation found
**no evidence the access was used** and **no secrets or API keys anywhere in the
HAL tree** (HAL references credentials through the macOS keychain, never inline —
so there is nothing to rotate). The **only** exposure was personal information
sitting in plaintext: HAL had **zero encryption**, while GRIP already encrypts
this same class of file.

This change closes that gap: it adds git-crypt to HAL and encrypts the five
tracked personal-information files.

## What is encrypted (and what is deliberately not)

**Encrypted** (`.gitattributes` filters):

| File | Contents |
|------|----------|
| `facts/cv.md` | A person's CV |
| `facts/team.md` | Team roster |
| `facts/identity.md` | Operator name / signature / role |
| `facts/people.md` | People facts (mostly template) |
| `config/peers.json` | Human-Channel-Gateway peer registry (names + contact handles) |

**Left in plaintext on purpose** — `config/providers.yaml`, `config/model_catalog.yaml`
and the rest of `config/`: these are runtime configuration (not personal data) that
every HAL machine reads at boot. Encrypting them would need the key present on
every machine before anything works, for no privacy benefit. They contain no
secrets (keychain-referenced).

## ⚠️ CRITICAL — do this BEFORE you merge

`lib/hal/identity_provider.py` reads **`facts/identity.md` at runtime** (a fallback
path when identity env vars are absent). If the branch is merged and a machine
pulls the encrypted `facts/identity.md` **without the git-crypt key**, that machine
sees ciphertext and HAL's identity resolution can break.

So the key must be provisioned on **every machine that runs HAL** — at minimum the
VPS (`grip-remote-v2`), and your other working machines (blackbook) — **before**
this branch is merged.

## The key

- Location on void: `~/CodeTonight/.hal-git-crypt.key` (symmetric, mode `600`,
  **outside** any repository — same pattern as the exampleorg-financials key).
- It is **not** in the repo and never will be (git-crypt keeps its live key in
  `.git/`, which is untracked).
- **Back it up now** to your password manager / secure storage. If this file is
  lost and no machine holds an unlocked copy, the encrypted files become
  unrecoverable.

## Unlock a machine (void, blackbook, VPS)

Copy `~/CodeTonight/.hal-git-crypt.key` to the machine (securely — e.g. over
Tailscale `scp`, not a channel), then:

```bash
cd ~/.hal          # or the HAL checkout on that machine
git-crypt unlock /path/to/.hal-git-crypt.key
```

After unlock, `git-crypt status -e` should list the five files as `encrypted` and
the working-tree copies read as normal plaintext.

### VPS specifically

```bash
# from void, copy the key to the VPS over Tailscale
scp ~/CodeTonight/.hal-git-crypt.key grip@100.64.0.10:~/.hal-git-crypt.key
ssh grip@100.64.0.10 'chmod 600 ~/.hal-git-crypt.key && cd ~/.hal && git-crypt unlock ~/.hal-git-crypt.key && git-crypt status -e | head'
```

Only after the VPS (and any other HAL machine) is unlocked should this branch be
merged.

## Verify

```bash
git-crypt status -e        # lists the 5 files as encrypted
git show HEAD:facts/identity.md | head -1   # WITHOUT the key → GITCRYPT binary header (ciphertext)
                                            # WITH the key    → readable plaintext
```

## Rollback (fully reversible)

git-crypt is reversible. To undo entirely: remove the five filter lines from
`.gitattributes`, run `git-crypt unlock`, re-add the files (they commit back as
plaintext), and drop the `.git-crypt/` directory. Nothing about this change is
destructive.

## Provenance

- Task: `#522` — INT-SECOPS Alex-access hardening (reversible).
- Memory: `alex-bunardzik-caution` (closed verdict: interpersonal conflict, not
  proven IP theft; do not accuse).
- Secret rotation: **not required** — no plaintext secrets were found in the HAL
  tree; exposure was personal-information only.
