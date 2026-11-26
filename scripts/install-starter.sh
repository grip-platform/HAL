#!/usr/bin/env bash
# HAL Starter Kit -- coexistence-aware LOCAL installer.
#
# Installs the HAL framework from THIS kit directory into ~/.hal -- WITHOUT
# cloning from GitHub. It is the kit's own installer (analogous to the GRIP
# starter kit's scripts/install-starter.sh): the kit was already sanitised and
# leak-scanned at build time, so this script only places it.
#
# If a HAL install (or any other framework) is already present at the target,
# it merges: framework directories are copied in, and a framework file of the
# same name is REPLACED WITHOUT A BACKUP (config/ included, so a second run
# resets config/spend_guard.json). Only the top-level files and docs it would
# overwrite are backed up first (.pre-hal-<timestamp>). Run it once, for a first
# install. It then provisions the runtime the way install.sh does: a
# dedicated venv, the KONO dependencies (xxhash, apsw, sqlite-vec, optional
# sqlite-lembed via the exact pip steps from install.sh), the directory
# skeleton, and default facts/identity.md + config/providers.yaml if missing.
# Safe to read before running; intended to be run once.
#
#   bash scripts/install-starter.sh            # install into ~/.hal
#   HAL_HOME=/path bash scripts/install-starter.sh   # install elsewhere
set -euo pipefail

KIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HAL_HOME="${HAL_HOME:-$HOME/.hal}"
VENV_DIR="$HAL_HOME/venv"
TS="$(date +%Y%m%d-%H%M%S)"
PY="$(command -v python3 || command -v python || true)"

say() { printf '[hal-starter] %s\n' "$1"; }

if [ -z "$PY" ]; then
    say "ERROR: python3 not found -- required for the venv + KONO deps."
    exit 1
fi

# HAL's library needs Python >= 3.11 (lib/hal/utils/cch.py imports tomllib,
# added in 3.11 with no fallback -- CodeTonight-SA/HAL#922, #924). install.sh
# checks the same floor, but this kit does not ship install.sh, so the kit's own
# installer checks it before it copies anything or builds a venv.
check_python_floor() {
    local version major minor
    version="$("$1" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || true)"
    major="${version%%.*}"
    minor="${version#*.}"
    if [ -z "$version" ] || [ "$major" -lt 3 ] || { [ "$major" -eq 3 ] && [ "$minor" -lt 11 ]; }; then
        say "ERROR: Python >= 3.11 required, found ${version:-an unreadable version} ($1)."
        exit 1
    fi
}
check_python_floor "$PY"

# --- 1. detect an existing framework ---------------------------------------
COEXIST=0
if [ -d "$HAL_HOME" ] && [ -n "$(ls -A "$HAL_HOME" 2>/dev/null)" ]; then
    COEXIST=1
    say "existing framework detected at $HAL_HOME -- coexistence install"
else
    say "fresh install into $HAL_HOME"
    mkdir -p "$HAL_HOME"
fi

# --- 2. framework directories (additive: merge alongside what is there) ----
# These three lists match lib/hal_starter_kit.py's SHIP_DIRS, SHIP_FILES and
# SHIP_PATHS; tests/test_hal_starter_kit.py fails if they drift apart.
FRAMEWORK_DIRS="lib bin scripts skills agents modes config configs hooks \
happi-sources rules deploy"
for d in $FRAMEWORK_DIRS; do
    [ -d "$KIT_DIR/$d" ] || continue
    mkdir -p "$HAL_HOME/$d"
    cp -aR "$KIT_DIR/$d/." "$HAL_HOME/$d/"
done
say "framework directories installed"

# --- 3. top-level and nested files (back up any existing one on a coexistence
# install). The kit carries no hal-init.sh or install.sh: both clone a private
# repository, so this script is the kit's installer.
TOP_FILES="AGENTS.md CLAUDE.md CONTRIBUTING.md README.md HAL.md happi.md \
NOTICE LICENSE"
DOC_FILES="docs/GETTING-STARTED.md docs/SOVEREIGN-DEFAULT.md \
docs/infinite-machine-contract.md"
for f in $TOP_FILES $DOC_FILES; do
    [ -f "$KIT_DIR/$f" ] || continue
    mkdir -p "$(dirname "$HAL_HOME/$f")"
    if [ -e "$HAL_HOME/$f" ] && [ "$COEXIST" = 1 ]; then
        cp -a "$HAL_HOME/$f" "$HAL_HOME/$f.pre-hal-$TS"
    fi
    cp -a "$KIT_DIR/$f" "$HAL_HOME/$f"
done
say "top-level files and docs installed"

# --- 4. directory skeleton (mirrors install.sh bootstrap_dirs) -------------
for d in config skills modes agents facts data state/hal memory lib/hal/providers; do
    mkdir -p "$HAL_HOME/$d"
done
say "directory skeleton ready"

# --- 5. venv + KONO deps (exact pip steps from install.sh) ------------------
# An existing venv was kept with no check on its own Python, so a venv built
# before the floor was raised to 3.11 (tomllib, HAL#922/#924) stayed on the
# old interpreter forever. ensure_venv checks it and rebuilds when it is too
# old, moving the old one aside rather than deleting it -- the same way
# install.sh's own ensure_venv does.
ensure_venv() {
    if [ -x "$VENV_DIR/bin/python3" ]; then
        local venv_version venv_major venv_minor
        venv_version=$("$VENV_DIR/bin/python3" -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>/dev/null || true)
        venv_major="${venv_version%%.*}"
        venv_minor="${venv_version#*.}"
        if [ -z "$venv_version" ] || [ "$venv_major" -lt 3 ] || { [ "$venv_major" -eq 3 ] && [ "$venv_minor" -lt 11 ]; }; then
            local old_venv="$VENV_DIR.pre-3.11-$(date +%Y%m%d-%H%M%S)"
            mv "$VENV_DIR" "$old_venv"
            say "existing venv was Python ${venv_version:-unreadable} (< 3.11) -- moved aside to $old_venv, rebuilding"
        fi
    fi

    if [ ! -x "$VENV_DIR/bin/python3" ]; then
        say "creating Python virtualenv at $VENV_DIR"
        "$PY" -m venv "$VENV_DIR" || { say "ERROR: venv creation failed at $VENV_DIR"; exit 1; }
    else
        say "venv already exists at $VENV_DIR"
    fi

    "$VENV_DIR/bin/pip3" install --quiet --upgrade pip \
        || say "warn: pip upgrade failed (non-fatal)"
}
ensure_venv

PIP_LOG="$HAL_HOME/.install-pip.log"
CORE_DEPS="xxhash apsw sqlite-vec"
if "$VENV_DIR/bin/pip3" install --prefer-binary $CORE_DEPS > "$PIP_LOG" 2>&1; then
    say "core KONO deps installed: $CORE_DEPS"
else
    say "ERROR: core KONO deps install FAILED. See $PIP_LOG"
    tail -n 40 "$PIP_LOG" >&2
    say "Resolve the errors above and re-run. HAL core needs these."
    exit 1
fi

OPTIONAL_DEPS="sqlite-lembed"
if "$VENV_DIR/bin/pip3" install --prefer-binary $OPTIONAL_DEPS >> "$PIP_LOG" 2>&1; then
    say "optional dep installed: $OPTIONAL_DEPS"
else
    say "warn: optional ($OPTIONAL_DEPS) install failed -- HAL will skip local embedding."
fi

# --- 6. default identity + provider config (heredocs from install.sh) -------
IDENTITY="$HAL_HOME/facts/identity.md"
if [ ! -f "$IDENTITY" ]; then
    cat > "$IDENTITY" <<IDENTITY_EOF
# HAL Identity

## Configuration

\`\`\`yaml
name: "User"
signature: "U>>"
mode: "solo"
role: "Knowledge Worker"
created: "$(date +%Y-%m-%d)"
\`\`\`
IDENTITY_EOF
    say "facts/identity.md created"
fi

PROVIDERS="$HAL_HOME/config/providers.yaml"
if [ ! -f "$PROVIDERS" ]; then
    cat > "$PROVIDERS" <<PROVIDERS_EOF
providers:
  - name: ollama
    type: local
    base_url: http://localhost:11434/v1
    detect: true
    models: auto

routing:
  default: fixed
  default_model: ollama/llama3
PROVIDERS_EOF
    say "config/providers.yaml created"
fi

# --- 7. closing guidance ---------------------------------------------------
cat <<EOF

[hal-starter] HAL framework install complete at $HAL_HOME

Next steps:
  1. Start interactive chat:  $HAL_HOME/bin/hal
  2. Show providers status:   $HAL_HOME/bin/hal --status
  3. Edit your identity:      $HAL_HOME/facts/identity.md
  4. Add provider keys + routing in: $HAL_HOME/config/providers.yaml
     (a zero-cost local fallback works out of the box if ollama is installed:
      https://ollama.ai)
  5. Keys in the keychain, the gateway, and what ships switched off:
     $HAL_HOME/docs/GETTING-STARTED.md
EOF
if [ "$COEXIST" = 1 ]; then
    cat <<EOF

Coexistence notes:
  - Your previous framework's files were kept. Anything this kit would have
    overwritten was copied to <name>.pre-hal-$TS first.
  - If two frameworks ship a skill/agent/command of the same name, this kit's
    copy now sits on top; the originals are recoverable from your backup.
EOF
fi
