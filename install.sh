#!/usr/bin/env bash
# HAL — unified installer.
#
# Bootstraps ~/.hal with the Harness Abstraction Layer runtime:
#   1. Clone or update the HAL repo at $HAL_HOME (default: ~/.hal)
#   2. Create a dedicated Python virtualenv at $HAL_HOME/venv
#   3. Install KONO dependencies (xxhash, apsw, sqlite-vec, sqlite-lembed)
#      with visible error output — no silent pip failures
#   4. Symlink bin/hal into /usr/local/bin (optional, asks if sudo needed)
#   5. Create default identity and provider config if missing
#   6. Optionally install the model warmup launchd/systemd service
#
# Usage:
#   curl -sSL https://raw.githubusercontent.com/CodeTonight-SA/HAL/main/install.sh | bash
#   bash install.sh                 # interactive
#   bash install.sh --with-warmup   # also install warmup service
#   bash install.sh --no-symlink    # skip /usr/local/bin symlink
#   bash install.sh --update        # fast path: update repo + venv only
#
# Requires: git, python3 >= 3.11, curl (for remote install only).

set -euo pipefail

HAL_HOME="${HAL_HOME:-$HOME/.hal}"
HAL_REPO="${HAL_REPO:-https://github.com/grip-platform/HAL.git}"
HAL_BRANCH="${HAL_BRANCH:-main}"
VENV_DIR="$HAL_HOME/venv"
SYSTEM_BIN="${HAL_SYSTEM_BIN:-/usr/local/bin}"

# CLI flags
WITH_WARMUP=0
NO_SYMLINK=0
UPDATE_ONLY=0
VENV_FRESH=0   # set by ensure_venv when it builds a venv, which then holds no packages
for arg in "$@"; do
    case "$arg" in
        --with-warmup) WITH_WARMUP=1 ;;
        --no-symlink)  NO_SYMLINK=1 ;;
        --update)      UPDATE_ONLY=1 ;;
        --help|-h)
            sed -n "2,20p" "$0" | sed "s/^# \{0,1\}//"
            exit 0
            ;;
        *)
            echo "Unknown flag: $arg (use --help)" >&2
            exit 2
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Colour + logging helpers
# ---------------------------------------------------------------------------

if [ -t 1 ]; then
    C_RESET="\033[0m"; C_DIM="\033[2m"; C_GREEN="\033[32m"
    C_YELLOW="\033[33m"; C_RED="\033[31m"; C_CYAN="\033[36m"
else
    C_RESET=""; C_DIM=""; C_GREEN=""; C_YELLOW=""; C_RED=""; C_CYAN=""
fi

log_step() { printf "%b==>%b %s\n" "$C_CYAN" "$C_RESET" "$1"; }
log_ok()   { printf "%b  ok%b  %s\n" "$C_GREEN" "$C_RESET" "$1"; }
log_warn() { printf "%b warn%b %s\n" "$C_YELLOW" "$C_RESET" "$1"; }
log_fail() { printf "%b fail%b %s\n" "$C_RED" "$C_RESET" "$1" >&2; }

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------

#   HAL's library needs Python >= 3.11 (lib/hal/utils/cch.py imports
#   tomllib, added in 3.11 with no fallback -- CodeTonight-SA/HAL#922, #924).
check_prereqs() {
    log_step "Checking prerequisites"
    local missing=()
    command -v git     >/dev/null 2>&1 || missing+=("git")
    command -v python3 >/dev/null 2>&1 || missing+=("python3")
    if [ ${#missing[@]} -gt 0 ]; then
        log_fail "Missing required tools: ${missing[*]}"
        log_fail "Install them via your package manager and re-run."
        exit 1
    fi

    local py_version
    py_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    local py_major py_minor
    py_major=$(printf "%s" "$py_version" | cut -d. -f1)
    py_minor=$(printf "%s" "$py_version" | cut -d. -f2)
    if [ "$py_major" -lt 3 ] || { [ "$py_major" -eq 3 ] && [ "$py_minor" -lt 11 ]; }; then
        log_fail "python3 >= 3.11 required, found $py_version"
        exit 1
    fi
    log_ok "git, python3 $py_version"
}

# ---------------------------------------------------------------------------
# Repo clone / update
# ---------------------------------------------------------------------------

clone_or_update_repo() {
    if [ -d "$HAL_HOME/.git" ]; then
        log_step "Updating HAL at $HAL_HOME"
        ( cd "$HAL_HOME" && git fetch --quiet origin "$HAL_BRANCH" \
            && git checkout --quiet "$HAL_BRANCH" \
            && git pull --quiet --rebase --autostash origin "$HAL_BRANCH" ) \
            && log_ok "Repo up to date" \
            || log_warn "Git update failed — continuing with existing checkout"
    else
        log_step "Cloning HAL into $HAL_HOME"
        git clone --quiet --branch "$HAL_BRANCH" "$HAL_REPO" "$HAL_HOME" \
            || { log_fail "Clone failed from $HAL_REPO"; exit 1; }
        log_ok "Cloned $HAL_REPO#$HAL_BRANCH"
    fi
}

# ---------------------------------------------------------------------------
# Virtualenv + KONO deps
# ---------------------------------------------------------------------------

# check_prereqs only checks the SYSTEM python3; a venv built before the
# floor was raised to 3.11 (HAL#922, #924) would otherwise be kept forever
# on its own stale interpreter, since nothing here looked at what was
# already there. ensure_venv checks the venv's own Python and rebuilds it
# when it predates the floor, moving the old one aside rather than
# deleting it.
ensure_venv() {
    log_step "Creating Python virtualenv at $VENV_DIR"
    if [ -x "$VENV_DIR/bin/python3" ]; then
        local venv_version venv_major venv_minor
        venv_version=$("$VENV_DIR/bin/python3" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || true)
        venv_major="${venv_version%%.*}"
        venv_minor="${venv_version#*.}"
        if [ -z "$venv_version" ] || [ "$venv_major" -lt 3 ] || { [ "$venv_major" -eq 3 ] && [ "$venv_minor" -lt 11 ]; }; then
            local old_venv="$VENV_DIR.pre-3.11-$(date +%Y%m%d-%H%M%S)"
            mv "$VENV_DIR" "$old_venv"
            log_warn "existing venv was Python ${venv_version:-unreadable} (< 3.11) -- moved aside to $old_venv, rebuilding"
        fi
    fi

    if [ ! -x "$VENV_DIR/bin/python3" ]; then
        python3 -m venv "$VENV_DIR" || {
            log_fail "venv creation failed at $VENV_DIR"
            exit 1
        }
        VENV_FRESH=1
        log_ok "venv created"
    else
        log_ok "venv already exists"
    fi

    # Upgrade pip — pinning to the venvs pip keeps the rest of the install
    # deterministic even on systems with an ancient pip in the stdlib venv.
    "$VENV_DIR/bin/pip3" install --quiet --upgrade pip \
        || log_warn "pip upgrade failed (non-fatal)"
}

install_kono_deps() {
    log_step "Installing KONO dependencies (xxhash, apsw, sqlite-vec, sqlite-lembed)"
    local pip_log="$HAL_HOME/.install-pip.log"
    mkdir -p "$(dirname "$pip_log")"

    # Core deps — required for signing and memory substrate.
    local core_deps="xxhash apsw sqlite-vec"
    if "$VENV_DIR/bin/pip3" install $core_deps > "$pip_log" 2>&1; then
        log_ok "core: $core_deps"
    else
        log_fail "Core KONO deps install FAILED. See $pip_log"
        log_fail "--- last 40 lines ---"
        tail -n 40 "$pip_log" >&2
        log_fail "Resolve the errors above and re-run. HAL core needs these."
        exit 1
    fi

    # Optional tier — sqlite-lembed frequently has no prebuilt wheel on
    # macOS arm64 / ppc64le. Failure degrades features but does not block HAL.
    local optional_deps="sqlite-lembed"
    if "$VENV_DIR/bin/pip3" install $optional_deps >> "$pip_log" 2>&1; then
        log_ok "optional: $optional_deps"
    else
        log_warn "optional ($optional_deps) install failed — HAL will skip local embedding."
        log_warn "Details: tail -n 20 $pip_log"
    fi
}

# ---------------------------------------------------------------------------
# Bootstrap config + identity
# ---------------------------------------------------------------------------

bootstrap_dirs() {
    log_step "Ensuring directory structure"
    local dirs=(config skills modes agents facts data state/hal memory lib/hal/providers)
    for d in "${dirs[@]}"; do
        mkdir -p "$HAL_HOME/$d"
    done
    log_ok "directories ready"
}

create_defaults() {
    local identity="$HAL_HOME/facts/identity.md"
    if [ ! -f "$identity" ]; then
        log_step "Creating default identity"
        cat > "$identity" <<IDENTITY
# HAL Identity

## Configuration

\`\`\`yaml
name: "User"
signature: "U>>"
mode: "solo"
role: "Knowledge Worker"
created: "$(date +%Y-%m-%d)"
\`\`\`
IDENTITY
        log_ok "facts/identity.md"
    fi

    local providers="$HAL_HOME/config/providers.yaml"
    if [ ! -f "$providers" ]; then
        log_step "Creating default provider config"
        cat > "$providers" <<PROVIDERS
providers:
  - name: ollama
    type: local
    base_url: http://localhost:11434/v1
    detect: true
    models: auto

routing:
  default: fixed
  default_model: ollama/llama3
PROVIDERS
        log_ok "config/providers.yaml"
    fi
}

# ---------------------------------------------------------------------------
# CLI symlink
# ---------------------------------------------------------------------------

install_symlink() {
    if [ "$NO_SYMLINK" -eq 1 ]; then
        return 0
    fi
    local target="$SYSTEM_BIN/hal"
    if [ -L "$target" ] && [ "$(readlink "$target")" = "$HAL_HOME/bin/hal" ]; then
        log_ok "symlink $target -> $HAL_HOME/bin/hal"
        return 0
    fi
    log_step "Installing symlink $target -> $HAL_HOME/bin/hal"
    if [ -w "$SYSTEM_BIN" ]; then
        ln -sf "$HAL_HOME/bin/hal" "$target" && log_ok "symlink installed"
    else
        log_warn "$SYSTEM_BIN not writable — skipping symlink"
        log_warn "Add to PATH manually: export PATH=\"$HAL_HOME/bin:\$PATH\""
    fi
}

# ---------------------------------------------------------------------------
# Warmup service (optional)
# ---------------------------------------------------------------------------

install_warmup() {
    if [ "$WITH_WARMUP" -ne 1 ]; then
        return 0
    fi
    local setup="$HAL_HOME/scripts/hal-warmup-setup.sh"
    if [ ! -x "$setup" ]; then
        log_warn "Warmup setup script not executable: $setup"
        return 0
    fi
    log_step "Installing model warmup service"
    HAL_HOME="$HAL_HOME" "$setup" && log_ok "warmup service installed"
}

# ---------------------------------------------------------------------------
# Status summary
# ---------------------------------------------------------------------------

print_summary() {
    printf "\n%bHAL ready%b at %s\n" "$C_GREEN" "$C_RESET" "$HAL_HOME"
    printf "%bvenv%b   %s\n" "$C_DIM" "$C_RESET" "$VENV_DIR"
    printf "%bCLI%b    %s\n" "$C_DIM" "$C_RESET" "$HAL_HOME/bin/hal"
    printf "\nNext steps:\n"
    printf "  1. Start interactive chat:  %shal%s\n" "$C_CYAN" "$C_RESET"
    printf "  2. Start web + API server:  %shal --serve%s\n" "$C_CYAN" "$C_RESET"
    printf "  3. Show providers status:   %shal --status%s\n" "$C_CYAN" "$C_RESET"
    if ! command -v ollama >/dev/null 2>&1; then
        printf "\n%bHint%b: install ollama for a zero-cost local fallback: https://ollama.ai\n" "$C_YELLOW" "$C_RESET"
    fi
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

main() {
    printf "%bHAL — Harness Abstraction Layer%b\n" "$C_CYAN" "$C_RESET"
    printf "%binstaller%b  home=%s  branch=%s\n\n" "$C_DIM" "$C_RESET" "$HAL_HOME" "$HAL_BRANCH"

    check_prereqs
    clone_or_update_repo
    ensure_venv

    if [ "$UPDATE_ONLY" -eq 1 ]; then
        # A venv ensure_venv has just built (one older than the floor that it
        # moved aside, or none at all) holds no packages yet.
        if [ "$VENV_FRESH" -eq 1 ]; then
            install_kono_deps
        fi
        log_ok "Update complete (--update mode)"
        print_summary
        return 0
    fi

    install_kono_deps
    bootstrap_dirs
    create_defaults
    install_symlink
    install_warmup
    print_summary
}

main "$@"
