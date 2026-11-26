#!/usr/bin/env bash
# Create HAL's test venv with Python 3.11+.
# HAL_TEST_PYTHON may be a space-separated command, for example "py -3".
set -euo pipefail

VENV_DIR="${HAL_TEST_VENV:-$HOME/.hal-test-venv}"
PY_CMD=()

_version_ok() {
    local out major minor
    out="$("$@" -c 'import sys; print(sys.version_info[0], sys.version_info[1])' 2>/dev/null)" || return 1
    major="${out%% *}"
    minor="${out##* }"
    [ "$major" = "3" ] && [ "${minor:-0}" -ge 11 ] 2>/dev/null
}

_find_python() {
    if [ -n "${HAL_TEST_PYTHON:-}" ]; then
        local override=()
        read -r -a override <<< "$HAL_TEST_PYTHON"
        if _version_ok "${override[@]}"; then
            PY_CMD=("${override[@]}")
            return 0
        fi
    fi

    local candidate
    for candidate in python3 python; do
        if _version_ok "$candidate"; then
            PY_CMD=("$candidate")
            return 0
        fi
    done

    if _version_ok py -3; then
        PY_CMD=(py -3)
        return 0
    fi
    return 1
}

_venv_python() {
    if [ -x "$VENV_DIR/bin/python" ]; then
        VENV_PYTHON="$VENV_DIR/bin/python"
        return 0
    fi
    if [ -x "$VENV_DIR/Scripts/python.exe" ]; then
        VENV_PYTHON="$VENV_DIR/Scripts/python.exe"
        return 0
    fi
    return 1
}

if [ "${1:-}" = "python" ]; then
    shift
    if ! _venv_python; then
        printf '%s\n' "test-venv.sh: no venv at $VENV_DIR yet -- run 'bash bin/test-venv.sh' first to create it" >&2
        exit 1
    fi
    exec "$VENV_PYTHON" "$@"
fi

if [ "$#" -ne 0 ]; then
    printf '%s\n' "test-venv.sh: usage: bash bin/test-venv.sh [python [args...]]" >&2
    exit 2
fi

if ! _find_python; then
    printf '%s\n' "test-venv.sh: no Python 3.11+ found (tried \$HAL_TEST_PYTHON, python3, python, py -3) -- install Python 3.11 or newer from https://www.python.org/downloads/, then re-run this script" >&2
    exit 1
fi

VENV_EXISTED=0
if _venv_python; then
    VENV_EXISTED=1
else
    "${PY_CMD[@]}" -m venv "$VENV_DIR"
fi
if ! _venv_python; then
    printf '%s\n' "test-venv.sh: virtual environment at $VENV_DIR has no executable Python after creation" >&2
    exit 1
fi
if [ "$VENV_EXISTED" -eq 1 ] && ! _version_ok "$VENV_PYTHON"; then
    printf '%s\n' "test-venv.sh: virtual environment at $VENV_DIR does not use Python 3.11+; delete $VENV_DIR and run this script again" >&2
    exit 1
fi
"$VENV_PYTHON" -m pip install -q pytest pyyaml cryptography apsw xxhash
