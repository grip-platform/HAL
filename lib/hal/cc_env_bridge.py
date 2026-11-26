"""HAL CC Environment Bridge — enable CC hidden features when running inside Claude Code.

When HAL detects it's running inside Claude Code (via harness_detect), it can
set environment variables to unlock gated features. These features exist in
the CC binary but are hidden behind env var checks.

When running standalone, these have no effect — HAL implements the features natively.

Usage:
    from lib.hal.cc_env_bridge import enable_cc_features, get_recommended_env

    # Check what can be enabled
    recommendations = get_recommended_env()

    # Enable all recommended features
    enable_cc_features()
"""
import os

# Features safe to enable — documented, no side effects, improve experience
_RECOMMENDED = {
    "ENABLE_TOOL_SEARCH": "true",
    "ENABLE_SESSION_PERSISTENCE": "true",
    "CLAUDE_CODE_ENABLE_TASKS": "true",
    "CLAUDE_CODE_RESUME_INTERRUPTED_TURN": "true",
    "CLAUDE_CODE_EFFORT_LEVEL": "max",
    "CLAUDE_CODE_VERIFY_PLAN": "true",
    "CLAUDE_CODE_NO_FLICKER": "1",
    "DISABLE_TELEMETRY": "1",
    "DISABLE_ERROR_REPORTING": "1",
    "CLAUDE_CODE_DISABLE_FEEDBACK_SURVEY": "1",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "true",
}

# Features that need user consent — behaviour changes
_OPTIONAL = {
    "CLAUDE_CODE_COORDINATOR_MODE": ("true", "Multi-agent orchestration"),
    "CLAUDE_CODE_PROACTIVE": ("true", "Agent takes initiative when idle"),
    "CLAUDE_CODE_BRIEF": ("true", "Compressed output for background ops"),
    "CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS": ("true", "Agent team features"),
    "EMBEDDED_SEARCH_TOOLS": ("true", "Embedded web search"),
    "ENABLE_LSP_TOOL": ("true", "Language Server Protocol for code intelligence"),
    "CLAUDE_CODE_STREAMING_OUTPUT": ("true", "Streamlined output format"),
}

# Features that should NEVER be auto-enabled — security sensitive
_NEVER_AUTO = {
    "CLAUDE_CODE_UNDERCOVER",  # Stealth mode — ethical concerns
    "CLAUDE_CODE_DISABLE_COMMAND_INJECTION_CHECK",  # Security bypass
}


def enable_cc_features(include_optional: bool = False) -> dict[str, str]:
    """Enable recommended CC hidden features via environment variables.

    Returns dict of {var: value} that were set. Any variable listed in
    ``_NEVER_AUTO`` is refused mechanically — even if a future diff
    accidentally moves it into ``_RECOMMENDED`` or ``_OPTIONAL``, the
    set operation is skipped. The list-level convention is backed up by
    a runtime gate so a typo cannot ship a security bypass.
    """
    enabled: dict[str, str] = {}

    for var, value in _RECOMMENDED.items():
        if var in _NEVER_AUTO:
            continue
        if var not in os.environ:
            os.environ[var] = value
            enabled[var] = value

    if include_optional:
        for var, (value, _desc) in _OPTIONAL.items():
            if var in _NEVER_AUTO:
                continue
            if var not in os.environ:
                os.environ[var] = value
                enabled[var] = value

    return enabled


def get_recommended_env() -> list[dict[str, str]]:
    """Get list of recommended env vars with descriptions."""
    result = []
    descriptions = {
        "ENABLE_TOOL_SEARCH": "Dynamic tool discovery — find tools by description",
        "ENABLE_SESSION_PERSISTENCE": "Sessions persist across restarts",
        "CLAUDE_CODE_ENABLE_TASKS": "Background task management",
        "CLAUDE_CODE_RESUME_INTERRUPTED_TURN": "Auto-resume after interruption",
        "CLAUDE_CODE_EFFORT_LEVEL": "Max-effort reasoning by default (ceiling thinking budget every turn)",
        "CLAUDE_CODE_VERIFY_PLAN": "Plan verification before execution",
        "CLAUDE_CODE_NO_FLICKER": "Virtualised viewport renderer — no flickering, mouse support, constant memory",
    }
    for var, value in _RECOMMENDED.items():
        current = os.environ.get(var, "")
        result.append({
            "variable": var,
            "value": value,
            "description": descriptions.get(var, ""),
            "currently_set": bool(current),
            "current_value": current,
        })
    return result


def get_optional_env() -> list[dict[str, str]]:
    """Get list of optional env vars with descriptions."""
    return [
        {"variable": var, "value": val, "description": desc,
         "currently_set": bool(os.environ.get(var, ""))}
        for var, (val, desc) in _OPTIONAL.items()
    ]


def is_inside_cc() -> bool:
    """Check if HAL is running inside Claude Code."""
    from lib.hal.harness_detect import detect_harness, Harness
    return detect_harness() == Harness.CLAUDE_CODE


# Z.ai Anthropic-compatible proxy — routes Claude Code to GLM models
# via coding subscription. Same API key as keychain:grip-zhipu.
_ZAI_PROXY_ENV = {
    "ANTHROPIC_BASE_URL": "https://api.z.ai/api/anthropic",
    "API_TIMEOUT_MS": "3000000",
}

# Z.ai model mapping: Claude Code model tiers -> GLM models
_ZAI_MODEL_MAP = {
    # glm-5.3, not glm-5.1 (GRIP#9431, 2026-11-24): Zhipu's GLM Coding Plan has
    # answered every `glm-5.1` request with GLM-5.3 since at least 2026-11-24 (two
    # probes, each beside a glm-5.3 control, both HTTP 200 with the reply's `model`
    # field reading glm-5.3). Naming the model that really answers; behaviour is
    # unchanged, since glm-5.1 already routed here.
    "ANTHROPIC_DEFAULT_OPUS_MODEL": "glm-5.3",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "glm-4.7",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "glm-4.5-air",
}


def enable_zai_proxy(api_key: str = "", model_map: dict[str, str] | None = None) -> dict[str, str]:
    """Enable Z.ai Anthropic-compatible proxy for Claude Code.

    When active, Claude Code routes to GLM models via Z.ai instead of
    Anthropic. Uses the coding subscription endpoint (prompt-counted).

    Returns dict of env vars that were set. Call disable_zai_proxy() to revert.
    """
    if not api_key:
        import subprocess
        try:
            result = subprocess.run(
                ["security", "find-generic-password", "-w", "-s", "grip-zhipu"],
                capture_output=True, text=True, timeout=5,
            )
            api_key = result.stdout.strip() if result.returncode == 0 else ""
        except (subprocess.TimeoutExpired, OSError):
            pass

    if not api_key:
        return {}

    enabled: dict[str, str] = {}

    # Save original values for restore
    for var in ("ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL", "API_TIMEOUT_MS"):
        orig = os.environ.get(var, "")
        if orig:
            os.environ[f"_HAL_ORIG_{var}"] = orig

    os.environ["ANTHROPIC_AUTH_TOKEN"] = api_key
    enabled["ANTHROPIC_AUTH_TOKEN"] = api_key[:8] + "..."

    for var, value in _ZAI_PROXY_ENV.items():
        os.environ[var] = value
        enabled[var] = value

    mapping = model_map or _ZAI_MODEL_MAP
    for var, model in mapping.items():
        os.environ[var] = model
        enabled[var] = model

    return enabled


def disable_zai_proxy() -> None:
    """Revert Z.ai proxy — restore original Anthropic env vars."""
    for var in ("ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL", "API_TIMEOUT_MS",
                *_ZAI_MODEL_MAP.keys()):
        orig_key = f"_HAL_ORIG_{var}"
        if orig_key in os.environ:
            os.environ[var] = os.environ.pop(orig_key)
        elif var in os.environ:
            del os.environ[var]
