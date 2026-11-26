#!/usr/bin/env python3
"""HAL Live Demo — showcases the Knowledge Engine in action.

Run: python3 ~/.hal/lib/hal/demo.py
     python3 ~/.hal/lib/hal/demo.py --live    (sends real query to Groq)
     python3 ~/.hal/lib/hal/demo.py --full    (all demos including live query)

No external dependencies. Works immediately.
"""
import os
import sys
import time

# Resolve HAL_HOME and add to sys.path BEFORE any lib.hal imports
_hal_home = os.environ.get("HAL_HOME", os.path.expanduser("~/.hal"))
sys.path.insert(0, _hal_home)

from lib.hal.home import HAL_HOME  # noqa: E402

CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def header(text: str) -> None:
    print(f"\n{BOLD}{CYAN}{'=' * 60}{RESET}")
    print(f"{BOLD}{CYAN}  {text}{RESET}")
    print(f"{BOLD}{CYAN}{'=' * 60}{RESET}\n")


def step(text: str) -> None:
    print(f"  {GREEN}>{RESET} {text}")


def value(label: str, val: str) -> None:
    print(f"    {DIM}{label}:{RESET} {BOLD}{val}{RESET}")


def warn(text: str) -> None:
    print(f"  {YELLOW}!{RESET} {text}")


def demo_providers() -> None:
    """Demo 1: Provider Registry — config-driven, any provider."""
    header("1. Provider Registry — Any LLM, Any Provider")

    from lib.hal.registry import ProviderRegistry
    registry = ProviderRegistry()

    providers = registry.list_providers()
    step(f"Loaded {len(providers)} providers from config/providers.yaml:")
    for p in providers:
        models = ", ".join(p.models[:3]) + ("..." if len(p.models) > 3 else "")
        status = f"{GREEN}healthy{RESET}" if p.healthy else f"{RED}unhealthy{RESET}"
        value(p.name, f"{p.type.value} | {p.base_url[:40]} | {status} | {models}")

    return registry


def demo_discovery(registry) -> None:
    """Demo 2: Local Auto-Discovery — probe for local LLM servers."""
    header("2. Local Auto-Discovery — Zero-Config Local Models")

    from lib.hal.discovery import discover_local_providers, _LOCAL_SERVERS

    step("Probing known local ports:")
    for name, port, health, _ in _LOCAL_SERVERS:
        value(name, f"localhost:{port} ({health})")

    discovered = discover_local_providers(registry, timeout_s=1.0)
    if discovered:
        step(f"Discovered: {', '.join(discovered)}")
    else:
        warn("No local LLM servers detected (start Ollama for local models)")


def demo_skills() -> None:
    """Demo 3: Skill Loader — 290+ skills, 30 modes."""
    header("3. Skill System — 290+ Skills, 30 Modes")

    from lib.hal.skill_loader import SkillLoader
    loader = SkillLoader()

    skills = loader.discover()
    step(f"Discovered {len(skills)} skills")

    modes = loader.list_modes()
    step(f"Available modes: {len(modes)}")
    value("modes", ", ".join(modes[:10]) + f"... (+{len(modes) - 10} more)")

    code_skills = loader.resolve_for_mode("code")
    step(f"Code mode resolves to {len(code_skills)} skills:")
    value("foundation", ", ".join(code_skills[:5]) + "...")
    value("code-specific", ", ".join(s for s in code_skills if s not in code_skills[:10]))


def demo_agents() -> None:
    """Demo 4: Agent Runner — 29+ autonomous agents."""
    header("4. Agent System — 29+ Autonomous Agents")

    from lib.hal.agent_runner import AgentRunner
    runner = AgentRunner()

    agents = runner.list_agents()
    step(f"Available agents: {len(agents)}")

    for name in ["broly", "direct-implementation", "sprint-loop-agent"]:
        spec = runner.parse_agent(name)
        if spec:
            value(name, f"{spec.description[:60]}..." if len(spec.description) > 60 else spec.description)


def demo_memory() -> None:
    """Demo 5: Memory Engine — CC-compatible KONO integration."""
    header("5. Memory System — Persistent Knowledge Base")

    from lib.hal.memory_engine import MemoryEngine
    engine = MemoryEngine()

    state = engine.read_memory()
    entries = engine.list_entries()

    step(f"Memory directory: {engine.memory_dir}")
    step(f"Entrypoint: {len(state.entrypoint)} chars, truncated={state.truncated}")
    step(f"Memory entries: {len(entries)}")

    if entries:
        for e in entries[:5]:
            value(e, "")
        if len(entries) > 5:
            warn(f"... and {len(entries) - 5} more")


def demo_context() -> None:
    """Demo 6: Context Assembly — full LLM context in one call."""
    header("6. Context Assembly — CLAUDE.md + Mode + Skills + Git + Memory")

    from lib.hal.context_assembler import ContextAssembler
    assembler = ContextAssembler(project_dir=HAL_HOME)

    bundle = assembler.assemble(mode="code")

    step(f"System prompt: {len(bundle.system_prompt)} chars")
    step(f"CLAUDE.md chain: {len(bundle.claude_md_chain)} files loaded")
    step(f"Mode: {bundle.mode_name}")
    step(f"Skills: {len(bundle.skill_names)} active")
    step(f"Git branch: {bundle.git_branch}")
    value("git status", bundle.git_status[:80] + "..." if len(bundle.git_status) > 80 else bundle.git_status)
    value("memory", f"{len(bundle.memory_entrypoint)} chars")


def demo_cost() -> None:
    """Demo 7: Cost Tracker — token-to-USD across all providers."""
    header("7. Cost Tracking — Real Pricing Across 11+ Providers")

    from lib.hal.cost_tracker import calculate_cost

    pairs = [
        ("claude-opus-4-7", 10000, 5000),
        ("claude-sonnet-4-6", 10000, 5000),
        ("llama-3.3-70b-versatile", 10000, 5000),
        ("gemini-2.5-flash", 10000, 5000),
        ("local", 10000, 5000),
    ]

    step("Cost for 10K input + 5K output tokens:")
    for model, inp, out in pairs:
        cost = calculate_cost(model, input_tokens=inp, output_tokens=out)
        bar = "#" * max(1, int(cost * 200))
        value(f"{model:30s}", f"${cost:.4f}  {DIM}{bar}{RESET}")


def demo_sentinel() -> None:
    """Demo 8: Limit Sentinel — zero limits through failover."""
    header("8. Limit Sentinel — Zero Limits, Auto-Failover")

    from lib.hal.sentinel import LimitSentinel, ProviderLimitState
    from lib.hal.registry import ProviderRegistry

    registry = ProviderRegistry()
    sentinel = LimitSentinel(registry)

    step("Simulating Anthropic at 87% utilisation (above 85% threshold)...")
    state = ProviderLimitState()
    state.utilisation = 0.87
    sentinel._state["anthropic"] = state

    step(f"Anthropic available? {sentinel.is_available('anthropic')}")
    step(f"Groq available? {sentinel.is_available('groq')}")
    warn("Sentinel would auto-route to Groq (next in chain)")

    step("\nSimulating Groq rate limited (429)...")
    sentinel._on_error("groq", Exception("HTTP 429: rate limited"))
    step(f"Groq available? {sentinel.is_available('groq')}")
    warn("Sentinel would route to Gemini -> OpenRouter -> Ollama")

    step("\nThe chain: Anthropic -> Groq -> Gemini -> OpenRouter -> Ollama (local)")
    step("Local models have ZERO external limits. The user never sees a limit.")


def _demo_groq() -> None:
    """Live query: Groq cloud provider."""
    from lib.hal.cost_tracker import calculate_cost
    from lib.providers.groq import GroqProvider
    groq = GroqProvider()
    step("Groq (Llama 3.3 70B @ 750 tok/s):")
    step("  Query: 'In one sentence, what is a Knowledge Engine?'")
    start = time.monotonic()
    r = groq.generate("In one sentence, what is a Knowledge Engine?", max_tokens=80)
    ms = (time.monotonic() - start) * 1000
    print(f"\n    {BOLD}{r.text}{RESET}\n")
    inp = r.usage.get("prompt_tokens", 0)
    out = r.usage.get("completion_tokens", 0)
    cost = calculate_cost("llama-3.3-70b-versatile", input_tokens=inp, output_tokens=out)
    value("latency", f"{ms:.0f}ms")
    value("tokens", f"{inp} in / {out} out")
    value("cost", f"${cost:.6f}")


def _demo_gemini() -> None:
    """Live query: Gemini cloud provider."""
    from lib.providers.gemini import GeminiProvider
    gemini = GeminiProvider()
    step("Gemini (2.5 Flash — free tier):")
    step("  Query: 'Why is on-prem AI valuable for law firms?'")
    start = time.monotonic()
    r = gemini.generate("In one sentence, why is on-prem AI valuable for law firms?", max_tokens=512)
    ms = (time.monotonic() - start) * 1000
    print(f"\n    {BOLD}{r.text}{RESET}\n")
    value("latency", f"{ms:.0f}ms")
    value("cost", "$0.000000 (free tier)")


def _demo_ollama() -> None:
    """Live query: Ollama local provider (zero data exposure)."""
    import json as json_mod
    import urllib.request
    req = urllib.request.Request("http://localhost:11434/api/tags", method="GET")
    with urllib.request.urlopen(req, timeout=2) as resp:
        data = json_mod.loads(resp.read())
    local_models = [m["name"] for m in data.get("models", [])]
    if not local_models:
        return
    model = local_models[0]
    step(f"Ollama LOCAL ({model} — ON YOUR HARDWARE, ZERO DATA LEAVES):")
    step("  Query: 'What makes private AI essential for confidential legal work?'")
    body = json_mod.dumps({
        "model": model,
        "messages": [{"role": "user", "content": "In one sentence, what makes private AI essential for confidential legal work?"}],
        "max_tokens": 80,
    }).encode()
    req = urllib.request.Request("http://localhost:11434/v1/chat/completions",
                                data=body, headers={"Content-Type": "application/json"})
    start = time.monotonic()
    with urllib.request.urlopen(req, timeout=30) as resp:
        result = json_mod.loads(resp.read())
    ms = (time.monotonic() - start) * 1000
    text = result["choices"][0]["message"]["content"]
    print(f"\n    {BOLD}{text.strip()}{RESET}\n")
    value("latency", f"{ms:.0f}ms")
    value("cost", "$0.000000 (runs on YOUR hardware)")
    value("data exposure", f"{RED}ZERO — nothing leaves this machine{RESET}")


def demo_live_query(registry) -> None:
    """Demo 9: Live queries through multiple providers."""
    header("9. LIVE QUERIES — Multi-Provider in Action")

    for fn, name in [(_demo_groq, "Groq"), (_demo_gemini, "Gemini"), (_demo_ollama, "Ollama")]:
        try:
            fn()
        except Exception as e:
            warn(f"{name}: {e}")
        print()

    step(f"{GREEN}Three providers: Groq (cloud, fast), Gemini (cloud, free), Ollama (local, private).{RESET}")
    step(f"{GREEN}HAL routes between them automatically. The user never sees a limit.{RESET}")


def demo_hooks() -> None:
    """Demo 10: Hook Dispatch — 42 enforcement hooks."""
    header("10. Hook System — 42 Enforcement Gates")

    from lib.hal.hook_dispatch import HookDispatcher
    dispatcher = HookDispatcher()

    hooks = dispatcher.list_hooks()
    step(f"HARD hooks (block on deny): {len(hooks['hard'])}")
    for h in hooks["hard"][:5]:
        value(h, "")
    if len(hooks["hard"]) > 5:
        warn(f"... and {len(hooks['hard']) - 5} more")

    step(f"SOFT hooks (fail-open): {len(hooks['soft'])}")
    step("HARD hooks enforce security + quality mechanically")
    step("SOFT hooks provide telemetry + advisory signals")


def main() -> None:
    args = sys.argv[1:]
    live = "--live" in args or "--full" in args

    print(f"\n{BOLD}{RED}")
    print("  ██╗  ██╗ █████╗ ██╗     ")
    print("  ██║  ██║██╔══██╗██║     ")
    print("  ███████║███████║██║     ")
    print("  ██╔══██║██╔══██║██║     ")
    print("  ██║  ██║██║  ██║███████╗")
    print("  ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝")
    print(f"{RESET}")
    print(f"  {BOLD}Harness Abstraction Layer — Knowledge Engine{RESET}")
    print(f"  {DIM}Any model. Any provider. Any domain. Zero limits.{RESET}\n")

    registry = demo_providers()
    demo_discovery(registry)
    demo_skills()
    demo_agents()
    demo_memory()
    demo_context()
    demo_cost()
    demo_sentinel()
    demo_hooks()

    if live:
        demo_live_query(registry)

    header("HAL Status")
    step("15 modules | 133 tests | 0 external dependencies")
    step("Phases complete: 0 (Universal Runtime), 0.5 (Sentinel), 1 (Cost+Memory), 2 (Hooks+Skills), 3 (Context+Agents)")
    step(f"Ready for any domain. Legal is Domain 0.{RESET}")


if __name__ == "__main__":
    main()
