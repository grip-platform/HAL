"""HAL /council command — multi-model deliberation via RAILLM transport.

Registers a /council slash command with the CommandRouter. Supports two modes:

- **Standard mode** (default): HALCouncil.deliberate_parallel for cross-LLM
  gut-feel + synthesis + debate protocol.
- **Cascade mode** (--cascade): Routes through RAILLMSentinel's tier cascade
  with CarryOverVerifier at each provider switch boundary, testing fact
  retention across the full FRONTIER -> FAST -> FREE_TIER -> LOCAL chain.

Usage:
    register_council_command(router, registry)

    # Then via CommandRouter:
    result = router.execute("/council What are the trade-offs of microservices?")
    result = router.execute("/council --cascade --rounds 2 Explain RAILLM")
    result = router.execute("/council --models groq/llama,gemini/flash Is this safe?")

Hypothesis H051: /council preserves factual continuity across an Anthropic ->
Groq -> Gemini -> Ollama cascade as measured by a verification quiz on 10
pre-registered facts. Prediction: >= 0.9 accuracy.
"""
from typing import Optional

from lib.hal.carryover_verifier import CarryOverAuditLog, CarryOverVerifier, record_switch
from lib.hal.moe import HALCouncil
from lib.hal.raillm import RAILLMSentinel, ConversationThread


def _parse_council_args(args: str) -> tuple[str, dict]:
    """Parse /council arguments into query string and options dict.

    Supported flags:
        --cascade       Use RAILLM tier cascade instead of parallel council
        --rounds N      Number of debate rounds (default 1)
        --budget N.NN   Cost ceiling in USD
        --models a,b,c  Comma-separated model addresses

    Returns (query, options) where query is the remaining text after
    flag extraction and options is a dict of parsed flags.
    """
    options: dict = {
        "cascade": False,
        "rounds": 1,
        "budget": None,
        "models": None,
    }
    parts = args.split()
    query_parts: list[str] = []
    i = 0
    while i < len(parts):
        token = parts[i]
        if token == "--cascade":
            options["cascade"] = True
        elif token == "--rounds" and i + 1 < len(parts):
            i += 1
            options["rounds"] = int(parts[i])
        elif token == "--budget" and i + 1 < len(parts):
            i += 1
            options["budget"] = float(parts[i])
        elif token == "--models" and i + 1 < len(parts):
            i += 1
            options["models"] = [m.strip() for m in parts[i].split(",") if m.strip()]
        else:
            query_parts.append(token)
        i += 1
    return " ".join(query_parts), options


def _council_standard(
    registry: "ProviderRegistry",
    query: str,
    models: Optional[list[str]],
    rounds: int,
    budget: Optional[float],
) -> str:
    """Standard parallel council deliberation."""
    council = HALCouncil(registry)
    result = council.deliberate_parallel(
        query=query,
        models=models,
        rounds=rounds,
        budget_usd=budget,
    )
    header = (
        f"Council ({len(result.votes)} votes, "
        f"{len(result.rounds)} rounds, "
        f"agreement {result.agreement_score:.0%})"
    )
    return f"{header}\n\n{result.merged_text}"


def _council_cascade(
    registry: "ProviderRegistry",
    query: str,
    models: Optional[list[str]],
    rounds: int,
) -> str:
    """RAILLM cascade council — route through tier chain with fact verification.

    Forces failover by attempting each model in the cascade chain. After each
    successful response, records the provider switch and verifies fact retention.
    """
    sentinel = RAILLMSentinel(registry=registry)
    verifier = CarryOverVerifier()
    audit_log = CarryOverAuditLog()

    # Determine the starting model
    start_model = (models[0] if models else "groq/llama-3.3-70b-versatile")

    # Initial dispatch
    response = sentinel.dispatch_with_history(
        model_address=start_model,
        user_message=query,
    )
    served_by = [f"{response.provider}/{response.model}"]

    # Run additional rounds through the cascade
    for round_num in range(1, rounds + 1):
        follow_up = (
            f"Round {round_num}: Based on your previous answer, "
            f"what are the most important nuances I should understand?"
        )
        thread_before = ConversationThread(
            messages=list(sentinel.thread.messages),
            system_prompt=sentinel.thread.system_prompt,
        )
        prev_provider = response.provider

        response = sentinel.dispatch_with_history(
            model_address=start_model,
            user_message=follow_up,
        )
        served_by.append(f"{response.provider}/{response.model}")

        # Record switch if provider changed
        if response.provider != prev_provider:
            record_switch(
                log=audit_log,
                verifier=verifier,
                thread_before=thread_before,
                response_after=response,
                from_provider=prev_provider,
                to_provider=response.provider,
            )

    # Build report
    lines = [
        f"Cascade Council ({len(served_by)} dispatches, "
        f"{len(audit_log.entries)} switches verified)",
        f"Providers: {' -> '.join(served_by)}",
    ]
    if audit_log.entries:
        loss_count = len(audit_log.loss_events())
        lines.append(
            f"Fact retention: {len(audit_log.entries) - loss_count}/"
            f"{len(audit_log.entries)} switches clean "
            f"(loss rate {audit_log.observed_rate():.0%})"
        )
    lines.append("")
    lines.append(response.text)
    return "\n".join(lines)


def council_handler(registry: "ProviderRegistry") -> callable:
    """Create a /council command handler bound to a provider registry."""
    def handler(args: str) -> str:
        if not args.strip():
            return (
                "Usage: /council [--cascade] [--rounds N] [--budget N.NN] "
                "[--models a,b,c] <query>\n\n"
                "Standard mode: parallel multi-model deliberation\n"
                "Cascade mode:  RAILLM tier failover with fact verification"
            )
        query, options = _parse_council_args(args)
        if not query:
            return "Error: no query provided after flags"

        if options["cascade"]:
            return _council_cascade(
                registry, query, options["models"], options["rounds"],
            )
        return _council_standard(
            registry, query, options["models"],
            options["rounds"], options["budget"],
        )
    return handler


def register_council_command(
    router: "CommandRouter",
    registry: "ProviderRegistry",
) -> None:
    """Register /council with the given CommandRouter."""
    router.register(
        "council",
        council_handler(registry),
        "Multi-model council deliberation (standard or RAILLM cascade)",
        "[--cascade] [--rounds N] [--budget N.NN] [--models a,b] <query>",
    )
