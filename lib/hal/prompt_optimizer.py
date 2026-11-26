"""HAL Prompt Optimizer — reduce token usage without losing meaning.

Optimizes prompts by removing redundancy, compressing whitespace,
and shortening verbose instructions. Saves cost on every query.

Usage:
    from lib.hal.prompt_optimizer import PromptOptimizer
    opt = PromptOptimizer()
    result = opt.optimize("Please can you help me to write a function...")
"""
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class OptimizeResult:
    original_chars: int
    optimized_chars: int
    text: str

    @property
    def savings_pct(self) -> float:
        if self.original_chars == 0:
            return 0.0
        return (1 - self.optimized_chars / self.original_chars) * 100


# Verbose phrases that can be shortened
_VERBOSE_MAP: dict[str, str] = {
    "please can you": "",
    "could you please": "",
    "i would like you to": "",
    "i want you to": "",
    "can you help me": "",
    "please help me": "",
    "in order to": "to",
    "as well as": "and",
    "due to the fact that": "because",
    "at this point in time": "now",
    "in the event that": "if",
    "for the purpose of": "to",
    "with regard to": "about",
    "in addition to": "besides",
    "a large number of": "many",
    "the majority of": "most",
}


class PromptOptimizer:
    """Reduces prompt verbosity to save tokens."""

    def optimize(self, text: str) -> OptimizeResult:
        original = len(text)
        result = text
        result = _remove_verbose(result)
        result = _compress_whitespace(result)
        result = _remove_filler(result)
        return OptimizeResult(original, len(result), result.strip())


def _remove_verbose(text: str) -> str:
    result = text
    for verbose, short in _VERBOSE_MAP.items():
        result = re.sub(re.escape(verbose), short, result, flags=re.IGNORECASE)
    return result


def _compress_whitespace(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def _remove_filler(text: str) -> str:
    fillers = [
        r"\bbasically\b", r"\bactually\b", r"\bjust\b",
        r"\breally\b", r"\bvery\b", r"\bquite\b",
    ]
    for filler in fillers:
        text = re.sub(filler + r"\s*", "", text, flags=re.IGNORECASE)
    return text
