"""HAL Response Formatter — format LLM responses for different targets.

Adapts LLM responses for different output contexts: terminal (ANSI),
plain text, markdown, HTML, and JSON. Handles code block styling,
emphasis, and structural formatting.

Usage:
    from lib.hal.response_formatter import ResponseFormatter

    fmt = ResponseFormatter()
    terminal_output = fmt.for_terminal(response_text)
    plain = fmt.for_plain(response_text)
    html = fmt.for_html(response_text)
"""
import re


# ANSI escape codes for terminal formatting
_BOLD = "\033[1m"
_DIM = "\033[2m"
_ITALIC = "\033[3m"
_RESET = "\033[0m"
_CYAN = "\033[36m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"

_CODE_BLOCK_RE = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`([^`]+)`")
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_HEADER_RE = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)


class ResponseFormatter:
    """Formats LLM responses for different output targets."""

    def for_terminal(self, text: str) -> str:
        """Format for terminal with ANSI colours."""
        result = _CODE_BLOCK_RE.sub(
            lambda m: f"\n{_DIM}```{m.group(1)}{_RESET}\n{_CYAN}{m.group(2)}{_RESET}{_DIM}```{_RESET}\n",
            text,
        )
        result = _INLINE_CODE_RE.sub(f"{_CYAN}\\1{_RESET}", result)
        result = _BOLD_RE.sub(f"{_BOLD}\\1{_RESET}", result)
        result = _HEADER_RE.sub(f"{_BOLD}{_GREEN}\\2{_RESET}", result)
        return result

    def for_plain(self, text: str) -> str:
        """Strip all formatting to plain text."""
        result = _CODE_BLOCK_RE.sub(lambda m: m.group(2), text)
        result = _INLINE_CODE_RE.sub("\\1", result)
        result = _BOLD_RE.sub("\\1", result)
        result = _HEADER_RE.sub("\\2", result)
        return result

    def for_html(self, text: str) -> str:
        """Convert markdown-style formatting to HTML."""
        result = _escape_html(text)
        result = re.sub(
            r"```(\w*)\n(.*?)```",
            lambda m: f"<pre><code class=\"language-{m.group(1)}\">{m.group(2)}</code></pre>",
            result, flags=re.DOTALL,
        )
        result = re.sub(r"`([^`]+)`", r"<code>\1</code>", result)
        result = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", result)
        result = re.sub(r"^### (.+)$", r"<h3>\1</h3>", result, flags=re.MULTILINE)
        result = re.sub(r"^## (.+)$", r"<h2>\1</h2>", result, flags=re.MULTILINE)
        result = re.sub(r"^# (.+)$", r"<h1>\1</h1>", result, flags=re.MULTILINE)
        result = result.replace("\n\n", "</p><p>")
        return f"<p>{result}</p>"

    def for_json(self, text: str) -> dict:
        """Parse response into structured JSON sections."""
        sections: list[dict] = []
        current_text: list[str] = []

        for line in text.split("\n"):
            code_match = re.match(r"^```(\w*)$", line)
            if code_match and not any(s.get("type") == "code" and not s.get("closed") for s in sections):
                if current_text:
                    sections.append({"type": "text", "content": "\n".join(current_text)})
                    current_text = []
                sections.append({"type": "code", "language": code_match.group(1), "content": "", "closed": False})
            elif line == "```" and sections and sections[-1].get("type") == "code" and not sections[-1].get("closed"):
                sections[-1]["closed"] = True
            elif sections and sections[-1].get("type") == "code" and not sections[-1].get("closed"):
                sections[-1]["content"] += line + "\n"
            else:
                current_text.append(line)

        if current_text:
            sections.append({"type": "text", "content": "\n".join(current_text)})

        # Clean up internal flags
        for s in sections:
            s.pop("closed", None)
            if s["type"] == "code":
                s["content"] = s["content"].rstrip("\n")

        return {"sections": sections, "section_count": len(sections)}


def _escape_html(text: str) -> str:
    """Escape HTML entities."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
