"""HAL Export — multi-format conversation export.

Exports conversations as markdown, JSON, HTML, or plain text for sharing,
archiving, or feeding into other tools.

Usage:
    from lib.hal.export import ConversationExporter

    exporter = ConversationExporter()
    exporter.to_markdown(history, "output.md", title="Bug Fix Session")
    html = exporter.to_html_string(history, title="Code Review")
"""
import json
import time

from lib.hal.types import ChatMessage


def to_markdown(
    history: list[ChatMessage], path: str,
    title: str = "HAL Conversation", model: str = "",
) -> None:
    """Export conversation as markdown."""
    user_turns = sum(1 for m in history if m.role == "user")
    exported_at = time.strftime("%Y-%m-%d %H:%M")
    lines = [
        f"# {title}\n",
        f"*Exported: {exported_at} | {user_turns} turns*\n",
    ]
    if model:
        lines.append(f"*Model: {model}*\n")
    lines.append("---\n")

    role_headers = {"user": "## User", "assistant": "## Assistant", "system": "## System"}
    for msg in history:
        header = role_headers.get(msg.role, f"## {msg.role.title()}")
        lines.append(f"{header}\n\n{msg.content}\n")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def to_json(
    history: list[ChatMessage], path: str,
    title: str = "HAL Conversation", model: str = "",
) -> None:
    """Export conversation as structured JSON."""
    data = {
        "title": title,
        "exported_at": time.strftime("%Y-%m-%d %H:%M"),
        "turn_count": sum(1 for m in history if m.role == "user"),
        "model": model,
        "messages": [{"role": m.role, "content": m.content} for m in history],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def to_text(history: list[ChatMessage], path: str) -> None:
    """Export conversation as plain text."""
    lines = [f"[{m.role.upper()}]\n{m.content}\n" for m in history]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def to_html_string(
    history: list[ChatMessage],
    title: str = "HAL Conversation",
) -> str:
    """Export conversation as an HTML string (for embedding/sharing)."""
    user_turns = sum(1 for m in history if m.role == "user")
    exported_at = time.strftime("%Y-%m-%d %H:%M")

    msg_blocks = "\n".join(
        f'<div class="message {m.role}">'
        f'<div class="role">{m.role}</div>'
        f'<div class="content">{_escape_html(m.content)}</div>'
        f'</div>'
        for m in history
    )

    return _HTML_TEMPLATE.format(
        title=_escape_html(title),
        exported_at=exported_at,
        turn_count=user_turns,
        messages=msg_blocks,
    )


def _escape_html(text: str) -> str:
    """Escape HTML special characters."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("\n", "<br>")
    )


_HTML_TEMPLATE = """<!DOCTYPE html>
<html><head>
<meta charset="UTF-8">
<title>{title}</title>
<style>
body {{ font-family: system-ui, sans-serif; max-width: 800px; margin: 2rem auto; padding: 0 1rem; background: #fafafa; }}
h1 {{ border-bottom: 2px solid #333; padding-bottom: 0.5rem; }}
.meta {{ color: #666; font-size: 0.9rem; margin-bottom: 2rem; }}
.message {{ margin: 1rem 0; padding: 1rem; border-radius: 8px; }}
.message.user {{ background: #e3f2fd; border-left: 4px solid #1976d2; }}
.message.assistant {{ background: #f3e5f5; border-left: 4px solid #7b1fa2; }}
.message.system {{ background: #fff3e0; border-left: 4px solid #f57c00; }}
.role {{ font-weight: bold; text-transform: uppercase; font-size: 0.8rem; margin-bottom: 0.5rem; }}
.content {{ line-height: 1.6; white-space: pre-wrap; }}
</style>
</head><body>
<h1>{title}</h1>
<div class="meta">Exported: {exported_at} | {turn_count} turns</div>
{messages}
</body></html>"""
