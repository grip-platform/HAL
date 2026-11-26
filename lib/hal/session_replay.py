"""HAL Session Replay — timeline model for agent session recording and review.

Records all conversation turns, tool calls, and decisions on a timeline.
Enables post-session review, annotation, and export. Adapted from Recordly's
timeline-as-declarative-transformation concept.

Usage:
    from lib.hal.session_replay import SessionRecorder, ReplayTimeline

    recorder = SessionRecorder()
    recorder.record_turn("user", "Fix the bug in auth.py")
    recorder.record_tool("Read", {"file": "auth.py"}, duration_ms=45)
    recorder.record_turn("assistant", "Found the issue on line 42...")

    timeline = recorder.to_timeline()
    print(timeline.summary)
    timeline.export_markdown("session-replay.md")
"""
import json
import os
import time
from dataclasses import dataclass, field

from lib.hal.home import HAL_HOME as _HAL_HOME

_REPLAYS_DIR = os.path.join(_HAL_HOME, "state", "hal", "replays")


@dataclass
class TimelineEvent:
    """A single event on the session timeline."""
    timestamp: float
    event_type: str  # turn, tool, decision, error, milestone
    role: str = ""  # user, assistant, system
    content: str = ""
    metadata: dict = field(default_factory=dict)
    duration_ms: int = 0
    annotation: str = ""

    @property
    def elapsed_s(self) -> float:
        return self.timestamp

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "role": self.role,
            "content": self.content[:500],
            "metadata": self.metadata,
            "duration_ms": self.duration_ms,
            "annotation": self.annotation,
        }


@dataclass
class ReplayTimeline:
    """Complete session timeline for review and export."""
    session_id: str
    events: list[TimelineEvent] = field(default_factory=list)
    start_time: float = 0.0

    @property
    def duration_s(self) -> float:
        if not self.events:
            return 0.0
        return self.events[-1].timestamp - self.start_time

    @property
    def turn_count(self) -> int:
        return sum(1 for e in self.events if e.event_type == "turn" and e.role == "user")

    @property
    def tool_count(self) -> int:
        return sum(1 for e in self.events if e.event_type == "tool")

    @property
    def summary(self) -> str:
        return (
            f"Session {self.session_id}: {self.turn_count} turns, "
            f"{self.tool_count} tool calls, {self.duration_s:.0f}s"
        )

    def milestones(self) -> list[TimelineEvent]:
        """Get milestone events only (key decision points)."""
        return [e for e in self.events if e.event_type == "milestone"]

    def errors(self) -> list[TimelineEvent]:
        """Get error events only."""
        return [e for e in self.events if e.event_type == "error"]

    def export_markdown(self, path: str) -> None:
        """Export timeline as human-readable markdown."""
        lines = [f"# Session Replay: {self.session_id}\n"]
        lines.append(f"**Duration**: {self.duration_s:.0f}s | "
                     f"**Turns**: {self.turn_count} | "
                     f"**Tools**: {self.tool_count}\n")
        lines.append("## Timeline\n")

        for event in self.events:
            elapsed = event.timestamp - self.start_time
            prefix = f"[{elapsed:>6.1f}s]"
            annotation = f" *{event.annotation}*" if event.annotation else ""

            if event.event_type == "turn":
                role_icon = ">" if event.role == "user" else "<"
                lines.append(f"{prefix} {role_icon} **{event.role}**: {event.content[:200]}{annotation}")
            elif event.event_type == "tool":
                lines.append(f"{prefix}   tool: {event.content} ({event.duration_ms}ms){annotation}")
            elif event.event_type == "milestone":
                lines.append(f"{prefix} --- **{event.content}** ---{annotation}")
            elif event.event_type == "error":
                lines.append(f"{prefix} !! {event.content}{annotation}")

        with open(path, "w") as f:
            f.write("\n".join(lines))

    def export_json(self, path: str) -> None:
        """Export timeline as machine-readable JSON."""
        data = {
            "session_id": self.session_id,
            "start_time": self.start_time,
            "duration_s": self.duration_s,
            "turn_count": self.turn_count,
            "tool_count": self.tool_count,
            "events": [e.to_dict() for e in self.events],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "start_time": self.start_time,
            "duration_s": self.duration_s,
            "turn_count": self.turn_count,
            "tool_count": self.tool_count,
            "events": [e.to_dict() for e in self.events],
        }


class SessionRecorder:
    """Records session events for timeline replay."""

    def __init__(self, session_id: str = ""):
        self._session_id = session_id or f"hal-{int(time.time())}"
        self._start = time.time()
        self._events: list[TimelineEvent] = []

    @property
    def session_id(self) -> str:
        return self._session_id

    def record(
        self, event_type: str, content: str, *,
        role: str = "", metadata: dict | None = None,
        duration_ms: int = 0,
    ) -> None:
        """Record any event type: turn, tool, milestone, error.

        Args:
            event_type: "turn", "tool", "milestone", or "error"
            content: Message text, tool name, or description
            role: "user"/"assistant" for turns, empty otherwise
            metadata: Extra data (tool params, error details, etc.)
            duration_ms: Execution time for tool calls
        """
        self._events.append(TimelineEvent(
            timestamp=time.time(),
            event_type=event_type,
            role=role,
            content=content,
            metadata=metadata or {},
            duration_ms=duration_ms,
        ))

    def annotate_last(self, annotation: str) -> None:
        """Add annotation to the most recent event."""
        if self._events:
            self._events[-1].annotation = annotation

    def to_timeline(self) -> ReplayTimeline:
        """Convert recorded events to a ReplayTimeline."""
        return ReplayTimeline(
            session_id=self._session_id,
            events=list(self._events),
            start_time=self._start,
        )

    def save(self) -> str:
        """Save timeline to disk. Returns the file path."""
        os.makedirs(_REPLAYS_DIR, exist_ok=True)
        path = os.path.join(_REPLAYS_DIR, f"{self._session_id}.json")
        self.to_timeline().export_json(path)
        return path


def export_hal_format(timeline: ReplayTimeline, path: str) -> None:
    """Export timeline as portable .hal session format.

    The .hal format is a self-contained JSON file that includes:
    - Session metadata (id, timestamps, summary stats)
    - Full event timeline with annotations
    - Provider/model attribution per turn
    - Importable by any HAL instance for cross-device session review

    File extension convention: .hal (e.g. session-20260416.hal)
    """
    data = {
        "format": "hal-session-v1",
        "version": 1,
        "session_id": timeline.session_id,
        "start_time": timeline.start_time,
        "duration_s": timeline.duration_s,
        "summary": {
            "turns": timeline.turn_count,
            "tool_calls": timeline.tool_count,
            "milestones": len(timeline.milestones()),
            "errors": len(timeline.errors()),
        },
        "events": [e.to_dict() for e in timeline.events],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def import_hal_format(path: str) -> ReplayTimeline | None:
    """Import a .hal portable session file back into a ReplayTimeline."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    if data.get("format") != "hal-session-v1":
        return None
    events = [
        TimelineEvent(
            timestamp=e["timestamp"], event_type=e["event_type"],
            role=e.get("role", ""), content=e.get("content", ""),
            metadata=e.get("metadata", {}), duration_ms=e.get("duration_ms", 0),
            annotation=e.get("annotation", ""),
        )
        for e in data.get("events", [])
    ]
    return ReplayTimeline(
        session_id=data["session_id"],
        events=events,
        start_time=data.get("start_time", 0.0),
    )


def load_replay(session_id: str) -> ReplayTimeline | None:
    """Load a saved replay timeline."""
    path = os.path.join(_REPLAYS_DIR, f"{session_id}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    events = [
        TimelineEvent(
            timestamp=e["timestamp"], event_type=e["event_type"],
            role=e.get("role", ""), content=e.get("content", ""),
            metadata=e.get("metadata", {}), duration_ms=e.get("duration_ms", 0),
            annotation=e.get("annotation", ""),
        )
        for e in data.get("events", [])
    ]
    return ReplayTimeline(
        session_id=data["session_id"],
        events=events,
        start_time=data.get("start_time", 0.0),
    )
