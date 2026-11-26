"""HAL Tracing — lightweight observability for LLM operations.

OpenTelemetry-compatible span model without requiring the full OTel SDK.
Traces LLM calls, tool executions, and workflow steps with timing and
metadata. Can export to JSON for analysis or feed into OTel collectors.

Usage:
    from lib.hal.tracing import Tracer

    tracer = Tracer("hal-conversation")
    with tracer.span("llm_call", model="groq/llama") as s:
        response = registry.chat(...)
        s.set_attribute("tokens", response.usage.prompt_tokens)
    tracer.export_json("trace.json")
"""
import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

from lib.hal.home import HAL_HOME as _HAL_HOME

_TRACES_DIR = os.path.join(_HAL_HOME, "state", "hal", "traces")


@dataclass
class Span:
    """A single traced operation."""
    name: str
    trace_id: str
    span_id: str
    start_time: float = 0.0
    end_time: float = 0.0
    attributes: dict = field(default_factory=dict)
    status: str = "ok"  # "ok", "error"
    error_message: str = ""
    _elapsed_ms: float | None = field(default=None, repr=False, compare=False)

    @property
    def duration_ms(self) -> float:
        if self._elapsed_ms is not None:
            return self._elapsed_ms
        return (self.end_time - self.start_time) * 1000

    def set_attribute(self, key: str, value: object) -> None:
        """Set a span attribute."""
        self.attributes[key] = value

    def set_error(self, message: str) -> None:
        """Mark span as errored."""
        self.status = "error"
        self.error_message = message

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": round(self.duration_ms, 2),
            "attributes": self.attributes,
            "status": self.status,
            "error_message": self.error_message,
        }


class Tracer:
    """Lightweight tracer for HAL operations."""

    def __init__(self, service_name: str = "hal"):
        self._service = service_name
        self._trace_id = f"{service_name}-{int(time.time() * 1000)}"
        self._spans: list[Span] = []
        self._span_counter = 0

    @property
    def trace_id(self) -> str:
        return self._trace_id

    @property
    def spans(self) -> list[Span]:
        return list(self._spans)

    @contextmanager
    def span(self, name: str, **attributes):
        """Context manager that creates and records a span."""
        self._span_counter += 1
        elapsed_start = time.perf_counter()
        s = Span(
            name=name,
            trace_id=self._trace_id,
            span_id=f"{self._trace_id}-{self._span_counter}",
            start_time=time.time(),
            attributes=dict(attributes),
        )
        try:
            yield s
        except Exception as e:
            s.set_error(str(e)[:200])
            raise
        finally:
            s._elapsed_ms = (time.perf_counter() - elapsed_start) * 1000
            # Derive the end from the measured interval. On a coarse wall clock
            # (Python 3.12 on Windows, steps of about 15.6 ms) a second time.time()
            # read can equal the first, and the exported timestamps would then
            # disagree with duration_ms.
            s.end_time = s.start_time + s._elapsed_ms / 1000
            self._spans.append(s)

    def record(self, name: str, duration_ms: float = 0, **attributes) -> Span:
        """Record a completed span directly (for operations already timed)."""
        self._span_counter += 1
        now = time.time()
        s = Span(
            name=name,
            trace_id=self._trace_id,
            span_id=f"{self._trace_id}-{self._span_counter}",
            start_time=now - (duration_ms / 1000),
            end_time=now,
            attributes=dict(attributes),
            _elapsed_ms=float(duration_ms),
        )
        self._spans.append(s)
        return s

    def summary(self) -> dict:
        """Trace summary statistics."""
        total_ms = sum(s.duration_ms for s in self._spans)
        errors = sum(1 for s in self._spans if s.status == "error")
        return {
            "trace_id": self._trace_id,
            "service": self._service,
            "span_count": len(self._spans),
            "total_duration_ms": round(total_ms, 2),
            "error_count": errors,
        }

    def export_json(self, path: str = "") -> str:
        """Export trace as JSON. Returns file path."""
        if not path:
            os.makedirs(_TRACES_DIR, exist_ok=True)
            path = os.path.join(_TRACES_DIR, f"{self._trace_id}.json")

        data = {
            "trace_id": self._trace_id,
            "service": self._service,
            "spans": [s.to_dict() for s in self._spans],
            "summary": self.summary(),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return path
