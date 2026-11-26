"""HAL Channel Router — provider-agnostic transport dispatch.

Single process, many transports. The router holds a registry of
ChannelAdapter instances keyed by transport name (telegram, slack,
discord, whatsapp, signal, email, voice) and a single inbound handler
callback that receives normalised MessageEnvelope objects from any
adapter.

Outbound: caller constructs a MessageEnvelope and passes it to
`router.send(envelope)`. The router looks up the adapter and delegates.

Inbound: each adapter's recv loop normalises its native payload into a
MessageEnvelope, then calls `router.deliver(envelope)`. The router
invokes the registered handler (typically an agent_runner hook).

No assumption about the handler's shape — sync or async, any return
value. The router stays transport-agnostic.

Usage:
    from lib.hal.channels import ChannelRouter, MessageEnvelope

    router = ChannelRouter()
    router.register("telegram", telegram_adapter)
    router.register("slack", slack_adapter)

    def on_message(envelope: MessageEnvelope) -> None:
        reply_body = agent_runner.handle(envelope.body)
        reply = envelope.reply(reply_body)
        router.send(reply)

    router.set_handler(on_message)
    telegram_adapter.start()  # adapter recv loop calls router.deliver()
"""
import threading
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Optional


@dataclass(frozen=True)
class MessageEnvelope:
    """Normalised cross-transport message.

    Canonical shape for inbound and outbound messages, regardless of
    source transport. Adapters translate native payloads to/from this
    envelope.

    Fields:
        transport: Transport name (telegram, slack, discord, ...).
        external_id: Platform-native sender ID (Telegram user_id,
            Slack user, WhatsApp MSISDN, email address).
        thread_id: Platform-native conversation/chat/channel ID.
        body: Text content of the message.
        kind: 'text' | 'voice' | 'image' | 'file' | 'system'.
        meta: Additional platform-specific fields the adapter wants
            to preserve (message_id, attachments, etc).
        direction: 'inbound' for messages received from transport,
            'outbound' for messages to be sent.
        timestamp: Unix epoch seconds at envelope construction.
    """
    transport: str
    external_id: str
    thread_id: str
    body: str = ""
    kind: str = "text"
    meta: dict = field(default_factory=dict)
    direction: str = "inbound"
    timestamp: float = field(default_factory=time.time)

    def reply(self, body: str, *, kind: str = "text", meta: Optional[dict] = None) -> "MessageEnvelope":
        """Build an outbound envelope addressed to the same recipient.

        The reply envelope preserves transport, external_id, and
        thread_id so the adapter can route it back to the same chat.
        """
        return replace(
            self,
            body=body,
            kind=kind,
            meta=meta or {},
            direction="outbound",
            timestamp=time.time(),
        )

    def to_dict(self) -> dict:
        return {
            "transport": self.transport,
            "external_id": self.external_id,
            "thread_id": self.thread_id,
            "body": self.body,
            "kind": self.kind,
            "meta": dict(self.meta),
            "direction": self.direction,
            "timestamp": self.timestamp,
        }


class UnknownTransportError(KeyError):
    """Raised when router.send() is called for an unregistered transport."""


class ChannelRouter:
    """Routes MessageEnvelope instances between transports and a handler.

    The router is intentionally thin — it holds a registry, a single
    inbound handler, and dispatch primitives. All transport-specific
    logic lives in ChannelAdapter implementations.

    Thread-safe: all registry mutations hold a lock. deliver() and
    send() take the lock only to look up the target; the handler/adapter
    call itself runs outside the lock to avoid re-entrant deadlock.
    """

    def __init__(self, fleet_counters: Any = None):
        self._lock = threading.Lock()
        self._adapters: dict[str, Any] = {}
        self._handler: Optional[Callable[[MessageEnvelope], Any]] = None
        # Optional FleetCounters instance — when set, deliver() records
        # one gateway message per inbound envelope so FleetSnapshot can
        # attribute multi-platform traffic per transport. None is fine;
        # everything works without fleet integration.
        self._fleet_counters = fleet_counters

    def register(self, transport: str, adapter: Any) -> None:
        """Register an adapter for a transport name.

        Re-registering the same transport replaces the previous adapter.
        """
        if not transport:
            raise ValueError("transport must be a non-empty string")
        with self._lock:
            self._adapters[transport] = adapter

    def unregister(self, transport: str) -> None:
        """Remove the adapter for a transport, if registered."""
        with self._lock:
            self._adapters.pop(transport, None)

    def transports(self) -> tuple:
        """List registered transport names."""
        with self._lock:
            return tuple(sorted(self._adapters))

    def set_handler(self, handler: Callable[[MessageEnvelope], Any]) -> None:
        """Set the inbound handler invoked by deliver()."""
        with self._lock:
            self._handler = handler

    def deliver(self, envelope: MessageEnvelope) -> Any:
        """Route an inbound envelope to the registered handler.

        Called by adapter recv loops. Returns the handler's return value
        so synchronous adapters can use it for immediate reply.

        Before invoking the handler, records the message on the optional
        FleetCounters so gateway traffic is visible in FleetSnapshot.
        """
        with self._lock:
            handler = self._handler
            counters = self._fleet_counters
        if counters is not None:
            try:
                counters.record_gateway_message(envelope.transport)
            except Exception:
                # Observability must never break the delivery path.
                pass
        if handler is None:
            return None
        return handler(envelope)

    def send(self, envelope: MessageEnvelope) -> Any:
        """Dispatch an outbound envelope to its transport adapter.

        Looks up the adapter by envelope.transport and calls
        adapter.send(envelope). Raises UnknownTransportError if the
        transport has no registered adapter.
        """
        with self._lock:
            adapter = self._adapters.get(envelope.transport)
        if adapter is None:
            raise UnknownTransportError(
                f"no adapter registered for transport {envelope.transport!r}"
            )
        return adapter.send(envelope)
