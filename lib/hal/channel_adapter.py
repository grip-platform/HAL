"""HAL ChannelAdapter — the contract every transport adapter implements.

Each transport (telegram, discord, slack, whatsapp, signal, email,
voice) subclasses ChannelAdapter and implements three operations:

- send(envelope) — deliver an outbound MessageEnvelope to the native
  transport (HTTP POST, SMTP send, etc).
- start() — begin receiving inbound messages. Adapters typically spawn
  a background thread that polls the transport and calls
  router.deliver(envelope) for each inbound message. Idempotent.
- stop() — tear down the recv loop and release resources. Idempotent.

Capabilities are declared as an immutable set of feature strings:

- 'text'    — plain text messages
- 'voice'   — voice memos (adapter transcribes or passes through)
- 'image'   — image attachments
- 'file'    — arbitrary file attachments
- 'threads' — native thread/reply affordance (Slack, Discord)
- 'buttons' — inline buttons / quick replies (Telegram)

Callers can check `adapter.supports('voice')` to gate behaviour. The
router never inspects capabilities — it's a hint for higher layers.

Usage:
    from lib.hal.channel_adapter import ChannelAdapter
    from lib.hal.channels import MessageEnvelope, ChannelRouter

    class MyAdapter(ChannelAdapter):
        transport = "my_transport"

        def __init__(self, router, token):
            super().__init__(router)
            self._token = token

        def send(self, envelope: MessageEnvelope) -> None:
            # POST to my_transport API using envelope.thread_id + envelope.body
            ...

        def capabilities(self) -> frozenset:
            return frozenset({'text', 'image'})

        def start(self) -> None:
            # Spawn recv thread that calls self.deliver(envelope)
            ...

        def stop(self) -> None:
            # Tear down recv thread
            ...
"""
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lib.hal.channels import ChannelRouter, MessageEnvelope


CAPABILITY_TEXT = "text"
CAPABILITY_VOICE = "voice"
CAPABILITY_IMAGE = "image"
CAPABILITY_FILE = "file"
CAPABILITY_THREADS = "threads"
CAPABILITY_BUTTONS = "buttons"

_KNOWN_CAPABILITIES = frozenset({
    CAPABILITY_TEXT,
    CAPABILITY_VOICE,
    CAPABILITY_IMAGE,
    CAPABILITY_FILE,
    CAPABILITY_THREADS,
    CAPABILITY_BUTTONS,
})


class ChannelAdapter(ABC):
    """Abstract base class for transport adapters.

    Subclasses must set the class attribute `transport` to the canonical
    name used by ChannelRouter.register(). The router never introspects
    the class; it dispatches purely on the transport string in the
    envelope.

    The base class holds a back-reference to the router so adapters can
    call `self.deliver(envelope)` from their recv loop without the
    subclass needing to store the router themselves.
    """

    transport: str = ""

    def __init__(self, router: "ChannelRouter"):
        if not self.transport:
            raise ValueError(
                f"{type(self).__name__} must set class attribute 'transport'"
            )
        self._router = router

    @abstractmethod
    def send(self, envelope: "MessageEnvelope") -> None:
        """Deliver an outbound envelope to the native transport."""

    @abstractmethod
    def capabilities(self) -> frozenset:
        """Return the set of supported capability strings."""

    def start(self) -> None:
        """Start the recv loop. Default is no-op (outbound-only adapters)."""

    def stop(self) -> None:
        """Stop the recv loop. Default is no-op."""

    def deliver(self, envelope: "MessageEnvelope") -> None:
        """Forward an inbound envelope to the router.

        Adapters call this from their recv loop after normalising the
        native payload.
        """
        self._router.deliver(envelope)

    def supports(self, capability: str) -> bool:
        """Return True if this adapter declares the given capability."""
        return capability in self.capabilities()


def is_known_capability(capability: str) -> bool:
    """Return True if the capability string is one of the documented kinds."""
    return capability in _KNOWN_CAPABILITIES
