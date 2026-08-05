"""In-memory network with deterministic fault injection.

Every message is (optionally) serialized with the real codec, then delivered
after a random delay drawn from a seeded RNG. Independent random delays mean
messages are routinely *reordered*. The network can also drop and duplicate
messages, cut individual directed links, partition the cluster into groups,
and isolate nodes. Links are checked both when a message is sent and when it
would be delivered, so a partition also destroys messages already in flight.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Iterable
from dataclasses import dataclass, field

from raftkv.messages import Message, NodeId, decode, encode
from raftkv.transport import Handler


@dataclass
class NetworkConfig:
    drop_rate: float = 0.0
    duplicate_rate: float = 0.0
    min_delay: float = 0.001
    max_delay: float = 0.005
    serialize: bool = True


@dataclass
class NetworkStats:
    sent: int = 0
    delivered: int = 0
    dropped: int = 0
    duplicated: int = 0
    by_type: dict[str, int] = field(default_factory=dict)


class SimNetwork:
    def __init__(self, seed: int | str, config: NetworkConfig | None = None) -> None:
        self.rng = random.Random(f"network/{seed}")
        self.config = config or NetworkConfig()
        self.stats = NetworkStats()
        self._endpoints: dict[NodeId, SimTransport] = {}
        self._cut: set[tuple[NodeId, NodeId]] = set()

    # -- topology control ------------------------------------------------------

    def cut(self, src: NodeId, dst: NodeId) -> None:
        """Drop all messages from ``src`` to ``dst`` (one direction)."""
        self._cut.add((src, dst))

    def partition(self, *groups: Iterable[NodeId]) -> None:
        """Split the listed nodes into groups that cannot talk to each other.

        Nodes not listed in any group (e.g. clients) stay connected to all.
        Replaces any previous partition.
        """
        self._cut.clear()
        sets = [list(g) for g in groups]
        for i, a in enumerate(sets):
            for j, b in enumerate(sets):
                if i != j:
                    for x in a:
                        for y in b:
                            self._cut.add((x, y))

    def isolate(self, node: NodeId, others: Iterable[NodeId]) -> None:
        for o in others:
            if o != node:
                self._cut.add((node, o))
                self._cut.add((o, node))

    def heal(self) -> None:
        self._cut.clear()

    def connected(self, src: NodeId, dst: NodeId) -> bool:
        return (src, dst) not in self._cut

    # -- endpoints ---------------------------------------------------------------

    def endpoint(self, address: NodeId) -> SimTransport:
        return SimTransport(self, address)

    def _register(self, t: SimTransport) -> None:
        self._endpoints[t.address] = t

    def _unregister(self, t: SimTransport) -> None:
        if self._endpoints.get(t.address) is t:
            del self._endpoints[t.address]

    # -- delivery ------------------------------------------------------------------

    def send(self, src: NodeId, dst: NodeId, msg: Message) -> None:
        cfg = self.config
        self.stats.sent += 1
        name = type(msg).__name__
        self.stats.by_type[name] = self.stats.by_type.get(name, 0) + 1
        if not self.connected(src, dst) or self.rng.random() < cfg.drop_rate:
            self.stats.dropped += 1
            return
        payload: bytes | Message = encode(src, msg) if cfg.serialize else msg
        copies = 1
        if self.rng.random() < cfg.duplicate_rate:
            copies = 2
            self.stats.duplicated += 1
        loop = asyncio.get_running_loop()
        for _ in range(copies):
            delay = self.rng.uniform(cfg.min_delay, cfg.max_delay)
            loop.call_later(delay, self._deliver, src, dst, payload)

    def _deliver(self, src: NodeId, dst: NodeId, payload: bytes | Message) -> None:
        endpoint = self._endpoints.get(dst)
        if endpoint is None or not self.connected(src, dst):
            self.stats.dropped += 1
            return
        if isinstance(payload, bytes):
            _, msg = decode(payload)
        else:
            msg = payload
        self.stats.delivered += 1
        endpoint.deliver(src, msg)


class SimTransport:
    """A :class:`raftkv.transport.Transport` attached to a :class:`SimNetwork`."""

    def __init__(self, network: SimNetwork, address: NodeId) -> None:
        self.network = network
        self.address = address
        self._handler: Handler | None = None
        self._open = False

    def set_handler(self, handler: Handler) -> None:
        self._handler = handler

    async def start(self) -> None:
        self._open = True
        self.network._register(self)

    def send(self, dest: NodeId, msg: Message) -> None:
        if self._open:
            self.network.send(self.address, dest, msg)

    def deliver(self, src: NodeId, msg: Message) -> None:
        if self._open and self._handler is not None:
            self._handler(src, msg)

    async def close(self) -> None:
        self._open = False
        self.network._unregister(self)
