"""Hand-driven cluster of bare ``RaftCore`` instances for precise unit tests.

No event loop and no randomness in delivery: the test decides exactly which
messages are delivered, dropped, or held back, and when time advances.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from typing import Any

from raftkv.core import RaftConfig, RaftCore, Role
from raftkv.messages import Command, Entry, Message, NodeId
from raftkv.statemachine import KVStateMachine
from raftkv.storage import MemoryStorage

CONFIG = RaftConfig(
    election_timeout_min=0.150,
    election_timeout_max=0.300,
    heartbeat_interval=0.050,
    snapshot_threshold=0,
)

Envelope = tuple[NodeId, NodeId, Message]


def storage_with(term: int, voted_for: NodeId | None, log_terms: Sequence[int]) -> MemoryStorage:
    """A MemoryStorage pre-loaded with a log whose entry i has term log_terms[i-1]."""
    st = MemoryStorage()
    st.save_hard_state(term, voted_for)
    st.append(
        [
            Entry(t, i, {"op": {"kind": "put", "key": "x", "value": f"{i}@{t}"}})
            for i, t in enumerate(log_terms, 1)
        ]
    )
    return st


class ManualCluster:
    def __init__(
        self,
        n: int = 3,
        *,
        config: RaftConfig = CONFIG,
        storages: dict[NodeId, MemoryStorage] | None = None,
        seed: int = 0,
    ) -> None:
        self.ids = [f"s{i}" for i in range(1, n + 1)]
        self.config = config
        self.storages = storages or {}
        for i in self.ids:
            self.storages.setdefault(i, MemoryStorage())
        self.seed = seed
        self.now = 0.0
        self.nodes: dict[NodeId, RaftCore] = {}
        self.down: set[NodeId] = set()
        self.queue: list[Envelope] = []
        for i in self.ids:
            self.boot(i)

    def boot(self, node_id: NodeId) -> RaftCore:
        core = RaftCore(
            node_id,
            self.ids,
            self.storages[node_id],
            KVStateMachine(),
            config=self.config,
            rng=random.Random(f"{self.seed}/{node_id}/{len(self.nodes)}"),
            now=self.now,
        )
        self.nodes[node_id] = core
        self.down.discard(node_id)
        return core

    def __getitem__(self, node_id: NodeId) -> RaftCore:
        return self.nodes[node_id]

    # -- message plumbing ------------------------------------------------------

    def collect(self) -> None:
        for i in self.ids:
            core = self.nodes[i]
            core.flush()
            for dest, msg in core.drain_outbox():
                if i not in self.down:
                    self.queue.append((i, dest, msg))

    def deliver(
        self,
        keep: Callable[[Envelope], bool] = lambda _e: True,
        rounds: int = 50,
        mutate: Callable[[Envelope], Message] | None = None,
    ) -> int:
        """Deliver queued messages (and the messages they cause) for ``rounds`` rounds.

        Messages rejected by ``keep`` or addressed to/from a down node are
        dropped; ``mutate`` may rewrite a message in transit.
        """
        delivered = 0
        for _ in range(rounds):
            self.collect()
            if not self.queue:
                break
            batch, self.queue = self.queue, []
            for env in batch:
                src, dst, msg = env
                if dst in self.down or src in self.down or not keep(env):
                    continue
                if mutate is not None:
                    msg = mutate(env)
                self.nodes[dst].step(src, msg, self.now)
                delivered += 1
        self.collect()
        return delivered

    def only_between(self, *group: NodeId) -> Callable[[Envelope], bool]:
        allowed = set(group)
        return lambda e: e[0] in allowed and e[1] in allowed

    def crash(self, node_id: NodeId) -> None:
        self.down.add(node_id)

    def restart(self, node_id: NodeId) -> RaftCore:
        return self.boot(node_id)

    # -- conveniences ------------------------------------------------------------

    def elect(self, node_id: NodeId, keep: Callable[[Envelope], bool] | None = None) -> RaftCore:
        core = self.nodes[node_id]
        core.campaign()
        self.deliver(keep or (lambda _e: True))
        return core

    def leaders(self) -> list[RaftCore]:
        return [c for i, c in self.nodes.items() if c.role is Role.LEADER and i not in self.down]

    def advance(self, dt: float) -> None:
        self.now += dt
        for i, core in self.nodes.items():
            if i not in self.down:
                core.tick(self.now)

    def propose(self, node_id: NodeId, value: str) -> int | None:
        cmd: Command = {"client_id": None, "op": {"kind": "put", "key": "x", "value": value}}
        return self.nodes[node_id].propose(cmd)

    def log_terms(self, node_id: NodeId) -> list[int]:
        log = self.nodes[node_id].log
        return [
            t
            for i in range(log.first_index, log.last_index + 1)
            if (t := log.term_at(i)) is not None
        ]


def describe(cores: Sequence[RaftCore]) -> list[dict[str, Any]]:
    return [c.describe() for c in cores]
