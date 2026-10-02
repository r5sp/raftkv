"""A cluster of real :class:`RaftNode` runtimes on a :class:`SimNetwork`.

Used by the integration and fault-injection tests. Must run inside
:func:`raftkv.sim.run_simulation` (virtual time) to be deterministic.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Callable
from typing import Any

from raftkv.client import RaftClient
from raftkv.core import RaftConfig, RaftCore, Role
from raftkv.messages import NodeId
from raftkv.node import RaftNode
from raftkv.sim.invariants import InvariantChecker
from raftkv.sim.network import NetworkConfig, SimNetwork
from raftkv.storage import MemoryStorage, Storage

SIM_RAFT_CONFIG = RaftConfig(
    election_timeout_min=0.150,
    election_timeout_max=0.300,
    heartbeat_interval=0.050,
    max_entries_per_append=32,
    snapshot_threshold=64,
)


class SimCluster:
    def __init__(
        self,
        size: int,
        seed: int | str,
        *,
        network: NetworkConfig | None = None,
        raft: RaftConfig | None = None,
        storage_factory: Callable[[NodeId], Storage] | None = None,
        log_matching_interval: float = 0.1,
    ) -> None:
        self.seed = seed
        self.ids: list[NodeId] = [f"n{i}" for i in range(1, size + 1)]
        self.net = SimNetwork(seed, network)
        self.raft_config = raft or SIM_RAFT_CONFIG
        factory = storage_factory or (lambda _id: MemoryStorage())
        self.storages: dict[NodeId, Storage] = {i: factory(i) for i in self.ids}
        self.nodes: dict[NodeId, RaftNode | None] = dict.fromkeys(self.ids)
        self.checker = InvariantChecker()
        self.errors: list[str] = []
        self._incarnation: dict[NodeId, int] = dict.fromkeys(self.ids, 0)
        self._checker_task: asyncio.Task[None] | None = None
        self._interval = log_matching_interval
        self.rng = random.Random(f"cluster/{seed}")
        self.stats_snapshots_installed = 0

    # -- lifecycle -------------------------------------------------------------------

    async def start(self) -> None:
        asyncio.get_running_loop().set_exception_handler(self._on_loop_error)
        for node_id in self.ids:
            await self.start_node(node_id)
        self._checker_task = asyncio.create_task(self._check_loop())

    async def stop(self) -> None:
        if self._checker_task is not None:
            self._checker_task.cancel()
        for node_id in self.ids:
            if self.nodes[node_id] is not None:
                await self.crash(node_id)

    def _on_loop_error(self, loop: asyncio.AbstractEventLoop, context: dict[str, Any]) -> None:
        exc = context.get("exception")
        self.errors.append(f"{context.get('message')}: {exc!r}")

    async def start_node(self, node_id: NodeId) -> RaftNode:
        assert self.nodes[node_id] is None, f"{node_id} already running"
        self._incarnation[node_id] += 1
        node = RaftNode(
            node_id,
            self.ids,
            self.net.endpoint(node_id),
            self.storages[node_id],
            config=self.raft_config,
            rng=random.Random(f"{self.seed}/{node_id}/{self._incarnation[node_id]}"),
            observer=self.checker,
        )
        await node.start()
        self.nodes[node_id] = node
        return node

    async def crash(self, node_id: NodeId) -> None:
        """Stop a node abruptly. Its storage survives; volatile state is lost."""
        node = self.nodes[node_id]
        if node is not None:
            self.stats_snapshots_installed += node.core.stats["snapshots_installed"]
            await node.stop()
            self.nodes[node_id] = None

    async def restart(self, node_id: NodeId) -> RaftNode:
        await self.crash(node_id)
        return await self.start_node(node_id)

    # -- inspection ------------------------------------------------------------------

    def live(self) -> list[RaftNode]:
        return [n for n in self.nodes.values() if n is not None]

    def cores(self) -> list[RaftCore]:
        return [n.core for n in self.live()]

    def leader(self) -> RaftNode | None:
        """The live leader with the highest term, if any."""
        leaders = [n for n in self.live() if n.core.role is Role.LEADER]
        return max(leaders, key=lambda n: n.core.current_term, default=None)

    async def wait_for_leader(self, timeout: float = 5.0) -> RaftNode:
        loop = asyncio.get_running_loop()
        end = loop.time() + timeout
        while loop.time() < end:
            leader = self.leader()
            # Require a leader that has committed its no-op, i.e. is usable.
            if leader is not None:
                core = leader.core
                if core.log.term_at(core.commit_index) == core.current_term:
                    return leader
            await asyncio.sleep(0.01)
        raise TimeoutError(f"no leader elected within {timeout}s")

    async def wait_converged(self, timeout: float = 5.0) -> None:
        """Wait until all live nodes have applied the same prefix."""
        loop = asyncio.get_running_loop()
        end = loop.time() + timeout
        while loop.time() < end:
            cores = self.cores()
            applied = {c.last_applied for c in cores}
            if len(applied) == 1 and all(c.commit_index == c.last_applied for c in cores):
                leader = self.leader()
                if leader is not None and leader.core.commit_index == leader.core.log.last_index:
                    return
            await asyncio.sleep(0.02)
        raise TimeoutError(
            "cluster did not converge: " + repr([c.describe() for c in self.cores()])
        )

    def client(self, name: str, **kwargs: Any) -> RaftClient:
        kwargs.setdefault("rng", random.Random(f"{self.seed}/{name}"))
        return RaftClient(self.net.endpoint(name), self.ids, client_id=name, **kwargs)

    def snapshots_installed(self) -> int:
        return self.stats_snapshots_installed + sum(
            n.core.stats["snapshots_installed"] for n in self.live()
        )

    async def _check_loop(self) -> None:
        while True:
            await asyncio.sleep(self._interval)
            self.checker.check_log_matching(self.cores())
