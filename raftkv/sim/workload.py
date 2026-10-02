"""Randomized client workloads and fault schedules ("nemesis") for simulation runs.

``run_fault_scenario`` is the core of the fault-injection test suite:

1. start an N-node cluster on a lossy, reordering, duplicating network;
2. run several concurrent clients issuing random get/put/delete/cas
   operations on a small key space, recording every invocation and response
   with virtual timestamps;
3. concurrently, a nemesis injects partitions, isolates leaders, and crashes
   and restarts nodes;
4. heal everything, require the cluster to elect a leader and make progress
   again (liveness after faults stop), and require all replicas to converge;
5. hand back the history for the linearizability checker, plus statistics.

Safety invariants are checked continuously throughout by ``InvariantChecker``.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Callable
from dataclasses import dataclass, field

from raftkv.client import ClusterUnavailable, RaftClient, RaftKVError
from raftkv.core import RaftConfig
from raftkv.linearizability import Operation
from raftkv.messages import NodeId
from raftkv.sim.cluster import SimCluster
from raftkv.sim.network import NetworkConfig
from raftkv.storage import Storage


@dataclass
class ScenarioConfig:
    nodes: int = 5
    clients: int = 4
    keys: int = 3
    duration: float = 8.0
    network: NetworkConfig = field(
        default_factory=lambda: NetworkConfig(
            drop_rate=0.05, duplicate_rate=0.05, min_delay=0.001, max_delay=0.030
        )
    )
    nemesis: bool = True
    crashes: bool = True
    think_time: float = 0.04
    raft: RaftConfig | None = None


@dataclass
class ScenarioResult:
    history: list[Operation]
    cluster: SimCluster
    faults: list[str]
    completed_ops: int
    indeterminate_ops: int
    final_state: dict[str, str]


async def _client_worker(
    client: RaftClient,
    name: str,
    keys: list[str],
    rng: random.Random,
    history: list[Operation],
    stop_at: float,
    think_time: float,
) -> None:
    loop = asyncio.get_running_loop()
    i = 0
    while loop.time() < stop_at:
        i += 1
        key = rng.choice(keys)
        kind = rng.choice(["get", "get", "put", "put", "cas", "delete"])
        value = f"{name}-{i}"
        args: dict[str, object] = {}
        if kind == "put":
            args = {"value": value}
        elif kind == "cas":
            # CAS against a recently written value (likely current) or "absent".
            expected: str | None = None
            recent = [o for o in history[-20:] if o.key == key and o.kind in ("put", "cas")]
            if recent and rng.random() < 0.8:
                expected = str(rng.choice(recent).args["value"])
            args = {"expected": expected, "value": value}
        op = Operation(name, kind, key, args, call=loop.time())
        history.append(op)
        try:
            if kind == "get":
                op.output = {"value": await client.get(key)}
            elif kind == "put":
                op.output = {"prev": await client.put(key, value)}
            elif kind == "delete":
                op.output = {"deleted": await client.delete(key)}
            else:
                swapped, current = await client.cas(key, args["expected"], value)  # type: ignore[arg-type]
                op.output = {"swapped": swapped, "value": current}
            op.ret = loop.time()
        except (ClusterUnavailable, RaftKVError):
            # Outcome unknown: leave ret=inf / output=None. A read with an
            # unknown outcome constrains nothing, so drop it entirely.
            if kind == "get":
                history.remove(op)
        await asyncio.sleep(rng.uniform(0, think_time))


async def _nemesis(
    cluster: SimCluster, rng: random.Random, stop_at: float, crashes: bool, log: list[str]
) -> None:
    loop = asyncio.get_running_loop()
    ids = cluster.ids
    crashed: list[str] = []
    while loop.time() < stop_at:
        await asyncio.sleep(rng.uniform(0.3, 1.2))
        if loop.time() >= stop_at:
            break
        action = rng.choice(
            ["partition", "isolate_leader", "heal", "crash", "restart", "crash_leader", "bridge"]
            if crashes
            else ["partition", "isolate_leader", "heal", "bridge"]
        )
        t = f"{loop.time():7.3f}"
        if action == "partition":
            shuffled = ids[:]
            rng.shuffle(shuffled)
            cut = rng.randint(1, len(ids) // 2)
            cluster.net.partition(shuffled[:cut], shuffled[cut:])
            log.append(f"{t} partition {sorted(shuffled[:cut])} | {sorted(shuffled[cut:])}")
        elif action == "bridge" and len(ids) >= 5:
            # Two halves connected only through one "bridge" node (non-transitive).
            s = ids[:]
            rng.shuffle(s)
            bridge, a, b = s[0], s[1:3], s[3:]
            cluster.net.partition(a, b)
            log.append(f"{t} bridge via {bridge}: {sorted(a)} | {sorted(b)}")
        elif action == "isolate_leader":
            leader = cluster.leader()
            if leader is not None:
                cluster.net.heal()
                cluster.net.isolate(leader.id, ids)
                log.append(f"{t} isolate leader {leader.id}")
        elif action == "heal":
            cluster.net.heal()
            log.append(f"{t} heal")
        elif action in ("crash", "crash_leader"):
            # Keep at most a minority down so progress remains possible after heal.
            if len(crashed) >= (len(ids) - 1) // 2:
                continue
            if action == "crash_leader":
                leader = cluster.leader()
                victim = leader.id if leader is not None else None
            else:
                victim = rng.choice([i for i in ids if i not in crashed])
            if victim is None or victim in crashed:
                continue
            await cluster.crash(victim)
            crashed.append(victim)
            log.append(f"{t} crash {victim}")
        elif action == "restart" and crashed:
            victim = crashed.pop(rng.randrange(len(crashed)))
            await cluster.start_node(victim)
            log.append(f"{t} restart {victim}")
    for victim in crashed:
        await cluster.start_node(victim)
        log.append(f"{loop.time():7.3f} restart {victim}")
    cluster.net.heal()
    log.append(f"{loop.time():7.3f} heal (final)")


async def run_fault_scenario(
    seed: int | str,
    config: ScenarioConfig | None = None,
    storage_factory: Callable[[NodeId], Storage] | None = None,
) -> ScenarioResult:
    cfg = config or ScenarioConfig()
    rng = random.Random(f"scenario/{seed}")
    cluster = SimCluster(
        cfg.nodes, seed, network=cfg.network, raft=cfg.raft, storage_factory=storage_factory
    )
    await cluster.start()
    loop = asyncio.get_running_loop()
    start = loop.time()
    stop_at = start + cfg.duration
    keys = [f"k{i}" for i in range(cfg.keys)]
    history: list[Operation] = []
    faults: list[str] = []
    clients = [
        cluster.client(f"c{i}", request_timeout=0.4, deadline=3.0) for i in range(cfg.clients)
    ]
    workers = [
        asyncio.create_task(
            _client_worker(
                c,
                c.client_id,
                keys,
                random.Random(f"{seed}/w{i}"),
                history,
                stop_at,
                cfg.think_time,
            )
        )
        for i, c in enumerate(clients)
    ]
    if cfg.nemesis:
        await _nemesis(cluster, rng, stop_at, cfg.crashes, faults)
    await asyncio.gather(*workers)

    # Faults have stopped: the cluster must recover and serve requests again.
    cluster.net.heal()
    cluster.net.config = NetworkConfig(min_delay=0.001, max_delay=0.005)
    await cluster.wait_for_leader(timeout=10.0)
    probe = cluster.client("probe", request_timeout=0.5, deadline=10.0)
    op = Operation("probe", "put", "probe", {"value": "done"}, call=loop.time())
    op.output = {"prev": await probe.put("probe", "done")}
    op.ret = loop.time()
    history.append(op)
    await cluster.wait_converged(timeout=10.0)
    leader = cluster.leader()
    assert leader is not None
    final_state = dict(leader.sm.data)
    for node in cluster.live():
        if node.sm.data != final_state:
            raise AssertionError(f"replica divergence: {node.id} {node.sm.data} != {final_state}")
    await cluster.stop()
    completed = sum(1 for o in history if o.completed)
    return ScenarioResult(
        history, cluster, faults, completed, len(history) - completed, final_state
    )
