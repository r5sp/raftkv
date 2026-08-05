"""End-to-end behaviour of real RaftNode runtimes + client over the simulated network."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path

import pytest

from raftkv.client import ClusterUnavailable
from raftkv.core import Role
from raftkv.sim import NetworkConfig, SimCluster, run_simulation
from raftkv.storage import FileStorage


def sim(
    size: int, seed: int = 1, network: NetworkConfig | None = None, **kw: object
) -> Callable[[Callable[[SimCluster], Awaitable[None]]], None]:
    """Run ``body(cluster)`` in virtual time, then assert no invariant broke."""

    def runner(body: Callable[[SimCluster], Awaitable[None]]) -> None:
        async def main() -> SimCluster:
            cluster = SimCluster(size, seed, network=network, **kw)  # type: ignore[arg-type]
            await cluster.start()
            try:
                await body(cluster)
            finally:
                await cluster.stop()
            return cluster

        cluster = run_simulation(main)
        assert cluster.errors == []
        assert cluster.checker.violations == []

    return runner


def test_basic_operations() -> None:
    @sim(3)
    async def body(c: SimCluster) -> None:
        await c.wait_for_leader()
        client = c.client("alice")
        assert await client.put("a", "1") is None
        assert await client.get("a") == "1"
        assert await client.cas("a", "1", "2") == (True, "2")
        assert await client.cas("a", "1", "3") == (False, "2")
        assert await client.delete("a") is True
        assert await client.get("a") is None
        assert await client.cas("a", None, "fresh") == (True, "fresh")
        await c.wait_converged()
        assert {n.id: n.sm.data for n in c.live()} == {i: {"a": "fresh"} for i in c.ids}


def test_client_follows_redirect_from_follower() -> None:
    @sim(5)
    async def body(c: SimCluster) -> None:
        leader = await c.wait_for_leader()
        client = c.client("bob")
        client.leader = next(i for i in c.ids if i != leader.id)  # start at a follower
        await client.put("k", "v")
        assert client.leader == leader.id


def test_leader_crash_triggers_reelection_and_service_continues() -> None:
    @sim(5)
    async def body(c: SimCluster) -> None:
        old = await c.wait_for_leader()
        client = c.client("carol")
        for i in range(10):
            await client.put(f"k{i}", str(i))
        await c.crash(old.id)
        new = await c.wait_for_leader()
        assert new.id != old.id
        assert new.core.current_term > old.core.current_term
        for i in range(10):
            assert await client.get(f"k{i}") == str(i)
        await client.put("after", "crash")
        await c.start_node(old.id)  # rejoins as follower and catches up
        await c.wait_converged()
        assert c.nodes[old.id] is not None
        assert c.nodes[old.id].sm.data["after"] == "crash"  # type: ignore[union-attr]


def test_whole_cluster_restart_preserves_data() -> None:
    @sim(3)
    async def body(c: SimCluster) -> None:
        await c.wait_for_leader()
        client = c.client("dave")
        for i in range(100):  # crosses the snapshot threshold
            await client.put("counter", str(i))
        for node_id in c.ids:
            await c.crash(node_id)
        for node_id in c.ids:
            await c.start_node(node_id)
        await c.wait_for_leader()
        assert await client.get("counter") == "99"


def test_writes_are_exactly_once_on_a_lossy_duplicating_network() -> None:
    """A retried CAS whose first attempt committed must still report success.

    Without server-side sessions the retry would re-execute, find the value
    already swapped and return swapped=False -- or worse, a retried put
    could clobber a later write.
    """
    lossy = NetworkConfig(drop_rate=0.2, duplicate_rate=0.3, min_delay=0.001, max_delay=0.03)

    @sim(3, seed=4, network=lossy)
    async def body(c: SimCluster) -> None:
        await c.wait_for_leader()
        client = c.client("eve", request_timeout=0.15)
        assert await client.cas("n", None, "0") == (True, "0")
        for i in range(60):
            swapped, value = await client.cas("n", str(i), str(i + 1))
            assert swapped, f"cas {i}->{i + 1} reported failure; value={value}"
        assert await client.get("n") == "60"


def test_minority_partition_cannot_serve_reads_or_writes() -> None:
    @sim(5)
    async def body(c: SimCluster) -> None:
        leader = await c.wait_for_leader()
        client = c.client("frank")
        await client.put("k", "before")
        minority = [leader.id, next(i for i in c.ids if i != leader.id)]
        majority = [i for i in c.ids if i not in minority]
        c.net.partition(minority, majority)
        # A client that can only reach the minority gets neither reads nor writes.
        stuck = c.client("stuck", request_timeout=0.2, deadline=2.0)
        stuck.servers = minority
        with pytest.raises(ClusterUnavailable):
            await stuck.get("k")
        with pytest.raises(ClusterUnavailable):
            await stuck.put("k", "lost?")
        # The majority side elects a leader and moves on.
        await client.put("k", "after")
        c.net.heal()
        await c.wait_converged()
        assert all(n.sm.data["k"] == "after" for n in c.live())
        assert sum(n.core.role is Role.LEADER for n in c.live()) == 1


def test_lagging_node_receives_snapshot() -> None:
    @sim(3)
    async def body(c: SimCluster) -> None:
        await c.wait_for_leader()
        victim = next(i for i in c.ids if c.nodes[i] is not c.leader())
        await c.crash(victim)
        client = c.client("gina")
        for i in range(200):
            await client.put(f"key{i % 7}", str(i))
        await c.start_node(victim)
        await c.wait_converged()
        node = c.nodes[victim]
        assert node is not None
        assert node.core.stats["snapshots_installed"] >= 1
        leader = c.leader()
        assert leader is not None
        assert node.sm.data == leader.sm.data


def test_cluster_on_real_disk_survives_crashes(tmp_path: Path) -> None:
    def disk(node_id: str) -> FileStorage:
        return FileStorage(tmp_path / node_id)

    @sim(3, storage_factory=disk)
    async def body(c: SimCluster) -> None:
        await c.wait_for_leader()
        client = c.client("hal")
        for i in range(80):
            await client.put("x", str(i))
            if i % 25 == 24:
                victim = c.leader()
                assert victim is not None
                await c.crash(victim.id)
                c.storages[victim.id] = disk(victim.id)  # reopen from files
                await c.start_node(victim.id)
        await c.wait_converged()
        assert all(n.sm.data == {"x": "79"} for n in c.live())


def test_single_node_cluster() -> None:
    @sim(1)
    async def body(c: SimCluster) -> None:
        await c.wait_for_leader()
        client = c.client("solo")
        await client.put("a", "b")
        assert await client.get("a") == "b"


def test_status_reports_role() -> None:
    @sim(3)
    async def body(c: SimCluster) -> None:
        leader = await c.wait_for_leader()
        client = c.client("ops")
        status = await client.status(leader.id)
        assert status["role"] == "leader"
        assert status["id"] == leader.id
        await asyncio.sleep(0.1)
        follower = next(i for i in c.ids if i != leader.id)
        assert (await client.status(follower))["leader"] == leader.id
