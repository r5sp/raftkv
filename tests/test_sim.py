"""The simulation substrate itself: virtual time, determinism, fault injection."""

from __future__ import annotations

import asyncio
import time

import pytest

from raftkv.messages import Message, NodeId, RequestVote
from raftkv.sim import NetworkConfig, SimNetwork, run_simulation
from raftkv.sim.loop import SimulationDeadlock


def test_virtual_time_skips_sleeps() -> None:
    async def main() -> float:
        loop = asyncio.get_running_loop()
        await asyncio.sleep(3600)
        await asyncio.wait_for(asyncio.sleep(10), timeout=20)
        return loop.time()

    wall = time.perf_counter()
    assert run_simulation(main) == pytest.approx(3610)
    assert time.perf_counter() - wall < 1.0


def test_wait_for_times_out_in_virtual_time() -> None:
    async def main() -> float:
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[None] = loop.create_future()
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(fut, 5.0)
        return loop.time()

    assert run_simulation(main) == pytest.approx(5.0)


def test_deadlock_is_detected() -> None:
    async def main() -> None:
        await asyncio.get_running_loop().create_future()

    with pytest.raises(SimulationDeadlock):
        run_simulation(main)


class Recorder:
    def __init__(self) -> None:
        self.got: list[tuple[NodeId, Message, float]] = []

    def __call__(self, src: NodeId, msg: Message) -> None:
        self.got.append((src, msg, asyncio.get_running_loop().time()))


def _run_net(
    config: NetworkConfig, n: int = 200, setup: object = None
) -> tuple[SimNetwork, Recorder]:
    rec = Recorder()

    async def main() -> SimNetwork:
        net = SimNetwork(seed=1, config=config)
        a, b = net.endpoint("a"), net.endpoint("b")
        b.set_handler(rec)
        await a.start()
        await b.start()
        if callable(setup):
            setup(net)
        for i in range(n):
            a.send("b", RequestVote(i, "a", 0, 0))
        await asyncio.sleep(1.0)
        return net

    return run_simulation(main), rec


def test_reliable_network_delivers_everything_with_delay() -> None:
    _, rec = _run_net(NetworkConfig(min_delay=0.01, max_delay=0.02))
    assert sorted(m.term for _, m, _ in rec.got) == list(range(200))  # type: ignore[union-attr]
    assert all(0.01 <= t <= 0.02 + 1e-9 for *_, t in rec.got)


def test_random_delays_reorder_messages() -> None:
    _, rec = _run_net(NetworkConfig(min_delay=0.001, max_delay=0.05))
    order = [m.term for _, m, _ in rec.got]  # type: ignore[union-attr]
    assert order != sorted(order)


def test_drop_and_duplicate_rates() -> None:
    net, rec = _run_net(NetworkConfig(drop_rate=0.3, duplicate_rate=0.3), n=2000)
    terms = [m.term for _, m, _ in rec.got]  # type: ignore[union-attr]
    unique = len(set(terms))
    assert 1200 < unique < 1600  # ~70% survive
    assert len(terms) - unique > 200  # duplicates arrived
    assert net.stats.dropped > 0
    assert net.stats.duplicated > 0


def test_partition_blocks_and_heal_restores() -> None:
    _, rec = _run_net(NetworkConfig(), setup=lambda net: net.partition(["a"], ["b"]))
    assert rec.got == []


def test_one_way_link_cut() -> None:
    _, rec = _run_net(NetworkConfig(), setup=lambda net: net.cut("b", "a"))
    assert len(rec.got) == 200  # a -> b still works


def test_partition_destroys_in_flight_messages() -> None:
    rec = Recorder()

    async def main() -> None:
        net = SimNetwork(seed=1, config=NetworkConfig(min_delay=0.1, max_delay=0.1))
        a, b = net.endpoint("a"), net.endpoint("b")
        b.set_handler(rec)
        await a.start()
        await b.start()
        a.send("b", RequestVote(1, "a", 0, 0))
        await asyncio.sleep(0.05)
        net.partition(["a"], ["b"])
        await asyncio.sleep(0.1)

    run_simulation(main)
    assert rec.got == []


def test_messages_are_serialized_not_shared() -> None:
    """Receivers get a decoded copy, so aliasing bugs cannot hide in the simulator."""
    from raftkv.messages import AppendEntries, Entry

    rec = Recorder()
    original = AppendEntries(1, "a", 0, 0, [Entry(1, 1, {"k": "v"})], 0)

    async def main() -> None:
        net = SimNetwork(seed=1)
        a, b = net.endpoint("a"), net.endpoint("b")
        b.set_handler(rec)
        await a.start()
        await b.start()
        a.send("b", original)
        await asyncio.sleep(0.1)

    run_simulation(main)
    ((_, got, _),) = rec.got
    assert got == original
    assert got is not original
