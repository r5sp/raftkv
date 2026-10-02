"""Real sockets: a 3-node cluster on localhost with on-disk storage."""

from __future__ import annotations

import asyncio
import socket
import struct
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from raftkv.client import RaftClient, connect
from raftkv.core import RaftConfig, Role
from raftkv.messages import Message, NodeId, RequestVote
from raftkv.node import RaftNode
from raftkv.storage import FileStorage
from raftkv.transport import Address, TcpTransport, parse_members

FAST = RaftConfig(
    election_timeout_min=0.15,
    election_timeout_max=0.3,
    heartbeat_interval=0.03,
    snapshot_threshold=50,
)


def free_ports(n: int) -> list[int]:
    socks = [socket.socket() for _ in range(n)]
    for s in socks:
        s.bind(("127.0.0.1", 0))
    ports = [s.getsockname()[1] for s in socks]
    for s in socks:
        s.close()
    return ports


class TcpCluster:
    def __init__(self, root: Path, n: int = 3) -> None:
        self.root = root
        self.addrs: dict[NodeId, Address] = {
            str(i): ("127.0.0.1", p) for i, p in enumerate(free_ports(n), 1)
        }
        self.nodes: dict[NodeId, RaftNode] = {}

    async def start(self, node_id: NodeId) -> RaftNode:
        node = RaftNode(
            node_id,
            list(self.addrs),
            TcpTransport(node_id, self.addrs, listen=self.addrs[node_id]),
            FileStorage(self.root / node_id),
            config=FAST,
        )
        await node.start()
        self.nodes[node_id] = node
        return node

    async def stop(self, node_id: NodeId) -> None:
        node = self.nodes.pop(node_id)
        await node.stop()
        node.storage.close()

    async def leader(self, timeout: float = 5.0) -> RaftNode:
        async with asyncio.timeout(timeout):
            while True:
                for n in self.nodes.values():
                    if (
                        n.core.role is Role.LEADER
                        and n.core.log.term_at(n.core.commit_index) == n.core.current_term
                    ):
                        return n
                await asyncio.sleep(0.02)

    def client(self) -> RaftClient:
        return connect(self.addrs, request_timeout=0.5, deadline=10.0)


@pytest.fixture
async def cluster(tmp_path: Path) -> AsyncIterator[TcpCluster]:
    c = TcpCluster(tmp_path)
    for node_id in list(c.addrs):
        await c.start(node_id)
    yield c
    for node_id in list(c.nodes):
        await c.stop(node_id)


async def test_tcp_cluster_end_to_end(cluster: TcpCluster) -> None:
    await cluster.leader()
    client = cluster.client()
    try:
        assert await client.put("lang", "python") is None
        assert await client.get("lang") == "python"
        assert await client.cas("lang", "python", "rust") == (True, "rust")
        assert await client.delete("lang")
        for i in range(60):  # crosses the snapshot threshold
            await client.put("i", str(i))
    finally:
        await client.close()


async def test_tcp_leader_failover_and_rejoin(cluster: TcpCluster) -> None:
    old = await cluster.leader()
    client = cluster.client()
    try:
        await client.put("k", "v1")
        await cluster.stop(old.id)
        new = await cluster.leader()
        assert new.id != old.id
        assert await client.get("k") == "v1"
        await client.put("k", "v2")
        # Restart the old leader from its data directory; it catches up.
        rejoined = await cluster.start(old.id)
        async with asyncio.timeout(5):
            while rejoined.sm.data.get("k") != "v2":
                await asyncio.sleep(0.02)
        assert rejoined.core.role is Role.FOLLOWER
    finally:
        await client.close()


async def test_transport_survives_garbage_and_unknown_destinations(tmp_path: Path) -> None:
    (port,) = free_ports(1)
    got: list[tuple[NodeId, Message]] = []
    server = TcpTransport("srv", {}, listen=("127.0.0.1", port))
    server.set_handler(lambda src, msg: got.append((src, msg)))
    await server.start()
    try:
        # Garbage frame: connection is dropped, server keeps running.
        _, w = await asyncio.open_connection("127.0.0.1", port)
        w.write(struct.pack(">I", 5) + b"nope!")
        await w.drain()
        w.close()
        # A well-formed client still gets through.
        client = TcpTransport("cli", {"srv": ("127.0.0.1", port)})
        client.send("nobody", RequestVote(1, "cli", 0, 0))  # silently dropped
        client.send("srv", RequestVote(1, "cli", 0, 0))
        async with asyncio.timeout(2):
            while not got:
                await asyncio.sleep(0.01)
        assert got[0][0] == "cli"
        await client.close()
    finally:
        await server.close()


def test_parse_members() -> None:
    assert parse_members("1=127.0.0.1:7001, 2=localhost:7002") == {
        "1": ("127.0.0.1", 7001),
        "2": ("localhost", 7002),
    }
    with pytest.raises(ValueError, match="id=host:port"):
        parse_members("127.0.0.1:7001")
    with pytest.raises(ValueError, match="host:port"):
        parse_members("1=nohost")
