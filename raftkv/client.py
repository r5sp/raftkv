"""Client library: leader discovery, redirects, retries, exactly-once writes.

Each client has a unique ``client_id`` and numbers its writes with ``seq``.
A write is retried with the *same* ``seq`` until some leader answers, so the
server-side session table applies it exactly once even if earlier attempts
were committed but their replies were lost. Operations on a single client are
serialized (one outstanding request), which the session protocol requires.
"""

from __future__ import annotations

import asyncio
import random
import uuid
from collections.abc import Sequence
from typing import Any

from raftkv.messages import ClientReply, ClientRequest, Message, NodeId
from raftkv.transport import TcpTransport, Transport, parse_members


class RaftKVError(Exception):
    """The cluster returned an application-level error."""


class ClusterUnavailable(TimeoutError):
    """No leader answered within the client's overall deadline."""


class RaftClient:
    def __init__(
        self,
        transport: Transport,
        servers: Sequence[NodeId],
        *,
        client_id: str | None = None,
        request_timeout: float = 0.5,
        deadline: float = 10.0,
        backoff: float = 0.05,
        rng: random.Random | None = None,
    ) -> None:
        if not servers:
            raise ValueError("need at least one server")
        self.transport = transport
        self.servers = list(servers)
        self.client_id = client_id or uuid.uuid4().hex
        self.request_timeout = request_timeout
        self.deadline = deadline
        self.backoff = backoff
        self.rng = rng or random.Random()
        self.leader: NodeId | None = None
        self._seq = 0
        self._request_id = 0
        self._waiting: tuple[int, asyncio.Future[ClientReply]] | None = None
        self._lock = asyncio.Lock()
        self._started = False
        transport.set_handler(self._on_message)

    async def start(self) -> None:
        if not self._started:
            await self.transport.start()
            self._started = True

    async def close(self) -> None:
        await self.transport.close()

    # -- public API ----------------------------------------------------------

    async def get(self, key: str) -> str | None:
        result = await self._call({"kind": "get", "key": key}, write=False)
        value: str | None = result["value"]
        return value

    async def put(self, key: str, value: str) -> str | None:
        """Set ``key``; returns the previous value."""
        result = await self._call({"kind": "put", "key": key, "value": value}, write=True)
        prev: str | None = result["prev"]
        return prev

    async def delete(self, key: str) -> bool:
        result = await self._call({"kind": "delete", "key": key}, write=True)
        return bool(result["deleted"])

    async def cas(self, key: str, expected: str | None, value: str) -> tuple[bool, str | None]:
        """Set ``key`` to ``value`` iff it currently equals ``expected``.

        ``expected=None`` means "only if absent". Returns ``(swapped, current)``.
        """
        op = {"kind": "cas", "key": key, "expected": expected, "value": value}
        result = await self._call(op, write=True)
        return bool(result["swapped"]), result["value"]

    async def status(self, server: NodeId) -> dict[str, Any]:
        """Ask one specific server for its Raft status (no redirects)."""
        await self.start()
        async with self._lock:
            reply = await self._attempt(server, 0, {"kind": "status"})
        if reply is None or not reply.ok:
            raise ClusterUnavailable(f"{server} did not answer")
        result: dict[str, Any] = reply.result
        return result

    # -- internals -------------------------------------------------------------

    def _on_message(self, src: NodeId, msg: Message) -> None:
        if not isinstance(msg, ClientReply) or self._waiting is None:
            return
        request_id, fut = self._waiting
        if msg.request_id == request_id and not fut.done():
            fut.set_result(msg)

    async def _attempt(self, server: NodeId, seq: int, op: dict[str, Any]) -> ClientReply | None:
        self._request_id += 1
        fut: asyncio.Future[ClientReply] = asyncio.get_running_loop().create_future()
        self._waiting = (self._request_id, fut)
        self.transport.send(server, ClientRequest(self._request_id, self.client_id, seq, op))
        try:
            return await asyncio.wait_for(fut, self.request_timeout)
        except TimeoutError:
            return None
        finally:
            self._waiting = None

    def _next_server(self, current: NodeId | None) -> NodeId:
        others = [s for s in self.servers if s != current] or self.servers
        return self.rng.choice(others)

    async def _call(self, op: dict[str, Any], *, write: bool) -> dict[str, Any]:
        await self.start()
        async with self._lock:
            loop = asyncio.get_running_loop()
            if write:
                self._seq += 1
            seq = self._seq if write else 0
            give_up = loop.time() + self.deadline
            target = self.leader or self.rng.choice(self.servers)
            while loop.time() < give_up:
                reply = await self._attempt(target, seq, op)
                if reply is None:  # timeout: maybe the server is down or partitioned
                    self.leader = None
                    target = self._next_server(target)
                    continue
                if reply.ok:
                    self.leader = target
                    result: dict[str, Any] = reply.result
                    if "error" in result:
                        raise RaftKVError(result["error"])
                    return result
                if reply.error == "not_leader":
                    hint = reply.leader_hint
                    if hint is not None and hint != target and hint in self.servers:
                        target = hint
                    else:
                        target = self._next_server(target)
                        await asyncio.sleep(self.backoff * (0.5 + self.rng.random()))
                elif reply.error == "retry":
                    await asyncio.sleep(self.backoff * (0.5 + self.rng.random()))
                else:
                    raise RaftKVError(reply.error or "unknown error")
            raise ClusterUnavailable(f"no leader answered within {self.deadline}s")


def connect(members: str | dict[NodeId, tuple[str, int]], **kwargs: Any) -> RaftClient:
    """Build a TCP client for a cluster spec like ``"1=127.0.0.1:7001,2=..."``."""
    addresses = parse_members(members) if isinstance(members, str) else dict(members)
    client_id = kwargs.pop("client_id", None) or uuid.uuid4().hex
    transport = TcpTransport(client_id, addresses, listen=None)
    return RaftClient(transport, sorted(addresses), client_id=client_id, **kwargs)
