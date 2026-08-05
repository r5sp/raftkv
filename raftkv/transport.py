"""Pluggable message transports.

A transport delivers :mod:`raftkv.messages` between named endpoints with
fire-and-forget, at-most-once-ish semantics: messages may be dropped (for
example while a peer is down) and Raft is designed to tolerate that. The same
interface is implemented by :class:`TcpTransport` (real sockets) and by
:class:`raftkv.sim.network.SimTransport` (deterministic fault injection).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import struct
from collections.abc import Callable, Coroutine
from typing import Any, Protocol

from raftkv.messages import CodecError, Message, NodeId, decode, encode

log = logging.getLogger(__name__)

Handler = Callable[[NodeId, Message], None]
Address = tuple[str, int]

_FRAME = struct.Struct(">I")
MAX_FRAME = 64 * 1024 * 1024


class Transport(Protocol):
    def set_handler(self, handler: Handler) -> None: ...

    async def start(self) -> None: ...

    def send(self, dest: NodeId, msg: Message) -> None:
        """Queue ``msg`` for ``dest``. Never blocks; may silently drop."""
        ...

    async def close(self) -> None: ...


def parse_address(text: str) -> Address:
    host, _, port = text.rpartition(":")
    if not host or not port.isdigit():
        raise ValueError(f"expected host:port, got {text!r}")
    return host, int(port)


def parse_members(spec: str) -> dict[NodeId, Address]:
    """Parse ``"1=127.0.0.1:7001,2=127.0.0.1:7002"`` into an address map."""
    members: dict[NodeId, Address] = {}
    for item in filter(None, (s.strip() for s in spec.split(","))):
        node_id, sep, addr = item.partition("=")
        if not sep:
            raise ValueError(f"expected id=host:port, got {item!r}")
        members[node_id.strip()] = parse_address(addr.strip())
    if not members:
        raise ValueError("empty member list")
    return members


class TcpTransport:
    """Length-prefixed JSON frames over asyncio TCP streams.

    * Each endpoint in ``addresses`` gets one lazily-opened outbound
      connection; on failure it is re-dialled with a short back-off and
      messages sent meanwhile are dropped.
    * Every connection, inbound or outbound, is read for frames. Senders that
      are not in ``addresses`` (clients) are remembered so replies can be sent
      back on the connection they arrived on.
    * Per-connection write buffers are bounded; when a peer is too slow,
      messages are dropped rather than buffered without limit.
    """

    def __init__(
        self,
        node_id: NodeId,
        addresses: dict[NodeId, Address],
        listen: Address | None = None,
        *,
        reconnect_backoff: float = 0.1,
        max_buffer: int = 8 * 1024 * 1024,
    ) -> None:
        self.id = node_id
        self.addresses = {k: v for k, v in addresses.items() if k != node_id}
        self.listen = listen
        self._handler: Handler | None = None
        self._server: asyncio.Server | None = None
        self._out: dict[NodeId, asyncio.StreamWriter] = {}
        self._dialing: dict[NodeId, list[bytes]] = {}
        self._retry_at: dict[NodeId, float] = {}
        self._routes: dict[NodeId, asyncio.StreamWriter] = {}
        self._tasks: set[asyncio.Task[None]] = set()
        self._writers: set[asyncio.StreamWriter] = set()
        self._backoff = reconnect_backoff
        self._max_buffer = max_buffer
        self._closed = False

    def set_handler(self, handler: Handler) -> None:
        self._handler = handler

    async def start(self) -> None:
        if self.listen is not None:
            host, port = self.listen
            self._server = await asyncio.start_server(self._on_connection, host, port)

    @property
    def bound_port(self) -> int | None:
        if self._server is None or not self._server.sockets:
            return None
        port: int = self._server.sockets[0].getsockname()[1]
        return port

    def send(self, dest: NodeId, msg: Message) -> None:
        if self._closed:
            return
        payload = encode(self.id, msg)
        frame = _FRAME.pack(len(payload)) + payload
        if dest in self.addresses:
            writer = self._out.get(dest)
            if writer is not None and not writer.is_closing():
                self._write(writer, frame)
            else:
                self._dial(dest, frame)
        elif (writer := self._routes.get(dest)) is not None and not writer.is_closing():
            self._write(writer, frame)

    def _write(self, writer: asyncio.StreamWriter, frame: bytes) -> None:
        if writer.transport.get_write_buffer_size() > self._max_buffer:
            return
        writer.write(frame)

    def _dial(self, dest: NodeId, frame: bytes) -> None:
        if dest in self._dialing:
            queue = self._dialing[dest]
            if len(queue) < 1024:
                queue.append(frame)
            return
        loop = asyncio.get_running_loop()
        if loop.time() < self._retry_at.get(dest, 0.0):
            return
        self._dialing[dest] = [frame]
        self._spawn(self._connect(dest))

    async def _connect(self, dest: NodeId) -> None:
        host, port = self.addresses[dest]
        try:
            reader, writer = await asyncio.open_connection(host, port)
        except OSError:
            self._dialing.pop(dest, None)
            self._retry_at[dest] = asyncio.get_running_loop().time() + self._backoff
            return
        queued = self._dialing.pop(dest, [])
        if self._closed:
            writer.close()
            return
        self._out[dest] = writer
        for frame in queued:
            self._write(writer, frame)
        await self._read_loop(reader, writer, outbound_to=dest)

    async def _on_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        await self._read_loop(reader, writer)

    async def _read_loop(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        outbound_to: NodeId | None = None,
    ) -> None:
        self._writers.add(writer)
        try:
            while True:
                header = await reader.readexactly(_FRAME.size)
                (length,) = _FRAME.unpack(header)
                if length > MAX_FRAME:
                    raise CodecError(f"frame too large: {length}")
                src, msg = decode(await reader.readexactly(length))
                if src not in self.addresses:
                    self._routes[src] = writer
                if self._handler is not None:
                    self._handler(src, msg)
        except (asyncio.IncompleteReadError, ConnectionError, OSError):
            pass
        except CodecError as exc:
            log.warning("%s: dropping connection after bad frame: %s", self.id, exc)
        finally:
            self._writers.discard(writer)
            if outbound_to is not None and self._out.get(outbound_to) is writer:
                del self._out[outbound_to]
            for k in [k for k, w in self._routes.items() if w is writer]:
                del self._routes[k]
            writer.close()

    def _spawn(self, coro: Coroutine[Any, Any, None]) -> None:
        task = asyncio.ensure_future(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def close(self) -> None:
        self._closed = True
        if self._server is not None:
            self._server.close()
        for w in list(self._writers):
            w.close()
        for t in list(self._tasks):
            t.cancel()
        for t in list(self._tasks):
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await t
        if self._server is not None:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(self._server.wait_closed(), 1.0)
