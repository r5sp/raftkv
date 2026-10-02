"""asyncio runtime that drives a :class:`RaftCore` over a :class:`Transport`.

The runtime owns everything the core deliberately does not: the clock and
timers, the transport, client request bookkeeping, and the KV state machine.
It runs entirely on one event loop thread, so no locking is needed.
"""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from raftkv.core import RaftConfig, RaftCore, RaftObserver, Role
from raftkv.messages import ClientReply, ClientRequest, Message, NodeId
from raftkv.statemachine import WRITE_OPS, KVStateMachine
from raftkv.storage import Storage
from raftkv.transport import Transport

log = logging.getLogger(__name__)


@dataclass
class _Waiter:
    term: int
    client: NodeId
    request_id: int


@dataclass(frozen=True)
class _ReadCtx:
    client: NodeId
    request_id: int
    key: str


class RaftNode:
    def __init__(
        self,
        node_id: NodeId,
        members: Sequence[NodeId],
        transport: Transport,
        storage: Storage,
        *,
        config: RaftConfig | None = None,
        rng: random.Random | None = None,
        state_machine: KVStateMachine | None = None,
        observer: RaftObserver | None = None,
    ) -> None:
        self.id = node_id
        self.members = list(members)
        self.transport = transport
        self.storage = storage
        self.config = config or RaftConfig()
        self.rng = rng or random.Random()
        self.sm = state_machine or KVStateMachine()
        self.observer = observer
        self._core: RaftCore | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._timer: asyncio.TimerHandle | None = None
        self._timer_when = float("inf")
        self._kick_scheduled = False
        self._pending: dict[int, list[_Waiter]] = {}
        self._was_leader = False
        self.running = False

    @property
    def core(self) -> RaftCore:
        if self._core is None:
            raise RuntimeError("node not started")
        return self._core

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._core = RaftCore(
            self.id,
            self.members,
            self.storage,
            self.sm,
            config=self.config,
            rng=self.rng,
            now=self._loop.time(),
            observer=self.observer,
        )
        self.transport.set_handler(self._on_message)
        await self.transport.start()
        self.running = True
        self._reschedule()

    async def stop(self) -> None:
        """Stop immediately (models a crash: no graceful hand-off)."""
        self.running = False
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        self._pending.clear()
        await self.transport.close()

    # ------------------------------------------------------------------
    # Inputs
    # ------------------------------------------------------------------

    def _now(self) -> float:
        assert self._loop is not None
        return self._loop.time()

    def _on_message(self, src: NodeId, msg: Message) -> None:
        if not self.running:
            return
        if isinstance(msg, ClientRequest):
            self._on_client_request(src, msg)
        elif isinstance(msg, ClientReply):
            return
        else:
            self.core.step(src, msg, self._now())
        self._kick()

    def _on_timer(self) -> None:
        self._timer = None
        self._timer_when = float("inf")
        if not self.running:
            return
        self.core.tick(self._now())
        self._kick()

    def _on_client_request(self, src: NodeId, req: ClientRequest) -> None:
        core = self.core
        kind = req.op.get("kind")
        if kind == "status":
            self._reply(src, req.request_id, True, core.describe())
            return
        if core.role is not Role.LEADER:
            self._reply(src, req.request_id, False, error="not_leader")
            return
        if kind == "get":
            core.request_read(_ReadCtx(src, req.request_id, str(req.op.get("key"))))
            return
        if kind not in WRITE_OPS:
            self._reply(src, req.request_id, False, error=f"unknown op {kind!r}")
            return
        cached = self.sm.cached_result(req.client_id, req.seq)
        if cached is not None:
            # Retry of a write that has already been applied: answer from the session.
            self._reply(src, req.request_id, True, cached)
            return
        command = {"client_id": req.client_id, "seq": req.seq, "op": req.op}
        index = core.propose(command)
        assert index is not None
        self._pending.setdefault(index, []).append(_Waiter(core.current_term, src, req.request_id))

    # ------------------------------------------------------------------
    # Output processing
    # ------------------------------------------------------------------

    def _kick(self) -> None:
        """Coalesce output processing to once per event-loop iteration.

        Proposals that arrive in the same iteration are thus replicated in a
        single AppendEntries round.
        """
        if not self._kick_scheduled:
            self._kick_scheduled = True
            assert self._loop is not None
            self._loop.call_soon(self._process)

    def _process(self) -> None:
        self._kick_scheduled = False
        if not self.running:
            return
        core = self.core
        core.flush()
        for dest, msg in core.drain_outbox():
            self.transport.send(dest, msg)

        applied, core.applied = core.applied, []
        for index, term, result in applied:
            for w in self._pending.pop(index, ()):
                if w.term == term:
                    self._reply(w.client, w.request_id, True, result)
                else:
                    # A different leader's entry won this index; outcome unknown
                    # to this request, so the client must retry (sessions dedup).
                    self._reply(w.client, w.request_id, False, error="retry")

        ready, core.ready_reads = core.ready_reads, []
        for ctx, _read_index in ready:
            assert isinstance(ctx, _ReadCtx)
            self._reply(ctx.client, ctx.request_id, True, self.sm.get(ctx.key))
        failed, core.failed_reads = core.failed_reads, []
        for ctx in failed:
            assert isinstance(ctx, _ReadCtx)
            self._reply(ctx.client, ctx.request_id, False, error="not_leader")

        is_leader = core.role is Role.LEADER
        if self._was_leader and not is_leader:
            for waiters in self._pending.values():
                for w in waiters:
                    self._reply(w.client, w.request_id, False, error="not_leader")
            self._pending.clear()
        self._was_leader = is_leader
        self._reschedule()

    def _reply(
        self,
        client: NodeId,
        request_id: int,
        ok: bool,
        result: Any = None,
        error: str | None = None,
    ) -> None:
        hint = self._core.leader_id if self._core is not None else None
        self.transport.send(client, ClientReply(request_id, ok, result, error, hint))

    def _reschedule(self) -> None:
        when = self.core.next_deadline()
        if self._timer is not None and when == self._timer_when:
            return
        if self._timer is not None:
            self._timer.cancel()
        assert self._loop is not None
        self._timer_when = when
        self._timer = self._loop.call_at(when, self._on_timer)
