"""The Raft consensus algorithm as a deterministic, I/O-free state machine.

``RaftCore`` never touches sockets, clocks, or tasks. The runtime feeds it
inputs -- ``step(src, msg, now)`` for messages, ``tick(now)`` when a deadline
passes, ``propose``/``request_read`` for clients -- and drains its outputs:

* ``outbox``      -- ``(dest, message)`` pairs to send
* ``applied``     -- ``(index, term, result)`` for each newly applied command
* ``ready_reads`` -- ``(ctx, read_index)`` for linearizable reads that may now be served
* ``failed_reads``-- read contexts that must be retried elsewhere (lost leadership)

Durable state is written through ``Storage`` *synchronously, before* any
message that depends on it is placed in the outbox.

This separation (as in etcd/raft) is what makes the algorithm unit-testable
message-by-message, e.g. the Figure 8 scenario in ``tests/test_core_replication.py``.

References to "Figure 2" and section numbers are to Ongaro & Ousterhout,
"In Search of an Understandable Consensus Algorithm (Extended Version)", 2014.
"""

from __future__ import annotations

import enum
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from raftkv.log import RaftLog
from raftkv.messages import (
    AppendEntries,
    AppendEntriesReply,
    Command,
    Entry,
    InstallSnapshot,
    InstallSnapshotReply,
    Message,
    NodeId,
    RequestVote,
    RequestVoteReply,
)
from raftkv.storage import Storage

# Slack for float comparisons between the runtime clock and core deadlines.
_EPS = 1e-6


class Role(enum.Enum):
    FOLLOWER = "follower"
    CANDIDATE = "candidate"
    LEADER = "leader"


@dataclass(frozen=True)
class RaftConfig:
    election_timeout_min: float = 0.150
    election_timeout_max: float = 0.300
    heartbeat_interval: float = 0.050
    max_entries_per_append: int = 128
    # Take a snapshot once this many applied entries sit in the log; 0 disables.
    snapshot_threshold: int = 1000

    def __post_init__(self) -> None:
        if not 0 < self.heartbeat_interval < self.election_timeout_min:
            raise ValueError("heartbeat interval must be below the election timeout")
        if self.election_timeout_max < self.election_timeout_min:
            raise ValueError("election_timeout_max < election_timeout_min")


class StateMachine(Protocol):
    def apply(self, index: int, command: Command) -> Any: ...

    def snapshot(self) -> bytes: ...

    def restore(self, data: bytes) -> None: ...


class RaftObserver(Protocol):
    """Hooks used by the simulator's invariant checker. Not needed in production."""

    def on_become_leader(self, core: RaftCore) -> None: ...

    def on_apply(self, core: RaftCore, entry: Entry) -> None: ...


@dataclass
class _PendingRead:
    ctx: object
    read_index: int
    seq: int


class RaftCore:
    def __init__(
        self,
        node_id: NodeId,
        members: Sequence[NodeId],
        storage: Storage,
        state_machine: StateMachine,
        *,
        config: RaftConfig | None = None,
        rng: random.Random | None = None,
        now: float = 0.0,
        observer: RaftObserver | None = None,
    ) -> None:
        if node_id not in members:
            raise ValueError(f"{node_id!r} is not in the member list {list(members)!r}")
        self.id = node_id
        self.peers: list[NodeId] = sorted(m for m in set(members) if m != node_id)
        self.cluster_size = len(self.peers) + 1
        self.quorum = self.cluster_size // 2 + 1
        self.config = config or RaftConfig()
        self.storage = storage
        self.sm = state_machine
        self.rng = rng or random.Random()
        self.observer = observer

        # Persistent state (Figure 2), restored from storage.
        st = storage.load()
        self.current_term = st.term
        self.voted_for = st.voted_for
        self.log = RaftLog(st.snapshot_index, st.snapshot_term, st.entries)
        self.snapshot_data = st.snapshot_data
        if st.snapshot_data is not None:
            self.sm.restore(st.snapshot_data)

        # Volatile state on all servers.
        self.commit_index = self.log.snapshot_index
        self.last_applied = self.log.snapshot_index
        self.role = Role.FOLLOWER
        self.leader_id: NodeId | None = None

        # Volatile leader / candidate state.
        self.next_index: dict[NodeId, int] = {}
        self.match_index: dict[NodeId, int] = {}
        self._sent_upto: dict[NodeId, int] = {}
        self._acked_seq: dict[NodeId, int] = {}
        self._votes: dict[NodeId, bool] = {}
        self._seq = 0
        self._dirty = False
        self._need_heartbeat = False
        self._pending_reads: list[_PendingRead] = []
        self._reads_awaiting_noop: list[object] = []

        # Outputs drained by the runtime.
        self.outbox: list[tuple[NodeId, Message]] = []
        self.applied: list[tuple[int, int, Any]] = []
        self.ready_reads: list[tuple[object, int]] = []
        self.failed_reads: list[object] = []

        self.stats: dict[str, int] = {
            "elections": 0,
            "terms_led": 0,
            "snapshots_taken": 0,
            "snapshots_installed": 0,
            "snapshots_sent": 0,
        }

        self._now = now
        self._election_deadline = self._random_election_deadline()
        self._heartbeat_due = float("inf")

    # ------------------------------------------------------------------
    # Public inputs
    # ------------------------------------------------------------------

    def tick(self, now: float) -> None:
        """Advance time; fires election timeouts and leader heartbeats."""
        self._now = now
        if self.role is Role.LEADER:
            if now + _EPS >= self._heartbeat_due:
                self._broadcast_append(heartbeat=True)
        elif now + _EPS >= self._election_deadline:
            self.campaign()

    def next_deadline(self) -> float:
        """Time at which ``tick`` next needs to be called."""
        return self._heartbeat_due if self.role is Role.LEADER else self._election_deadline

    def step(self, src: NodeId, msg: Message, now: float) -> None:
        """Process one incoming Raft RPC or reply."""
        self._now = now
        term = getattr(msg, "term", None)
        if term is None:
            return
        # All servers: a higher term means our view is stale; revert to follower.
        if term > self.current_term:
            self._become_follower(term, None)

        if isinstance(msg, AppendEntries):
            self._on_append_entries(src, msg)
        elif isinstance(msg, AppendEntriesReply):
            self._on_append_reply(msg)
        elif isinstance(msg, RequestVote):
            self._on_request_vote(src, msg)
        elif isinstance(msg, RequestVoteReply):
            self._on_vote_reply(msg)
        elif isinstance(msg, InstallSnapshot):
            self._on_install_snapshot(src, msg)
        elif isinstance(msg, InstallSnapshotReply):
            self._on_snapshot_reply(msg)

    def propose(self, command: Command) -> int | None:
        """Append a client command to the leader's log; returns its index.

        Returns ``None`` if this node is not the leader. The entry is sent to
        followers on the next ``flush`` so that a burst of proposals shares one
        AppendEntries round.
        """
        if self.role is not Role.LEADER:
            return None
        entry = Entry(self.current_term, self.log.last_index + 1, command)
        self._append([entry])
        self._dirty = True
        if self.cluster_size == 1:
            self._advance_commit(entry.index)
        return entry.index

    def request_read(self, ctx: object) -> bool:
        """Begin a linearizable read using the ReadIndex protocol (section 6.4).

        1. The leader must have committed an entry from its current term (its
           no-op) so that its commit index is at least that of any prior leader.
        2. It records ``read_index = commit_index``.
        3. It confirms it is still leader by collecting heartbeat acks, sent
           *after* the read arrived, from a quorum.
        4. Once applied up to ``read_index`` the read is served from local state.

        Returns ``False`` immediately if this node is not the leader.
        """
        if self.role is not Role.LEADER:
            return False
        if self.log.term_at(self.commit_index) != self.current_term:
            self._reads_awaiting_noop.append(ctx)
        else:
            self._register_read(ctx)
        return True

    def flush(self) -> None:
        """Send any AppendEntries made necessary by proposals or reads."""
        if self.role is Role.LEADER and (self._dirty or self._need_heartbeat):
            self._broadcast_append(heartbeat=self._need_heartbeat)

    def campaign(self) -> None:
        """Start an election (section 5.2). Called on election timeout."""
        self.role = Role.CANDIDATE
        self.current_term += 1
        self.voted_for = self.id
        self.leader_id = None
        self._persist_hard_state()
        self._votes = {self.id: True}
        self._election_deadline = self._random_election_deadline()
        self.stats["elections"] += 1
        if len(self._votes) >= self.quorum:
            self._become_leader()
            return
        req = RequestVote(self.current_term, self.id, self.log.last_index, self.log.last_term)
        for p in self.peers:
            self._send(p, req)

    def take_snapshot(self) -> None:
        """Compact the log up to ``last_applied`` (section 7)."""
        index = self.last_applied
        if index <= self.log.snapshot_index:
            return
        term = self.log.term_at(index)
        assert term is not None
        data = self.sm.snapshot()
        keep = self.log.slice(index + 1, self.log.last_index + 1)
        self.storage.save_snapshot(index, term, data, keep)
        self.log.compact(index, term)
        self.snapshot_data = data
        self.stats["snapshots_taken"] += 1

    # ------------------------------------------------------------------
    # Role transitions
    # ------------------------------------------------------------------

    def _become_follower(self, term: int, leader: NodeId | None) -> None:
        was_leader = self.role is Role.LEADER
        if term > self.current_term:
            self.current_term = term
            self.voted_for = None
            self._persist_hard_state()
        self.role = Role.FOLLOWER
        self.leader_id = leader
        if was_leader:
            self._heartbeat_due = float("inf")
            self._election_deadline = self._random_election_deadline()
            self.failed_reads.extend(r.ctx for r in self._pending_reads)
            self.failed_reads.extend(self._reads_awaiting_noop)
            self._pending_reads.clear()
            self._reads_awaiting_noop.clear()
            self._dirty = self._need_heartbeat = False

    def _become_leader(self) -> None:
        self.role = Role.LEADER
        self.leader_id = self.id
        self.stats["terms_led"] += 1
        last = self.log.last_index
        for p in self.peers:
            self.next_index[p] = last + 1
            self.match_index[p] = 0
            self._sent_upto[p] = 0
            self._acked_seq[p] = 0
        if self.observer is not None:
            self.observer.on_become_leader(self)
        # A no-op entry from the new term lets the leader learn which entries
        # are committed (section 8) and is a prerequisite for ReadIndex.
        noop = Entry(self.current_term, last + 1, None)
        self._append([noop])
        if self.cluster_size == 1:
            self._advance_commit(noop.index)
        self._broadcast_append(heartbeat=True)

    # ------------------------------------------------------------------
    # Elections
    # ------------------------------------------------------------------

    def _candidate_log_ok(self, m: RequestVote) -> bool:
        # Section 5.4.1: grant only if the candidate's log is at least as
        # up-to-date as ours (compare last term, then length).
        if m.last_log_term != self.log.last_term:
            return m.last_log_term > self.log.last_term
        return m.last_log_index >= self.log.last_index

    def _on_request_vote(self, src: NodeId, m: RequestVote) -> None:
        granted = False
        if (
            m.term == self.current_term
            and self.voted_for in (None, m.candidate_id)
            and self._candidate_log_ok(m)
        ):
            granted = True
            if self.voted_for != m.candidate_id:
                self.voted_for = m.candidate_id
                self._persist_hard_state()
            self._election_deadline = self._random_election_deadline()
        self._send(src, RequestVoteReply(self.current_term, self.id, granted))

    def _on_vote_reply(self, m: RequestVoteReply) -> None:
        if self.role is not Role.CANDIDATE or m.term != self.current_term or not m.granted:
            return
        self._votes[m.voter_id] = True
        if len(self._votes) >= self.quorum:
            self._become_leader()

    # ------------------------------------------------------------------
    # Log replication: follower side
    # ------------------------------------------------------------------

    def _accept_leader(self, leader_id: NodeId) -> None:
        if self.role is not Role.FOLLOWER:
            if self.role is Role.LEADER:
                # Two leaders in one term would violate Election Safety.
                raise AssertionError(
                    f"{self.id}: second leader {leader_id} in term {self.current_term}"
                )
            self._become_follower(self.current_term, leader_id)
        self.leader_id = leader_id
        self._election_deadline = self._random_election_deadline()

    def _on_append_entries(self, src: NodeId, m: AppendEntries) -> None:
        if m.term < self.current_term:
            self._send(src, AppendEntriesReply(self.current_term, self.id, False, seq=m.seq))
            return
        self._accept_leader(m.leader_id)

        prev_index, prev_term, entries = m.prev_log_index, m.prev_log_term, m.entries
        if prev_index < self.log.snapshot_index:
            # The prefix up to our snapshot is committed and therefore matches
            # the leader; skip the part of the message it covers.
            skip = self.log.snapshot_index - prev_index
            if skip >= len(entries):
                match = prev_index + len(entries)
                self._send(src, self._ae_ok(match, m.seq))
                return
            entries = entries[skip:]
            prev_index, prev_term = self.log.snapshot_index, self.log.snapshot_term

        local_prev_term = self.log.term_at(prev_index)
        if local_prev_term is None:
            # Our log is too short: tell the leader where it ends.
            reply = AppendEntriesReply(
                self.current_term, self.id, False, conflict_index=self.log.last_index + 1, seq=m.seq
            )
            self._send(src, reply)
            return
        if local_prev_term != prev_term:
            # Conflict: report the term and the first index of that term so the
            # leader can skip the whole term in one round trip.
            reply = AppendEntriesReply(
                self.current_term,
                self.id,
                False,
                conflict_index=self.log.first_index_of_term_run(prev_index),
                conflict_term=local_prev_term,
                seq=m.seq,
            )
            self._send(src, reply)
            return

        # Consistency check passed. Append new entries, truncating only on an
        # actual conflict -- never because a (possibly reordered, stale)
        # message is shorter than our log.
        for i, e in enumerate(entries):
            local = self.log.term_at(e.index)
            if local == e.term:
                continue
            if local is not None:
                if e.index <= self.commit_index:
                    raise AssertionError(
                        f"{self.id}: attempt to truncate committed entry {e.index}"
                    )
                self.log.truncate_from(e.index)
                self.storage.truncate_from(e.index)
            self._append(entries[i:])
            break

        match = prev_index + len(entries)
        if m.leader_commit > self.commit_index:
            self._advance_commit(min(m.leader_commit, match))
        self._send(src, self._ae_ok(match, m.seq))

    def _ae_ok(self, match: int, seq: int) -> AppendEntriesReply:
        return AppendEntriesReply(self.current_term, self.id, True, match_index=match, seq=seq)

    def _on_install_snapshot(self, src: NodeId, m: InstallSnapshot) -> None:
        if m.term < self.current_term:
            self._send(src, InstallSnapshotReply(self.current_term, self.id, 0, m.seq))
            return
        self._accept_leader(m.leader_id)
        idx, term = m.last_included_index, m.last_included_term
        if idx > self.commit_index:
            # Section 7: keep the log suffix only if it extends the snapshot.
            if self.log.term_at(idx) == term:
                keep = self.log.slice(idx + 1, self.log.last_index + 1)
            else:
                keep = []
            self.storage.save_snapshot(idx, term, m.data, keep)
            self.log = RaftLog(idx, term, keep)
            self.snapshot_data = m.data
            self.sm.restore(m.data)
            self.commit_index = self.last_applied = idx
            self.stats["snapshots_installed"] += 1
        self._send(src, InstallSnapshotReply(self.current_term, self.id, idx, m.seq))

    # ------------------------------------------------------------------
    # Log replication: leader side
    # ------------------------------------------------------------------

    def _broadcast_append(self, heartbeat: bool) -> None:
        for p in self.peers:
            self._send_append(p, force=heartbeat)
        if heartbeat:
            self._heartbeat_due = self._now + self.config.heartbeat_interval
        self._dirty = self._need_heartbeat = False

    def _send_append(self, peer: NodeId, force: bool) -> None:
        next_idx = self.next_index[peer]
        if next_idx <= self.log.snapshot_index:
            # The entries the follower needs have been compacted away.
            if force or self._sent_upto[peer] < self.log.snapshot_index:
                assert self.snapshot_data is not None
                self._sent_upto[peer] = self.log.snapshot_index
                self.stats["snapshots_sent"] += 1
                msg = InstallSnapshot(
                    self.current_term,
                    self.id,
                    self.log.snapshot_index,
                    self.log.snapshot_term,
                    self.snapshot_data,
                    self._seq,
                )
                self._send(peer, msg)
            return
        if not force and self._sent_upto[peer] >= self.log.last_index:
            return  # everything is already in flight; wait for the reply
        prev = next_idx - 1
        prev_term = self.log.term_at(prev)
        assert prev_term is not None
        entries = self.log.slice(next_idx, next_idx + self.config.max_entries_per_append)
        self._sent_upto[peer] = prev + len(entries)
        self._send(
            peer,
            AppendEntries(
                self.current_term, self.id, prev, prev_term, entries, self.commit_index, self._seq
            ),
        )

    def _on_append_reply(self, m: AppendEntriesReply) -> None:
        if self.role is not Role.LEADER or m.term != self.current_term:
            return
        peer = m.follower_id
        if peer not in self.next_index:
            return
        self._acked_seq[peer] = max(self._acked_seq[peer], m.seq)
        if m.success:
            if m.match_index > self.match_index[peer]:
                self.match_index[peer] = m.match_index
                self._maybe_commit()
            self.next_index[peer] = max(self.next_index[peer], self.match_index[peer] + 1)
            if self.next_index[peer] <= self.log.last_index:
                self._send_append(peer, force=False)
        else:
            if m.conflict_term is not None:
                last = self.log.last_index_of_term(m.conflict_term)
                new_next = last + 1 if last is not None else m.conflict_index
            else:
                new_next = m.conflict_index
            new_next = min(max(new_next, self.match_index[peer] + 1), self.log.last_index + 1)
            if new_next < self.next_index[peer]:  # ignore stale/duplicate rejections
                self.next_index[peer] = new_next
                self._sent_upto[peer] = new_next - 1
                self._send_append(peer, force=True)
        self._check_reads()

    def _on_snapshot_reply(self, m: InstallSnapshotReply) -> None:
        if self.role is not Role.LEADER or m.term != self.current_term:
            return
        peer = m.follower_id
        if peer not in self.next_index:
            return
        self._acked_seq[peer] = max(self._acked_seq[peer], m.seq)
        if m.last_included_index > self.match_index[peer]:
            self.match_index[peer] = m.last_included_index
            self._maybe_commit()
        self.next_index[peer] = max(self.next_index[peer], self.match_index[peer] + 1)
        if self.next_index[peer] <= self.log.last_index:
            self._send_append(peer, force=False)
        self._check_reads()

    def _maybe_commit(self) -> None:
        # The highest index replicated on a quorum (leader counts itself).
        matches = sorted([self.log.last_index, *self.match_index.values()], reverse=True)
        n = matches[self.quorum - 1]
        # Section 5.4.2 / Figure 8: only entries from the current term are
        # committed by counting replicas; earlier ones commit indirectly.
        if n > self.commit_index and self.log.term_at(n) == self.current_term:
            self._advance_commit(n)

    # ------------------------------------------------------------------
    # Commit, apply, compaction
    # ------------------------------------------------------------------

    def _advance_commit(self, index: int) -> None:
        if index <= self.commit_index:
            return
        self.commit_index = index
        while self.last_applied < self.commit_index:
            self.last_applied += 1
            e = self.log.entry(self.last_applied)
            result = None if e.command is None else self.sm.apply(e.index, e.command)
            self.applied.append((e.index, e.term, result))
            if self.observer is not None:
                self.observer.on_apply(self, e)
        if (
            self.role is Role.LEADER
            and self._reads_awaiting_noop
            and self.log.term_at(self.commit_index) == self.current_term
        ):
            waiting, self._reads_awaiting_noop = self._reads_awaiting_noop, []
            for ctx in waiting:
                self._register_read(ctx)
        threshold = self.config.snapshot_threshold
        if threshold and self.last_applied - self.log.snapshot_index >= threshold:
            self.take_snapshot()

    # ------------------------------------------------------------------
    # ReadIndex
    # ------------------------------------------------------------------

    def _register_read(self, ctx: object) -> None:
        if self.cluster_size == 1:
            self.ready_reads.append((ctx, self.commit_index))
            return
        self._seq += 1
        self._pending_reads.append(_PendingRead(ctx, self.commit_index, self._seq))
        self._need_heartbeat = True

    def _check_reads(self) -> None:
        if not self._pending_reads:
            return
        acked = sorted(self._acked_seq.values(), reverse=True)
        # The leader implicitly acks itself, so it needs quorum-1 peers.
        quorum_seq = acked[self.quorum - 2]
        while self._pending_reads and self._pending_reads[0].seq <= quorum_seq:
            r = self._pending_reads.pop(0)
            self.ready_reads.append((r.ctx, r.read_index))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _append(self, entries: list[Entry]) -> None:
        if not entries:
            return
        self.log.append(entries)
        self.storage.append(entries)

    def _persist_hard_state(self) -> None:
        self.storage.save_hard_state(self.current_term, self.voted_for)

    def _send(self, dest: NodeId, msg: Message) -> None:
        self.outbox.append((dest, msg))

    def _random_election_deadline(self) -> float:
        c = self.config
        return self._now + self.rng.uniform(c.election_timeout_min, c.election_timeout_max)

    def drain_outbox(self) -> list[tuple[NodeId, Message]]:
        out, self.outbox = self.outbox, []
        return out

    def describe(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "role": self.role.value,
            "term": self.current_term,
            "leader": self.leader_id,
            "commit_index": self.commit_index,
            "last_applied": self.last_applied,
            "last_index": self.log.last_index,
            "snapshot_index": self.log.snapshot_index,
        }


ApplyCallback = Callable[[int, int, Any], None]
