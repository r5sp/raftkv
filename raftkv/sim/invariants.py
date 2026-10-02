"""Continuous checking of Raft's safety properties (Raft paper, Figure 3).

* **Election Safety** -- at most one leader per term. Checked at the instant
  any node becomes leader, over the whole run (including crashed nodes).
* **Leader Completeness** -- a new leader's log contains every entry that was
  committed before it was elected. Checked at the instant of election
  against every entry any node has applied so far.
* **State Machine Safety** -- no two nodes apply different entries at the
  same index. Checked on every apply.
* **Log Matching** -- if two logs contain an entry with the same index and
  term, the logs are identical up to that index. Checked periodically over
  every pair of live nodes.

(Leader Append-Only is enforced structurally: ``RaftLog`` refuses to
truncate below the commit index and the leader never truncates its own log.)

Any violation raises :class:`InvariantViolation` immediately, at the point
in the simulated execution where it happened.
"""

from __future__ import annotations

from collections.abc import Iterable

from raftkv.core import RaftCore
from raftkv.messages import Command, Entry, NodeId


class InvariantViolation(AssertionError):
    pass


class InvariantChecker:
    def __init__(self) -> None:
        self.leaders: dict[int, NodeId] = {}
        # index -> (term, command) for every entry any node has applied.
        self.committed: dict[int, tuple[int, Command | None]] = {}
        self.violations: list[str] = []
        self.checks = 0

    def _fail(self, message: str) -> None:
        self.violations.append(message)
        raise InvariantViolation(message)

    # -- RaftObserver hooks ------------------------------------------------------

    def on_become_leader(self, core: RaftCore) -> None:
        term = core.current_term
        prior = self.leaders.setdefault(term, core.id)
        if prior != core.id:
            self._fail(f"election safety: {prior} and {core.id} both leaders in term {term}")
        log = core.log
        for index, (term_c, _cmd) in self.committed.items():
            if index <= log.snapshot_index:
                continue  # compacted entries are committed by construction
            if log.term_at(index) != term_c:
                self._fail(
                    f"leader completeness: new leader {core.id} (term {term}) lacks committed "
                    f"entry {index}@{term_c}; has term {log.term_at(index)}"
                )

    def on_apply(self, core: RaftCore, entry: Entry) -> None:
        seen = self.committed.get(entry.index)
        record = (entry.term, entry.command)
        if seen is None:
            self.committed[entry.index] = record
        elif seen != record:
            self._fail(
                f"state machine safety: {core.id} applied {entry.index}@{entry.term} "
                f"{entry.command!r}, but {seen[0]} {seen[1]!r} was applied elsewhere"
            )

    # -- periodic checks -----------------------------------------------------------

    def check_log_matching(self, cores: Iterable[RaftCore]) -> None:
        self.checks += 1
        live = list(cores)
        for i, a in enumerate(live):
            for b in live[i + 1 :]:
                self._check_pair(a, b)

    def _check_pair(self, a: RaftCore, b: RaftCore) -> None:
        lo = max(a.log.snapshot_index, b.log.snapshot_index) + 1
        hi = min(a.log.last_index, b.log.last_index)
        # Find the highest common index where the terms agree...
        top = 0
        for idx in range(hi, lo - 1, -1):
            if a.log.term_at(idx) == b.log.term_at(idx):
                top = idx
                break
        # ...then every entry at or below it must be identical.
        for idx in range(lo, top + 1):
            ea, eb = a.log.entry(idx), b.log.entry(idx)
            if ea != eb:
                self._fail(
                    f"log matching: {a.id} and {b.id} agree at {top} but differ at {idx}: "
                    f"{ea} vs {eb}"
                )
