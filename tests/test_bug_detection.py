"""Testing the tests: deliberately broken Raft variants must be caught.

A fault-injection suite that never fails proves little. Each test here
monkeypatches a classic Raft bug into the implementation and asserts that the
randomized simulation (invariants + linearizability checker) detects it within
a bounded number of seeds. If one of these ever starts passing silently, the
test harness has lost its teeth.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable

import pytest

from raftkv.core import RaftCore, Role
from raftkv.linearizability import check_history
from raftkv.messages import AppendEntries, ClientRequest, NodeId, RequestVote
from raftkv.node import RaftNode
from raftkv.sim import run_simulation
from raftkv.sim.workload import ScenarioConfig, run_fault_scenario
from raftkv.statemachine import KVStateMachine

CFG = ScenarioConfig(nodes=5, clients=5, keys=2, duration=8.0)


def bug_detected(seeds: Iterable[int | str], config: ScenarioConfig = CFG) -> str | None:
    """Return a description of the first failure found, or None if all seeds pass."""
    for seed in seeds:
        try:
            result = run_simulation(lambda s=seed: run_fault_scenario(s, config))  # type: ignore[misc]
        except Exception as exc:  # divergence, liveness failure, assertion in core...
            return f"seed {seed}: {type(exc).__name__}: {exc}"
        if result.cluster.checker.violations:
            return f"seed {seed}: {result.cluster.checker.violations[0]}"
        if result.cluster.errors:
            return f"seed {seed}: {result.cluster.errors[0]}"
        if not check_history(result.history).ok:
            return f"seed {seed}: non-linearizable history"
    return None


def test_reads_served_without_readindex_are_caught(monkeypatch: pytest.MonkeyPatch) -> None:
    """Leader answers reads from local state without confirming leadership.

    A deposed leader that has not yet heard of the new term then serves stale
    data. This is the bug ReadIndex (or a correctly-bounded lease) prevents.
    """
    original = RaftNode._on_client_request

    def buggy(self: RaftNode, src: NodeId, req: ClientRequest) -> None:
        if req.op.get("kind") == "get" and self.core.role is Role.LEADER:
            self._reply(src, req.request_id, True, self.sm.get(str(req.op["key"])))
            return
        original(self, src, req)

    monkeypatch.setattr(RaftNode, "_on_client_request", buggy)
    assert bug_detected(range(40)) is not None


def test_missing_vote_persistence_is_caught(monkeypatch: pytest.MonkeyPatch) -> None:
    """votedFor/currentTerm kept only in memory: a restarted node can vote twice in a term."""
    monkeypatch.setattr(RaftCore, "_persist_hard_state", lambda self: None)
    crashy = dataclasses.replace(CFG, nodes=3, clients=3)
    assert bug_detected((f"vote-{i}" for i in range(80)), crashy) is not None


def test_missing_log_up_to_date_check_is_caught(monkeypatch: pytest.MonkeyPatch) -> None:
    """Voting for any candidate regardless of log freshness breaks Leader Completeness."""

    def always_ok(self: RaftCore, m: RequestVote) -> bool:
        return True

    monkeypatch.setattr(RaftCore, "_candidate_log_ok", always_ok)
    assert bug_detected(range(40)) is not None


def test_missing_session_dedup_is_caught(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without (client_id, seq) dedup, a retried write executes twice."""

    def no_dedup(self: KVStateMachine, index: int, command: dict[str, object]) -> object:
        self.last_applied = index
        return self.execute(command["op"])  # type: ignore[arg-type]

    monkeypatch.setattr(KVStateMachine, "apply", no_dedup)
    monkeypatch.setattr(KVStateMachine, "cached_result", lambda self, c, s: None)
    assert bug_detected(range(40)) is not None


def test_counting_old_term_replicas_fails_figure8(monkeypatch: pytest.MonkeyPatch) -> None:
    """Committing a previous-term entry by counting replicas (the Figure 8 bug)."""

    def buggy_commit(self: RaftCore) -> None:
        matches = sorted([self.log.last_index, *self.match_index.values()], reverse=True)
        n = matches[self.quorum - 1]
        if n > self.commit_index:
            self._advance_commit(n)

    monkeypatch.setattr(RaftCore, "_maybe_commit", buggy_commit)
    import test_core_replication as fig8

    with pytest.raises(AssertionError):
        fig8.test_figure8_old_term_entry_on_majority_is_not_committed()


def test_truncating_on_stale_append_is_caught(monkeypatch: pytest.MonkeyPatch) -> None:
    """Follower blindly replaces its suffix with each message's entries.

    With reordered/duplicated delivery, a stale AppendEntries then deletes
    entries the follower already acknowledged -- possibly committed ones.
    """
    original = RaftCore._on_append_entries

    def buggy(self: RaftCore, src: NodeId, m: AppendEntries) -> None:
        if (
            m.term == self.current_term
            and self.log.snapshot_index <= m.prev_log_index < self.log.last_index
            and self.log.term_at(m.prev_log_index) == m.prev_log_term
            and m.prev_log_index + 1 > self.log.snapshot_index
        ):
            # Ignores the commit-index guard and truncates unconditionally.
            self.log.truncate_from(m.prev_log_index + 1)
            self.storage.truncate_from(m.prev_log_index + 1)
            if self.commit_index > self.log.last_index:
                self.commit_index = self.log.last_index
        original(self, src, m)

    monkeypatch.setattr(RaftCore, "_on_append_entries", buggy)
    assert bug_detected(range(40)) is not None
