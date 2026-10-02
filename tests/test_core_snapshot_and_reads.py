# mypy: disable-error-code="comparison-overlap,unreachable"
# (role changes inside step() are invisible to mypy's attribute narrowing)
"""Log compaction / InstallSnapshot (section 7) and ReadIndex reads (section 6.4)."""

from __future__ import annotations

import dataclasses

from harness import CONFIG, ManualCluster

from raftkv.core import Role
from raftkv.messages import AppendEntries, Entry, InstallSnapshot, InstallSnapshotReply
from raftkv.statemachine import KVStateMachine

SNAP = dataclasses.replace(CONFIG, snapshot_threshold=10)


def kv(core: object) -> dict[str, str]:
    sm = core.sm  # type: ignore[attr-defined]
    assert isinstance(sm, KVStateMachine)
    return sm.data


def test_log_is_compacted_after_threshold() -> None:
    c = ManualCluster(3, config=SNAP)
    leader = c.elect("s1")
    for i in range(35):
        c.propose("s1", f"v{i}")
        c.deliver()
    assert leader.log.snapshot_index >= 30
    assert len(leader.log) < 10
    assert leader.stats["snapshots_taken"] >= 3
    stored = c.storages["s1"].load()
    assert stored.snapshot_index == leader.log.snapshot_index
    assert stored.snapshot_data is not None


def test_lagging_follower_catches_up_via_install_snapshot() -> None:
    c = ManualCluster(3, config=SNAP)
    leader = c.elect("s1")
    c.crash("s3")
    for i in range(40):
        c.propose("s1", f"v{i}")
        c.deliver()
    assert leader.log.snapshot_index > 1  # s3's next entry has been compacted away
    c.restart("s3")
    sent: list[InstallSnapshot] = []

    def spy(e: tuple[str, str, object]) -> bool:
        if isinstance(e[2], InstallSnapshot):
            sent.append(e[2])
        return True

    for _ in range(3):
        c.advance(0.05)
        c.deliver(spy)
    assert sent, "expected the leader to send InstallSnapshot"
    assert c["s3"].stats["snapshots_installed"] == 1
    assert kv(c["s3"]) == kv(leader) == {"x": "v39"}
    assert c["s3"].log.last_index == leader.log.last_index


def test_restart_restores_state_machine_from_snapshot_and_log() -> None:
    c = ManualCluster(3, config=SNAP)
    c.elect("s1")
    for i in range(25):
        c.propose("s1", f"v{i}")
        c.deliver()
    c.advance(0.05)
    c.deliver()
    before = c["s2"].log.snapshot_index
    assert before > 0
    c.restart("s2")
    # On restart only the snapshot is applied; commit index is volatile and
    # is re-learned from the leader, after which the log suffix is replayed.
    assert c["s2"].last_applied == before
    c.advance(0.05)
    c.deliver()
    assert kv(c["s2"]) == {"x": "v24"}


def test_install_snapshot_keeps_matching_suffix() -> None:
    c = ManualCluster(3)
    f = c["s2"]
    f.step("s1", _append(1, [Entry(1, i, None) for i in range(1, 6)], commit=0), 0.0)
    f.drain_outbox()
    data = KVStateMachine().snapshot()
    f.step("s1", InstallSnapshot(1, "s1", 3, 1, data, 0), 0.0)
    assert f.log.snapshot_index == 3
    assert f.log.last_index == 5  # entries 4 and 5 retained
    assert f.commit_index == 3


def test_install_snapshot_discards_conflicting_log() -> None:
    c = ManualCluster(3)
    f = c["s2"]
    f.step("s1", _append(1, [Entry(1, i, None) for i in range(1, 6)], commit=0), 0.0)
    f.drain_outbox()
    f.step("s1", InstallSnapshot(2, "s1", 3, 2, KVStateMachine().snapshot(), 0), 0.0)
    assert f.log.snapshot_index == 3
    assert f.log.last_index == 3


def test_stale_install_snapshot_is_ignored() -> None:
    c = ManualCluster(3)
    f = c["s2"]
    f.step("s1", _append(1, [Entry(1, i, None) for i in range(1, 6)], commit=5), 0.0)
    f.drain_outbox()
    f.step("s1", InstallSnapshot(1, "s1", 2, 1, b"ignored", 0), 0.0)
    ((_, reply),) = f.drain_outbox()
    assert isinstance(reply, InstallSnapshotReply)
    assert f.log.snapshot_index == 0
    assert f.commit_index == 5


def _append(term: int, entries: list[Entry], commit: int) -> AppendEntries:
    return AppendEntries(term, "s1", 0, 0, entries, commit)


# ---------------------------------------------------------------------------
# ReadIndex
# ---------------------------------------------------------------------------


def test_read_waits_for_quorum_confirmation() -> None:
    c = ManualCluster(3)
    leader = c.elect("s1")
    c.propose("s1", "v")
    c.deliver()
    assert leader.request_read("r1")
    assert leader.ready_reads == []  # not yet confirmed
    c.deliver()
    assert leader.ready_reads == [("r1", leader.commit_index)]


def test_read_rejected_on_follower() -> None:
    c = ManualCluster(3)
    c.elect("s1")
    assert not c["s2"].request_read("r")


def test_read_deferred_until_noop_committed() -> None:
    c = ManualCluster(3)
    leader = c["s1"]
    leader.campaign()
    # Deliver votes but not AppendEntries: the leader's no-op stays uncommitted.
    c.deliver(lambda e: type(e[2]).__name__.startswith("RequestVote"))
    assert leader.role is Role.LEADER
    assert leader.request_read("r")
    c.deliver(lambda e: type(e[2]).__name__.startswith("RequestVote"))
    assert leader.ready_reads == []
    c.advance(0.05)  # heartbeat re-sends the no-op
    c.deliver()
    assert [ctx for ctx, _ in leader.ready_reads] == ["r"]
    assert leader.ready_reads[0][1] >= 1


def test_partitioned_leader_never_serves_read() -> None:
    """The classic stale-read scenario: a deposed leader that has not noticed yet."""
    c = ManualCluster(5)
    old = c.elect("s1")
    c.propose("s1", "old")
    c.deliver()
    # Partition s1 away; the majority elects a new leader and writes.
    majority = c.only_between("s2", "s3", "s4", "s5")
    assert old.request_read("stale?")
    for _ in range(100):
        c.advance(0.01)
        c.deliver(majority)
        if any(x.id != "s1" for x in c.leaders()):
            break
    new = next(x for x in c.leaders() if x.id != "s1")
    c.propose(new.id, "new")
    c.deliver(majority)
    assert old.role is Role.LEADER  # still believes it leads...
    assert old.ready_reads == []  # ...but could not confirm, so served nothing
    # Healing: the old leader learns the new term and fails the read.
    c.advance(0.05)
    c.deliver()
    assert old.role is Role.FOLLOWER
    assert old.failed_reads == ["stale?"]


def test_single_node_reads_immediately() -> None:
    c = ManualCluster(1)
    c.advance(0.31)
    assert c["s1"].request_read("r")
    assert c["s1"].ready_reads == [("r", 1)]
