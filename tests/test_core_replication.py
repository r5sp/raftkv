# mypy: disable-error-code="comparison-overlap,unreachable"
# (role changes inside step() are invisible to mypy's attribute narrowing)
"""Log replication, the commitment rule, and the Figure 8 scenario."""

from __future__ import annotations

import dataclasses
import random
from collections.abc import Callable

from harness import CONFIG, Envelope, ManualCluster, storage_with

from raftkv.core import Role
from raftkv.messages import AppendEntries, AppendEntriesReply, Entry, Message


def test_replicates_and_commits_on_all_nodes() -> None:
    c = ManualCluster(3)
    leader = c.elect("s1")
    for v in "abc":
        c.propose("s1", v)
    c.deliver()
    assert leader.commit_index == 4  # no-op + 3 commands
    # Followers learn the commit index from the next AppendEntries.
    c.advance(0.05)
    c.deliver()
    for i in c.ids:
        assert c.log_terms(i) == [1, 1, 1, 1]
        assert c[i].commit_index == 4
        assert c[i].sm.data == {"x": "c"}  # type: ignore[attr-defined]


def test_entries_persisted_before_ack() -> None:
    c = ManualCluster(3)
    c.elect("s1")
    c.propose("s1", "v")
    c.deliver()
    for i in c.ids:
        assert [e.term for e in c.storages[i].load().entries] == [1, 1]


def test_no_commit_without_majority() -> None:
    c = ManualCluster(5)
    leader = c.elect("s1")
    noop_commit = leader.commit_index
    c.propose("s1", "v")
    c.deliver(c.only_between("s1", "s2"))  # leader + 1 follower = 2 of 5
    assert leader.commit_index == noop_commit
    c.deliver(c.only_between("s1", "s2", "s3"))
    c.advance(0.05)  # heartbeat re-sends to s3
    c.deliver(c.only_between("s1", "s2", "s3"))
    assert leader.commit_index == 2


def test_follower_reports_short_log() -> None:
    c = ManualCluster(3, storages={"s2": storage_with(1, None, [1])})
    c["s2"].step("s1", AppendEntries(1, "s1", 5, 1, [], 0), 0.0)
    ((_, reply),) = c["s2"].drain_outbox()
    assert isinstance(reply, AppendEntriesReply)
    assert (reply.success, reply.conflict_index, reply.conflict_term) == (False, 2, None)


def test_follower_reports_conflicting_term_run() -> None:
    c = ManualCluster(3, storages={"s2": storage_with(3, None, [1, 2, 2, 2, 3])})
    c["s2"].step("s1", AppendEntries(4, "s1", 4, 4, [], 0), 0.0)
    ((_, reply),) = c["s2"].drain_outbox()
    assert isinstance(reply, AppendEntriesReply)
    # Term at index 4 is 2; the run of term 2 starts at index 2.
    assert (reply.success, reply.conflict_index, reply.conflict_term) == (False, 2, 2)


def test_fast_backtracking_skips_whole_terms() -> None:
    """A follower with a long divergent suffix is repaired in O(terms), not O(entries)."""
    storages = {
        "s1": storage_with(3, None, [1] + [3] * 30),
        "s2": storage_with(3, None, [1] + [3] * 30),
        "s3": storage_with(2, None, [1] + [2] * 40),
    }
    c = ManualCluster(3, storages=storages)
    rejections = 0

    def count(e: Envelope) -> bool:
        nonlocal rejections
        if isinstance(e[2], AppendEntriesReply) and not e[2].success and e[0] == "s3":
            rejections += 1
        return True

    c.elect("s1", keep=count)
    for _ in range(5):
        c.advance(0.05)
        c.deliver(count)
    assert c.log_terms("s3") == c.log_terms("s1")
    assert rejections <= 2


def test_conflicting_uncommitted_suffix_is_overwritten() -> None:
    storages = {
        "s1": storage_with(3, None, [1, 3]),
        "s2": storage_with(3, None, [1, 3]),
        "s3": storage_with(2, None, [1, 2, 2, 2]),
    }
    c = ManualCluster(3, storages=storages)
    c.elect("s1")
    c.advance(0.05)
    c.deliver()
    assert c.log_terms("s3") == [1, 3, 4]
    assert [e.term for e in c.storages["s3"].load().entries] == [1, 3, 4]


def test_stale_append_entries_do_not_truncate() -> None:
    """A delayed, shorter AppendEntries must not remove later matching entries."""
    c = ManualCluster(3)
    c.elect("s1")
    c.propose("s1", "a")
    c.propose("s1", "b")
    c.deliver()
    assert c.log_terms("s2") == [1, 1, 1]
    old = AppendEntries(1, "s1", 0, 0, [c["s1"].log.entry(1)], 0)
    c["s2"].step("s1", old, 0.0)
    ((_, reply),) = c["s2"].drain_outbox()
    assert isinstance(reply, AppendEntriesReply)
    assert reply.success
    assert reply.match_index == 1
    assert c.log_terms("s2") == [1, 1, 1]


def test_follower_commit_bounded_by_verified_prefix() -> None:
    c = ManualCluster(3, storages={"s2": storage_with(1, None, [1, 1, 1])})
    # Leader claims commit 3 but only vouches for entries up to index 1.
    c["s2"].step("s1", AppendEntries(1, "s1", 0, 0, [Entry(1, 1, None)], 3), 0.0)
    assert c["s2"].commit_index == 1


def test_leader_commit_survives_leader_crash() -> None:
    c = ManualCluster(3)
    c.elect("s1")
    c.propose("s1", "durable")
    c.deliver()
    committed = c["s1"].commit_index
    c.crash("s1")
    for _ in range(100):
        c.advance(0.01)
        c.deliver()
        if c.leaders():
            break
    new = c.leaders()[0]
    assert new.id != "s1"
    assert new.log.term_at(committed) == 1
    c.advance(0.05)
    c.deliver()
    assert new.sm.data == {"x": "durable"}  # type: ignore[attr-defined]


def test_replication_tolerates_duplication_and_reordering() -> None:
    rng = random.Random(7)
    c = ManualCluster(5)
    c.elect("s1")
    for step in range(200):
        if step % 3 == 0:
            c.propose("s1", f"v{step}")
        c.collect()
        batch, c.queue = c.queue, []
        batch = batch + rng.sample(batch, k=len(batch) // 3)  # duplicates
        rng.shuffle(batch)  # reordering
        for src, dst, msg in batch:
            if rng.random() < 0.1:
                continue  # loss
            c[dst].step(src, msg, c.now)
        c.advance(0.01)
    for _ in range(20):
        c.advance(0.05)
        c.deliver()
    assert len({tuple(c.log_terms(i)) for i in c.ids}) == 1
    assert len({c[i].commit_index for i in c.ids}) == 1
    assert c["s1"].role is Role.LEADER


# ---------------------------------------------------------------------------
# Figure 8 of the Raft paper: why a leader may not count replicas of entries
# from *earlier* terms to decide commitment.
# ---------------------------------------------------------------------------


def figure8_cluster() -> ManualCluster:
    """State after Figure 8 (b): S1 replicated 2@2 to S2 only, S5 has 2@3 from term 3."""
    storages = {
        "s1": storage_with(2, "s1", [1, 2]),
        "s2": storage_with(2, "s1", [1, 2]),
        "s3": storage_with(3, "s5", [1]),
        "s4": storage_with(3, "s5", [1]),
        "s5": storage_with(3, "s5", [1, 3]),
    }
    config = dataclasses.replace(CONFIG, max_entries_per_append=1)
    c = ManualCluster(5, storages=storages, config=config)
    c.crash("s5")
    return c


def without_term(term: int) -> Callable[[Envelope], Message]:
    """Strip entries of ``term`` from AppendEntries in transit (a shorter batch)."""

    def mutate(e: Envelope) -> Message:
        msg = e[2]
        if isinstance(msg, AppendEntries):
            kept = [x for x in msg.entries if x.term != term]
            return dataclasses.replace(msg, entries=kept)
        return msg

    return mutate


def figure8_step_c(c: ManualCluster) -> None:
    """S1 becomes leader of term 4 and replicates 2@2 to S3 (but not its 3@4)."""
    group = c.only_between("s1", "s2", "s3")
    c["s1"].campaign()  # term 3: S3 already voted for S5 in term 3 -> loses
    c.deliver(group)
    assert c["s1"].role is Role.CANDIDATE
    c["s1"].campaign()  # term 4
    c.deliver(group, mutate=without_term(4))
    c.advance(0.05)
    c.deliver(group, mutate=without_term(4))
    assert c["s1"].role is Role.LEADER
    assert c["s1"].current_term == 4


def test_figure8_old_term_entry_on_majority_is_not_committed() -> None:
    c = figure8_cluster()
    figure8_step_c(c)
    # 2@2 is now stored on S1, S2, S3 -- a majority of five...
    for i in ("s1", "s2", "s3"):
        assert c[i].log.term_at(2) == 2
    # ...but it is from term 2, so the term-4 leader must not count it.
    assert c["s1"].commit_index < 2
    assert c["s1"].log.term_at(3) == 4  # its own no-op, not yet replicated


def test_figure8_uncommitted_entry_may_be_overwritten() -> None:
    """(d): S1 crashes, S5 is elected and overwrites 2@2 everywhere -- safely."""
    c = figure8_cluster()
    figure8_step_c(c)
    c.crash("s1")
    c.restart("s5")
    group = c.only_between("s2", "s3", "s4", "s5")
    c["s5"].campaign()  # term 4: S2/S3 voted for S1 in term 4
    c.deliver(group)
    if c["s5"].role is not Role.LEADER:
        c["s5"].campaign()  # term 5: S5's last term (3) beats S2/S3's (2)
        c.deliver(group)
    assert c["s5"].role is Role.LEADER
    for _ in range(5):
        c.advance(0.05)
        c.deliver(group)
    for i in ("s2", "s3", "s4", "s5"):
        assert c[i].log.term_at(2) == 3, i
    assert c["s5"].commit_index == 3


def test_figure8_current_term_entry_commits_both_and_blocks_s5() -> None:
    """(e): S1 replicates 3@4 to a majority, committing 2@2 indirectly; S5 can never win."""
    c = figure8_cluster()
    figure8_step_c(c)
    group = c.only_between("s1", "s2", "s3")
    c.advance(0.05)
    c.deliver(group)  # now without stripping: 3@4 reaches S2 and S3
    assert c["s1"].commit_index == 3
    c.crash("s1")
    c.restart("s5")
    for _ in range(300):
        c.advance(0.01)
        c.deliver()
        assert c["s5"].role is not Role.LEADER
        if c.leaders():
            break
    leader = c.leaders()[0]
    assert leader.id in ("s2", "s3")
    assert leader.log.term_at(2) == 2
    assert leader.log.term_at(3) == 4


def test_burst_of_proposals_is_group_committed() -> None:
    """Proposals in one batch share one leader fsync; nothing counts until durable."""
    c = ManualCluster(3)
    leader = c.elect("s1")
    store = c.storages["s1"]
    writes = store.writes
    for i in range(10):
        c.propose("s1", f"v{i}")
    assert store.writes == writes  # nothing written yet...
    assert len(store.load().entries) == 1  # ...only the no-op is on disk
    leader.flush()
    assert store.writes == writes + 1  # one append for all ten
    assert len(store.load().entries) == 11
    c.deliver()
    assert leader.commit_index == 11


def test_leader_does_not_count_its_unpersisted_entries() -> None:
    """Leaders may replicate before their own fsync, but only count themselves after it."""
    c = ManualCluster(5)
    leader = c.elect("s1")
    idx = c.propose("s1", "w")
    assert idx is not None
    leader._send_append("s2", force=True)
    leader._send_append("s3", force=True)
    for dest, msg in leader.drain_outbox():
        c[dest].step("s1", msg, 0.0)
        for _, reply in c[dest].drain_outbox():
            leader.step(dest, reply, 0.0)
    # s2 and s3 store it durably, but the leader has not fsync'd: 2 of 5.
    assert leader.match_index["s2"] == leader.match_index["s3"] == idx
    assert leader.commit_index < idx
    leader.flush()
    assert leader.commit_index == idx
