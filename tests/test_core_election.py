# mypy: disable-error-code="comparison-overlap,unreachable"
# (role changes inside step() are invisible to mypy's attribute narrowing)
"""Leader election (Raft section 5.2, 5.4.1), driven message-by-message."""

from __future__ import annotations

import pytest
from harness import ManualCluster, storage_with

from raftkv.core import RaftConfig, RaftCore, Role
from raftkv.messages import AppendEntries, RequestVote, RequestVoteReply
from raftkv.statemachine import KVStateMachine
from raftkv.storage import MemoryStorage


def test_single_node_elects_itself_and_commits() -> None:
    c = ManualCluster(1)
    c.advance(0.31)
    assert c["s1"].role is Role.LEADER
    idx = c.propose("s1", "v")
    assert idx == 2  # 1 is the leader's no-op
    assert c["s1"].commit_index == 2


def test_three_nodes_elect_exactly_one_leader() -> None:
    c = ManualCluster(3)
    for _ in range(40):
        c.advance(0.01)
        c.deliver()
    leaders = c.leaders()
    assert len(leaders) == 1
    leader = leaders[0]
    assert all(c[i].leader_id == leader.id for i in c.ids)
    assert all(c[i].current_term == leader.current_term for i in c.ids)


def test_election_timeouts_are_randomized() -> None:
    deadlines = {ManualCluster(1, seed=s)["s1"].next_deadline() for s in range(20)}
    assert len(deadlines) > 15
    assert all(0.150 <= d <= 0.300 for d in deadlines)


def test_candidate_increments_term_and_votes_for_itself_durably() -> None:
    c = ManualCluster(3)
    c["s1"].campaign()
    assert c["s1"].role is Role.CANDIDATE
    assert c["s1"].current_term == 1
    loaded = c.storages["s1"].load()
    assert (loaded.term, loaded.voted_for) == (1, "s1")
    out = c["s1"].drain_outbox()
    assert sorted(dest for dest, _ in out) == ["s2", "s3"]
    assert all(isinstance(m, RequestVote) and m.term == 1 for _, m in out)


def test_vote_granted_at_most_once_per_term() -> None:
    c = ManualCluster(3)
    s3 = c["s3"]
    s3.step("s1", RequestVote(1, "s1", 0, 0), 0.0)
    s3.step("s2", RequestVote(1, "s2", 0, 0), 0.0)
    replies = [m for _, m in s3.drain_outbox()]
    assert [r.granted for r in replies if isinstance(r, RequestVoteReply)] == [True, False]
    # The same candidate asking again (duplicate message) is granted again.
    s3.step("s1", RequestVote(1, "s1", 0, 0), 0.0)
    ((_, again),) = s3.drain_outbox()
    assert isinstance(again, RequestVoteReply)
    assert again.granted


def test_vote_survives_restart() -> None:
    """votedFor is persisted before the reply, so a restart cannot double-vote."""
    c = ManualCluster(3)
    c["s3"].step("s1", RequestVote(1, "s1", 0, 0), 0.0)
    c.restart("s3")
    c["s3"].step("s2", RequestVote(1, "s2", 0, 0), 0.0)
    ((_, reply),) = c["s3"].drain_outbox()
    assert isinstance(reply, RequestVoteReply)
    assert not reply.granted


def test_vote_denied_to_candidate_with_stale_log() -> None:
    # s3 has a log entry from term 2; candidate's last entry is term 1.
    storages = {"s3": storage_with(2, None, [1, 2])}
    c = ManualCluster(3, storages=storages)
    c["s3"].step("s1", RequestVote(3, "s1", 5, 1), 0.0)  # longer, but older last term
    ((_, reply),) = c["s3"].drain_outbox()
    assert isinstance(reply, RequestVoteReply)
    assert not reply.granted
    assert c["s3"].current_term == 3  # still adopts the higher term
    # Same last term but shorter log: also denied.
    c["s3"].step("s2", RequestVote(4, "s2", 1, 2), 0.0)
    ((_, reply),) = c["s3"].drain_outbox()
    assert isinstance(reply, RequestVoteReply)
    assert not reply.granted


def test_vote_granted_when_logs_equal() -> None:
    storages = {i: storage_with(2, None, [1, 2]) for i in ("s1", "s2", "s3")}
    c = ManualCluster(3, storages=storages)
    leader = c.elect("s2")
    assert leader.role is Role.LEADER
    assert leader.current_term == 3


def test_stale_term_request_vote_rejected() -> None:
    storages = {"s2": storage_with(5, None, [])}
    c = ManualCluster(3, storages=storages)
    c["s2"].step("s1", RequestVote(4, "s1", 0, 0), 0.0)
    ((_, reply),) = c["s2"].drain_outbox()
    assert isinstance(reply, RequestVoteReply)
    assert (reply.granted, reply.term) == (False, 5)


def test_leader_steps_down_on_higher_term() -> None:
    c = ManualCluster(3)
    leader = c.elect("s1")
    assert leader.role is Role.LEADER
    leader.step("s3", RequestVote(leader.current_term + 1, "s3", 0, 0), 0.0)
    assert leader.role is Role.FOLLOWER
    assert leader.current_term == 2


def test_candidate_steps_down_on_append_from_current_term_leader() -> None:
    c = ManualCluster(3)
    c["s2"].campaign()  # term 1, candidate
    c["s2"].drain_outbox()
    c["s2"].step("s1", AppendEntries(1, "s1", 0, 0, [], 0), 0.0)
    assert c["s2"].role is Role.FOLLOWER
    assert c["s2"].leader_id == "s1"


def test_split_vote_resolves_in_later_term() -> None:
    c = ManualCluster(4)
    # s1 and s2 campaign simultaneously; each gets one other vote -> no majority of 4.
    c["s1"].campaign()
    c["s2"].campaign()
    c.deliver(lambda e: not (e[0] == "s1" and e[1] == "s4") and not (e[0] == "s2" and e[1] == "s3"))
    assert c.leaders() == []
    for _ in range(100):
        c.advance(0.01)
        c.deliver()
        if c.leaders():
            break
    assert len(c.leaders()) == 1
    assert c.leaders()[0].current_term >= 2


def test_minority_partition_cannot_elect() -> None:
    c = ManualCluster(5)
    for _ in range(100):
        c.advance(0.01)
        c.deliver(c.only_between("s1", "s2"))
    assert c.leaders() == []
    assert c["s1"].current_term > 1  # it kept trying


def test_follower_with_heartbeats_never_times_out() -> None:
    c = ManualCluster(3)
    leader = c.elect("s1")
    term = leader.current_term
    for _ in range(200):
        c.advance(0.01)
        c.deliver()
    assert leader.role is Role.LEADER
    assert all(c[i].current_term == term for i in c.ids)


def test_member_validation() -> None:
    with pytest.raises(ValueError, match="not in the member list"):
        RaftCore("x", ["a", "b"], MemoryStorage(), KVStateMachine())
    with pytest.raises(ValueError, match="heartbeat"):
        RaftConfig(heartbeat_interval=0.5)
