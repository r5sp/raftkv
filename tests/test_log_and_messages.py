from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from raftkv.log import RaftLog
from raftkv.messages import (
    AppendEntries,
    AppendEntriesReply,
    ClientReply,
    ClientRequest,
    CodecError,
    Entry,
    InstallSnapshot,
    InstallSnapshotReply,
    RequestVote,
    RequestVoteReply,
    decode,
    encode,
)


def make_log(terms: list[int], snapshot_index: int = 0, snapshot_term: int = 0) -> RaftLog:
    entries = [Entry(t, snapshot_index + i, None) for i, t in enumerate(terms, 1)]
    return RaftLog(snapshot_index, snapshot_term, entries)


class TestRaftLog:
    def test_empty_log(self) -> None:
        log = RaftLog()
        assert log.last_index == 0
        assert log.last_term == 0
        assert log.term_at(0) == 0
        assert log.term_at(1) is None

    def test_term_at_and_slice(self) -> None:
        log = make_log([1, 1, 2, 3])
        assert [log.term_at(i) for i in range(5)] == [0, 1, 1, 2, 3]
        assert [e.index for e in log.slice(2, 4)] == [2, 3]
        assert [e.index for e in log.slice(0, 100)] == [1, 2, 3, 4]
        assert log.slice(5, 9) == []

    def test_rejects_gaps_and_term_regression(self) -> None:
        log = make_log([1, 2])
        with pytest.raises(ValueError, match="non-contiguous"):
            log.append([Entry(2, 4)])
        with pytest.raises(ValueError, match="term regression"):
            log.append([Entry(1, 3)])

    def test_truncate(self) -> None:
        log = make_log([1, 1, 2, 2])
        log.truncate_from(3)
        assert log.last_index == 2
        assert log.last_term == 1

    def test_compaction_keeps_boundary_term(self) -> None:
        log = make_log([1, 1, 2, 2, 3])
        log.compact(3, 2)
        assert log.snapshot_index == 3
        assert log.first_index == 4
        assert log.term_at(3) == 2  # boundary still answerable for consistency checks
        assert log.term_at(2) is None
        assert log.last_index == 5
        with pytest.raises(IndexError):
            log.entry(3)
        with pytest.raises(ValueError, match="snapshot"):
            log.truncate_from(3)

    def test_compaction_requires_matching_term(self) -> None:
        log = make_log([1, 2])
        with pytest.raises(ValueError, match="term"):
            log.compact(2, 1)

    def test_term_run_helpers(self) -> None:
        log = make_log([1, 1, 4, 4, 4, 5])
        assert log.first_index_of_term_run(5) == 3
        assert log.first_index_of_term_run(2) == 1
        assert log.last_index_of_term(4) == 5
        assert log.last_index_of_term(2) is None
        log.compact(3, 4)
        # Never points into the snapshot.
        assert log.first_index_of_term_run(5) == 4

    @given(st.lists(st.integers(1, 5), min_size=1, max_size=40).map(sorted), st.data())
    def test_compact_then_query_matches_uncompacted(
        self, terms: list[int], data: st.DataObject
    ) -> None:
        full = make_log(terms)
        cut = data.draw(st.integers(0, len(terms)))
        compacted = make_log(terms)
        if cut:
            compacted.compact(cut, terms[cut - 1])
        for i in range(cut, len(terms) + 2):
            assert compacted.term_at(i) == full.term_at(i)
        assert compacted.last_index == full.last_index
        assert compacted.last_term == full.last_term


MESSAGES = [
    RequestVote(3, "n1", 10, 2),
    RequestVoteReply(3, "n2", True),
    AppendEntries(4, "n1", 7, 3, [Entry(4, 8, {"op": {"kind": "put"}}), Entry(4, 9, None)], 6, 12),
    AppendEntriesReply(4, "n3", False, 0, 5, 2, 12),
    AppendEntriesReply(4, "n3", True, 9, 0, None, 13),
    InstallSnapshot(5, "n1", 100, 4, b"\x00\x01binary\xff", 3),
    InstallSnapshotReply(5, "n2", 100, 3),
    ClientRequest(1, "client-a", 7, {"kind": "cas", "key": "k", "expected": None, "value": "v"}),
    ClientReply(1, False, None, "not_leader", "n2"),
    ClientReply(2, True, {"value": "ünïcödé"}),
]


@pytest.mark.parametrize("msg", MESSAGES, ids=lambda m: type(m).__name__)
def test_codec_round_trip(msg: object) -> None:
    src, decoded = decode(encode("n9", msg))  # type: ignore[arg-type]
    assert src == "n9"
    assert decoded == msg


@pytest.mark.parametrize(
    "payload",
    [
        b"not json",
        b'{"src": "x"}',
        b'{"src":"x","msg":{"type":"Nope"}}',
        b'{"src":"x","msg":{"type":"RequestVote"}}',
    ],
)
def test_codec_rejects_garbage(payload: bytes) -> None:
    with pytest.raises(CodecError):
        decode(payload)
