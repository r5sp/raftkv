"""Durability: FileStorage must survive restarts, torn writes and interrupted compaction."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st

from raftkv.messages import Entry
from raftkv.storage import FileStorage, MemoryStorage, PersistentState


def entries(start: int, terms: list[int]) -> list[Entry]:
    return [
        Entry(t, start + i, {"op": {"kind": "put", "key": "k", "value": str(start + i)}})
        for i, t in enumerate(terms)
    ]


def test_hard_state_and_log_round_trip(tmp_path: Path) -> None:
    s = FileStorage(tmp_path)
    s.load()
    s.save_hard_state(7, "n2")
    s.append(entries(1, [1, 1, 2]))
    s.append(entries(4, [3]))
    s.close()

    st2 = FileStorage(tmp_path).load()
    assert (st2.term, st2.voted_for) == (7, "n2")
    assert [e.index for e in st2.entries] == [1, 2, 3, 4]
    assert [e.term for e in st2.entries] == [1, 1, 2, 3]
    assert st2.entries[0].command == {"op": {"kind": "put", "key": "k", "value": "1"}}


def test_truncate_then_append(tmp_path: Path) -> None:
    s = FileStorage(tmp_path)
    s.load()
    s.append(entries(1, [1, 1, 1, 1]))
    s.truncate_from(3)
    s.append(entries(3, [2]))
    s.close()
    loaded = FileStorage(tmp_path).load()
    assert [(e.index, e.term) for e in loaded.entries] == [(1, 1), (2, 1), (3, 2)]


def test_torn_tail_write_is_discarded(tmp_path: Path) -> None:
    s = FileStorage(tmp_path)
    s.load()
    s.append(entries(1, [1, 1, 1]))
    s.close()
    log = tmp_path / "log.bin"
    data = log.read_bytes()
    log.write_bytes(data[:-5])  # crash in the middle of writing record 3
    s2 = FileStorage(tmp_path)
    loaded = s2.load()
    assert [e.index for e in loaded.entries] == [1, 2]
    # ...and the file is usable for further appends.
    s2.append(entries(3, [2]))
    s2.close()
    assert [e.term for e in FileStorage(tmp_path).load().entries] == [1, 1, 2]


def test_corrupt_record_truncates_suffix(tmp_path: Path) -> None:
    s = FileStorage(tmp_path)
    s.load()
    s.append(entries(1, [1, 1, 1]))
    s.close()
    log = tmp_path / "log.bin"
    data = bytearray(log.read_bytes())
    data[-3] ^= 0xFF  # bit-rot inside record 3's payload -> CRC mismatch
    log.write_bytes(bytes(data))
    assert [e.index for e in FileStorage(tmp_path).load().entries] == [1, 2]


def test_snapshot_round_trip(tmp_path: Path) -> None:
    s = FileStorage(tmp_path)
    s.load()
    s.append(entries(1, [1, 1, 2, 2, 2]))
    keep = entries(4, [2, 2])
    s.save_snapshot(3, 2, b"snapshot-bytes", keep)
    s.append(entries(6, [3]))
    s.close()
    loaded = FileStorage(tmp_path).load()
    assert (loaded.snapshot_index, loaded.snapshot_term, loaded.snapshot_data) == (
        3,
        2,
        b"snapshot-bytes",
    )
    assert [(e.index, e.term) for e in loaded.entries] == [(4, 2), (5, 2), (6, 3)]


def test_crash_between_snapshot_and_log_rewrite(tmp_path: Path) -> None:
    """Snapshot written, but the log still holds compacted entries: they are skipped."""
    s = FileStorage(tmp_path)
    s.load()
    s.append(entries(1, [1, 1, 2, 2]))
    s.close()
    old_log = (tmp_path / "log.bin").read_bytes()
    s = FileStorage(tmp_path)
    s.load()
    s.save_snapshot(2, 1, b"S", entries(3, [2, 2]))
    s.close()
    (tmp_path / "log.bin").write_bytes(old_log)  # simulate rename not yet durable
    loaded = FileStorage(tmp_path).load()
    assert loaded.snapshot_index == 2
    assert [e.index for e in loaded.entries] == [3, 4]


def test_conflicting_log_after_install_snapshot_is_dropped(tmp_path: Path) -> None:
    """Old log disagrees with the installed snapshot at its boundary: discard it."""
    s = FileStorage(tmp_path)
    s.load()
    s.append(entries(1, [1, 1, 1]))  # entry 2 has term 1
    s.close()
    old_log = (tmp_path / "log.bin").read_bytes()
    s = FileStorage(tmp_path)
    s.load()
    s.save_snapshot(2, 5, b"S", [])  # leader's entry 2 has term 5
    s.close()
    (tmp_path / "log.bin").write_bytes(old_log)
    loaded = FileStorage(tmp_path).load()
    assert loaded.snapshot_index == 2
    assert loaded.entries == []


# A random sequence of storage operations must leave FileStorage (after a
# reload from disk) in exactly the same state as the trivially-correct
# MemoryStorage.
op_strategy = st.lists(
    st.one_of(
        st.tuples(st.just("hard"), st.integers(0, 50), st.sampled_from([None, "n1", "n2"])),
        st.tuples(st.just("append"), st.integers(1, 4)),
        st.tuples(st.just("truncate"), st.integers(0, 6)),
        st.tuples(st.just("snapshot"), st.integers(0, 6)),
        st.just(("reopen",)),
    ),
    max_size=30,
)


def _logical(s: PersistentState) -> tuple[object, ...]:
    return (s.term, s.voted_for, s.snapshot_index, s.snapshot_term, s.snapshot_data, s.entries)


@settings(max_examples=60, deadline=None)
@given(ops=op_strategy)
def test_file_storage_matches_memory_model(ops: list[tuple[Any, ...]]) -> None:
    with tempfile.TemporaryDirectory() as d:
        mem = MemoryStorage()
        disk = FileStorage(d)
        disk.load()
        model = mem.load()
        term = 1
        for op in ops:
            kind = op[0]
            last = model.snapshot_index + len(model.entries)
            if kind == "hard":
                mem.save_hard_state(int(op[1]), op[2])
                disk.save_hard_state(int(op[1]), op[2])
            elif kind == "append":
                term += 1
                new = entries(last + 1, [term] * int(op[1]))
                mem.append(new)
                disk.append(new)
            elif kind == "truncate":
                idx = model.snapshot_index + 1 + int(op[1])
                if idx <= last:
                    mem.truncate_from(idx)
                    disk.truncate_from(idx)
            elif kind == "snapshot":
                idx = model.snapshot_index + int(op[1])
                if model.snapshot_index < idx <= last:
                    t = model.entries[idx - model.snapshot_index - 1].term
                    keep = [e for e in model.entries if e.index > idx]
                    mem.save_snapshot(idx, t, f"snap{idx}".encode(), keep)
                    disk.save_snapshot(idx, t, f"snap{idx}".encode(), keep)
            elif kind == "reopen":
                disk.close()
                disk = FileStorage(d)
                disk.load()
            model = mem.load()
        disk.close()
        assert _logical(FileStorage(d).load()) == _logical(mem.load())
