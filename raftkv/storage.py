"""Durable Raft state: ``currentTerm``, ``votedFor``, the log, and the snapshot.

The consensus core calls these methods synchronously and only emits outgoing
messages *after* they return, so every RPC response is sent strictly after the
state it depends on is durable ("persist before you respond", Raft Figure 2).

Two implementations:

* :class:`MemoryStorage` keeps state in Python objects. A simulated crash
  discards the node but keeps its ``MemoryStorage``, which models a disk that
  survives the crash.
* :class:`FileStorage` writes to a directory with ``fsync``:

  - ``state.json``   -- term and vote, replaced atomically (write tmp, fsync, rename, fsync dir)
  - ``log.bin``      -- append-only records ``[len:u32][crc32:u32][json]``; a torn tail
                        from a crash mid-write fails its CRC and is truncated on load
  - ``snapshot.bin`` -- ``[meta_len:u32][meta json][data]``, replaced atomically
"""

from __future__ import annotations

import contextlib
import json
import os
import struct
import sys
import zlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Protocol

from raftkv.messages import Entry, NodeId, entry_from_wire, entry_to_wire


@dataclass
class PersistentState:
    term: int = 0
    voted_for: NodeId | None = None
    snapshot_index: int = 0
    snapshot_term: int = 0
    snapshot_data: bytes | None = None
    entries: list[Entry] = field(default_factory=list)


class Storage(Protocol):
    def load(self) -> PersistentState: ...

    def save_hard_state(self, term: int, voted_for: NodeId | None) -> None: ...

    def append(self, entries: Sequence[Entry]) -> None:
        """Durably append ``entries`` after the current last entry."""
        ...

    def truncate_from(self, index: int) -> None:
        """Durably delete every entry with index ``>= index``."""
        ...

    def save_snapshot(self, index: int, term: int, data: bytes, keep: Sequence[Entry]) -> None:
        """Durably replace the snapshot; afterwards the log is exactly ``keep``."""
        ...

    def close(self) -> None: ...


class MemoryStorage:
    """Storage that lives in memory; survives simulated crashes by design."""

    def __init__(self) -> None:
        self._state = PersistentState()
        self.writes = 0

    def load(self) -> PersistentState:
        s = self._state
        return PersistentState(
            s.term, s.voted_for, s.snapshot_index, s.snapshot_term, s.snapshot_data, list(s.entries)
        )

    def save_hard_state(self, term: int, voted_for: NodeId | None) -> None:
        self._state.term = term
        self._state.voted_for = voted_for
        self.writes += 1

    def append(self, entries: Sequence[Entry]) -> None:
        self._state.entries.extend(entries)
        self.writes += 1

    def truncate_from(self, index: int) -> None:
        s = self._state
        del s.entries[max(0, index - s.snapshot_index - 1) :]
        self.writes += 1

    def save_snapshot(self, index: int, term: int, data: bytes, keep: Sequence[Entry]) -> None:
        s = self._state
        s.snapshot_index, s.snapshot_term, s.snapshot_data = index, term, data
        s.entries = list(keep)
        self.writes += 1

    def close(self) -> None:
        pass


_REC_HEADER = struct.Struct(">II")  # payload length, crc32(payload)
_U32 = struct.Struct(">I")


def _fsync(fd: int, full: bool) -> None:
    # On macOS, fsync() only pushes data to the drive; F_FULLFSYNC also flushes
    # the drive's volatile cache. It is much slower, so it is opt-in.
    if full and sys.platform == "darwin":
        import fcntl

        fcntl.fcntl(fd, fcntl.F_FULLFSYNC)  # type: ignore[attr-defined,unused-ignore]
    else:
        os.fsync(fd)


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_write(path: Path, data: bytes, full_fsync: bool) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        _fsync(f.fileno(), full_fsync)
    os.replace(tmp, path)
    _fsync_dir(path.parent)


def _encode_record(e: Entry) -> bytes:
    payload = json.dumps(entry_to_wire(e), separators=(",", ":")).encode()
    return _REC_HEADER.pack(len(payload), zlib.crc32(payload)) + payload


class FileStorage:
    """Crash-safe on-disk storage. See the module docstring for the layout."""

    def __init__(self, directory: str | os.PathLike[str], *, full_fsync: bool = False) -> None:
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.full_fsync = full_fsync
        self._state_path = self.dir / "state.json"
        self._log_path = self.dir / "log.bin"
        self._snap_path = self.dir / "snapshot.bin"
        self._log: BinaryIO | None = None
        self._offsets: list[int] = []  # file offset of each live record
        self._first_index = 1  # log index of the first record in log.bin

    # -- loading -----------------------------------------------------------

    def load(self) -> PersistentState:
        st = PersistentState()
        if self._state_path.exists():
            raw = json.loads(self._state_path.read_bytes())
            st.term, st.voted_for = int(raw["term"]), raw["voted_for"]
        if self._snap_path.exists():
            blob = self._snap_path.read_bytes()
            (meta_len,) = _U32.unpack_from(blob, 0)
            meta = json.loads(blob[4 : 4 + meta_len])
            data = blob[4 + meta_len :]
            if zlib.crc32(data) != meta["crc"]:
                raise OSError(f"corrupt snapshot in {self._snap_path}")
            st.snapshot_index, st.snapshot_term = int(meta["index"]), int(meta["term"])
            st.snapshot_data = data
        raw_entries, offsets, good_end = self._read_log()
        # Drop entries covered by the snapshot. If a crash interrupted the log
        # rewrite after InstallSnapshot, the old log may disagree with the
        # snapshot; in that case everything after the snapshot is discarded.
        if raw_entries and raw_entries[0].index > st.snapshot_index + 1:
            # A gap between snapshot and log cannot be produced by this class;
            # treat it as corruption of the (uncommitted-safe) log suffix.
            raw_entries, offsets, good_end = [], [], 0
        if raw_entries and raw_entries[0].index <= st.snapshot_index:
            boundary = st.snapshot_index - raw_entries[0].index
            consistent = (
                boundary < len(raw_entries) and raw_entries[boundary].term == st.snapshot_term
            )
            keep = raw_entries[boundary + 1 :] if consistent else []
            self._rewrite_log(keep)
            st.entries = keep
        else:
            st.entries = raw_entries
            self._open_log(truncate_at=good_end)
            self._offsets = offsets
            self._first_index = raw_entries[0].index if raw_entries else st.snapshot_index + 1
        if not st.entries:
            self._first_index = st.snapshot_index + 1
        return st

    def _read_log(self) -> tuple[list[Entry], list[int], int]:
        entries: list[Entry] = []
        offsets: list[int] = []
        if not self._log_path.exists():
            return entries, offsets, 0
        blob = self._log_path.read_bytes()
        pos = 0
        while pos + _REC_HEADER.size <= len(blob):
            length, crc = _REC_HEADER.unpack_from(blob, pos)
            start = pos + _REC_HEADER.size
            payload = blob[start : start + length]
            if len(payload) < length or zlib.crc32(payload) != crc:
                break  # torn or corrupt tail: everything after is discarded
            e = entry_from_wire(json.loads(payload))
            if entries and e.index != entries[-1].index + 1:
                break
            entries.append(e)
            offsets.append(pos)
            pos = start + length
        return entries, offsets, pos

    def _open_log(self, truncate_at: int | None = None) -> None:
        if self._log is not None:
            self._log.close()
        self._log = open(self._log_path, "a+b")  # noqa: SIM115 - kept open for appends
        if truncate_at is not None and truncate_at < self._log.seek(0, os.SEEK_END):
            self._log.truncate(truncate_at)
            self._log.flush()
            _fsync(self._log.fileno(), self.full_fsync)

    def _rewrite_log(self, entries: Sequence[Entry]) -> None:
        records = [_encode_record(e) for e in entries]
        _atomic_write(self._log_path, b"".join(records), self.full_fsync)
        self._open_log()
        self._offsets = []
        pos = 0
        for r in records:
            self._offsets.append(pos)
            pos += len(r)
        if entries:
            self._first_index = entries[0].index

    # -- mutations ---------------------------------------------------------

    def save_hard_state(self, term: int, voted_for: NodeId | None) -> None:
        payload = json.dumps({"term": term, "voted_for": voted_for}).encode()
        _atomic_write(self._state_path, payload, self.full_fsync)

    def append(self, entries: Sequence[Entry]) -> None:
        if not entries:
            return
        assert self._log is not None, "load() must be called first"
        if not self._offsets:
            self._first_index = entries[0].index
        pos = self._log.seek(0, os.SEEK_END)
        chunks = []
        for e in entries:
            rec = _encode_record(e)
            self._offsets.append(pos)
            pos += len(rec)
            chunks.append(rec)
        self._log.write(b"".join(chunks))
        self._log.flush()
        _fsync(self._log.fileno(), self.full_fsync)

    def truncate_from(self, index: int) -> None:
        assert self._log is not None, "load() must be called first"
        pos = index - self._first_index
        if pos < 0:
            pos = 0
        if pos >= len(self._offsets):
            return
        self._log.truncate(self._offsets[pos])
        self._log.flush()
        _fsync(self._log.fileno(), self.full_fsync)
        del self._offsets[pos:]

    def save_snapshot(self, index: int, term: int, data: bytes, keep: Sequence[Entry]) -> None:
        meta = json.dumps({"index": index, "term": term, "crc": zlib.crc32(data)}).encode()
        _atomic_write(self._snap_path, _U32.pack(len(meta)) + meta + data, self.full_fsync)
        self._first_index = index + 1
        self._rewrite_log(keep)

    def close(self) -> None:
        if self._log is not None:
            with contextlib.suppress(OSError):
                self._log.close()
            self._log = None
