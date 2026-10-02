"""In-memory Raft log with a compacted prefix.

Indices are 1-based as in the Raft paper. Entries up to and including
``snapshot_index`` have been compacted into a snapshot; only their boundary
(``snapshot_index``, ``snapshot_term``) is remembered so that consistency
checks against the entry immediately preceding the live log still work.
"""

from __future__ import annotations

from collections.abc import Iterable

from raftkv.messages import Entry


class RaftLog:
    __slots__ = ("_entries", "snapshot_index", "snapshot_term")

    def __init__(
        self,
        snapshot_index: int = 0,
        snapshot_term: int = 0,
        entries: Iterable[Entry] = (),
    ) -> None:
        self.snapshot_index = snapshot_index
        self.snapshot_term = snapshot_term
        self._entries: list[Entry] = []
        self.append(entries)

    def __len__(self) -> int:
        """Number of live (non-compacted) entries."""
        return len(self._entries)

    @property
    def first_index(self) -> int:
        """Index of the first live entry (may exceed ``last_index`` if empty)."""
        return self.snapshot_index + 1

    @property
    def last_index(self) -> int:
        return self.snapshot_index + len(self._entries)

    @property
    def last_term(self) -> int:
        return self._entries[-1].term if self._entries else self.snapshot_term

    def term_at(self, index: int) -> int | None:
        """Term of the entry at ``index``; ``None`` if compacted or absent."""
        if index == self.snapshot_index:
            return self.snapshot_term
        if index < self.snapshot_index or index > self.last_index:
            return None
        return self._entries[index - self.snapshot_index - 1].term

    def entry(self, index: int) -> Entry:
        if not self.snapshot_index < index <= self.last_index:
            raise IndexError(
                f"entry {index} not in live log ({self.first_index}..{self.last_index})"
            )
        return self._entries[index - self.snapshot_index - 1]

    def slice(self, start: int, end: int) -> list[Entry]:
        """Entries with ``start <= index < end`` (clamped to the live log)."""
        lo = max(start, self.first_index) - self.snapshot_index - 1
        hi = min(end, self.last_index + 1) - self.snapshot_index - 1
        return self._entries[lo:hi] if hi > lo else []

    def append(self, entries: Iterable[Entry]) -> None:
        for e in entries:
            if e.index != self.last_index + 1:
                raise ValueError(
                    f"non-contiguous append: got {e.index}, expected {self.last_index + 1}"
                )
            if e.term < self.last_term:
                raise ValueError(f"term regression at index {e.index}: {e.term} < {self.last_term}")
            self._entries.append(e)

    def truncate_from(self, index: int) -> None:
        """Delete every entry with ``index >= index``."""
        if index <= self.snapshot_index:
            raise ValueError("cannot truncate into the snapshot")
        del self._entries[index - self.snapshot_index - 1 :]

    def compact(self, index: int, term: int) -> None:
        """Discard entries up to and including ``index`` (now in a snapshot)."""
        if index <= self.snapshot_index:
            return
        if index > self.last_index:
            raise ValueError("cannot compact beyond the end of the log")
        if self.term_at(index) != term:
            raise ValueError("snapshot term does not match log")
        del self._entries[: index - self.snapshot_index]
        self.snapshot_index = index
        self.snapshot_term = term

    def first_index_of_term_run(self, index: int) -> int:
        """First index of the contiguous run of ``term_at(index)`` ending at ``index``.

        Never returns an index inside the snapshot; used for the follower's
        conflict hint during fast log backtracking.
        """
        term = self.term_at(index)
        i = index
        while i - 1 > self.snapshot_index and self.term_at(i - 1) == term:
            i -= 1
        return i

    def last_index_of_term(self, term: int) -> int | None:
        """Highest live index whose entry has ``term``, or ``None``."""
        for e in reversed(self._entries):
            if e.term == term:
                return e.index
            if e.term < term:
                break
        return None
