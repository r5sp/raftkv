"""Wire messages: Raft RPCs and client requests, plus a JSON codec.

Every message is a plain dataclass. ``encode``/``decode`` turn a ``(src, msg)``
envelope into bytes and back; both the TCP transport and the simulated network
use the same codec, so the simulator exercises serialization too.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from typing import Any

NodeId = str
Command = dict[str, Any]


@dataclass(frozen=True, slots=True)
class Entry:
    """A single log entry. ``command is None`` marks a leader no-op entry."""

    term: int
    index: int
    command: Command | None = None


@dataclass(slots=True)
class RequestVote:
    term: int
    candidate_id: NodeId
    last_log_index: int
    last_log_term: int


@dataclass(slots=True)
class RequestVoteReply:
    term: int
    voter_id: NodeId
    granted: bool


@dataclass(slots=True)
class AppendEntries:
    term: int
    leader_id: NodeId
    prev_log_index: int
    prev_log_term: int
    entries: list[Entry]
    leader_commit: int
    # Monotonic per-leader counter, echoed in the reply; used by ReadIndex to
    # prove that a quorum still recognised this leader *after* a read arrived.
    seq: int = 0


@dataclass(slots=True)
class AppendEntriesReply:
    term: int
    follower_id: NodeId
    success: bool
    # On success: highest index known to match the leader's log.
    match_index: int = 0
    # On failure: fast-backtracking hints (see Raft paper section 5.3).
    conflict_index: int = 0
    conflict_term: int | None = None
    seq: int = 0


@dataclass(slots=True)
class InstallSnapshot:
    term: int
    leader_id: NodeId
    last_included_index: int
    last_included_term: int
    data: bytes
    seq: int = 0


@dataclass(slots=True)
class InstallSnapshotReply:
    term: int
    follower_id: NodeId
    last_included_index: int
    seq: int = 0


@dataclass(slots=True)
class ClientRequest:
    request_id: int
    client_id: str
    seq: int
    op: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ClientReply:
    request_id: int
    ok: bool
    result: Any = None
    error: str | None = None
    leader_hint: NodeId | None = None


Message = (
    RequestVote
    | RequestVoteReply
    | AppendEntries
    | AppendEntriesReply
    | InstallSnapshot
    | InstallSnapshotReply
    | ClientRequest
    | ClientReply
)

RAFT_MESSAGES = (
    RequestVote,
    RequestVoteReply,
    AppendEntries,
    AppendEntriesReply,
    InstallSnapshot,
    InstallSnapshotReply,
)

_TYPES: dict[str, type[Any]] = {
    cls.__name__: cls
    for cls in (
        RequestVote,
        RequestVoteReply,
        AppendEntries,
        AppendEntriesReply,
        InstallSnapshot,
        InstallSnapshotReply,
        ClientRequest,
        ClientReply,
    )
}


class CodecError(ValueError):
    """Raised when bytes on the wire cannot be decoded into a message."""


def entry_to_wire(e: Entry) -> list[Any]:
    return [e.term, e.index, e.command]


def entry_from_wire(raw: Any) -> Entry:
    term, index, command = raw
    return Entry(int(term), int(index), command)


def _to_wire(msg: Message) -> dict[str, Any]:
    out: dict[str, Any] = {"type": type(msg).__name__}
    for name in msg.__slots__:
        value = getattr(msg, name)
        if name == "entries":
            value = [entry_to_wire(e) for e in value]
        elif name == "data":
            value = base64.b64encode(value).decode("ascii")
        out[name] = value
    return out


def _from_wire(raw: dict[str, Any]) -> Message:
    try:
        cls = _TYPES[raw.pop("type")]
    except KeyError as exc:
        raise CodecError(f"unknown message type in {raw!r}") from exc
    if "entries" in raw:
        raw["entries"] = [entry_from_wire(e) for e in raw["entries"]]
    if "data" in raw:
        raw["data"] = base64.b64decode(raw["data"])
    try:
        msg: Message = cls(**raw)
    except TypeError as exc:
        raise CodecError(str(exc)) from exc
    return msg


def encode(src: NodeId, msg: Message) -> bytes:
    """Serialize a message envelope to UTF-8 JSON bytes."""
    return json.dumps({"src": src, "msg": _to_wire(msg)}, separators=(",", ":")).encode()


def decode(payload: bytes) -> tuple[NodeId, Message]:
    """Inverse of :func:`encode`."""
    try:
        raw = json.loads(payload)
        return str(raw["src"]), _from_wire(raw["msg"])
    except (ValueError, KeyError, TypeError) as exc:
        if isinstance(exc, CodecError):
            raise
        raise CodecError(f"malformed message: {exc}") from exc
