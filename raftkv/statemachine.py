"""The replicated key-value state machine with exactly-once client sessions.

Commands are JSON objects ``{"client_id": str, "seq": int, "op": {...}}``.
Each client numbers its writes with a strictly increasing ``seq`` and has at
most one outstanding write. The state machine remembers, per client, the last
applied ``seq`` and its result; a re-delivered or retried command with the same
``seq`` returns the cached result instead of executing twice. This is what
turns at-least-once client retries into exactly-once semantics (Raft
dissertation, section 6.3).

Operation results:

=======  =======================================  ==================================
op       request                                  result
=======  =======================================  ==================================
get      ``{"kind": "get", "key": k}``            ``{"value": v | None}``
put      ``{"kind": "put", "key": k, "value": v}``  ``{"prev": old | None}``
delete   ``{"kind": "delete", "key": k}``         ``{"deleted": bool}``
cas      ``{"kind": "cas", "key": k,``            ``{"swapped": bool, "value": cur}``
         ``"expected": e | None, "value": v}``
=======  =======================================  ==================================

``expected = None`` in a CAS means "only if the key is absent".
"""

from __future__ import annotations

import json
from typing import Any

from raftkv.messages import Command

Result = dict[str, Any]
WRITE_OPS = frozenset({"put", "delete", "cas"})


class KVStateMachine:
    def __init__(self) -> None:
        self.data: dict[str, str] = {}
        # client_id -> (last applied seq, result of that command)
        self.sessions: dict[str, tuple[int, Result]] = {}
        self.last_applied = 0

    # -- StateMachine protocol -------------------------------------------

    def apply(self, index: int, command: Command) -> Result:
        self.last_applied = index
        client_id = command.get("client_id")
        seq = int(command.get("seq", 0))
        if client_id is not None:
            session = self.sessions.get(client_id)
            if session is not None and seq <= session[0]:
                # Duplicate: either a retry of the last command (return the
                # cached result) or a stale packet from before it.
                return session[1] if seq == session[0] else {"error": "stale_request"}
        result = self.execute(command["op"])
        if client_id is not None:
            self.sessions[client_id] = (seq, result)
        return result

    def snapshot(self) -> bytes:
        return json.dumps(
            {
                "data": self.data,
                "sessions": {cid: [seq, res] for cid, (seq, res) in self.sessions.items()},
                "last_applied": self.last_applied,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()

    def restore(self, data: bytes) -> None:
        raw = json.loads(data)
        self.data = dict(raw["data"])
        self.sessions = {cid: (int(s), r) for cid, (s, r) in raw["sessions"].items()}
        self.last_applied = int(raw["last_applied"])

    # -- helpers -----------------------------------------------------------

    def cached_result(self, client_id: str, seq: int) -> Result | None:
        """Result of an already-applied write, if this exact ``seq`` was the last one."""
        session = self.sessions.get(client_id)
        if session is not None and session[0] == seq:
            return session[1]
        return None

    def get(self, key: str) -> Result:
        return {"value": self.data.get(key)}

    def execute(self, op: dict[str, Any]) -> Result:
        kind = op.get("kind")
        key = str(op.get("key"))
        if kind == "get":
            return self.get(key)
        if kind == "put":
            prev = self.data.get(key)
            self.data[key] = str(op["value"])
            return {"prev": prev}
        if kind == "delete":
            return {"deleted": self.data.pop(key, None) is not None}
        if kind == "cas":
            current = self.data.get(key)
            if current == op.get("expected"):
                self.data[key] = str(op["value"])
                return {"swapped": True, "value": self.data[key]}
            return {"swapped": False, "value": current}
        return {"error": f"unknown op {kind!r}"}
