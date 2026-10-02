"""KV state machine semantics and exactly-once sessions."""

from __future__ import annotations

from raftkv.statemachine import KVStateMachine


def cmd(client: str | None, seq: int, kind: str, key: str = "k", **kw: object) -> dict[str, object]:
    return {"client_id": client, "seq": seq, "op": {"kind": kind, "key": key, **kw}}


def test_operations() -> None:
    sm = KVStateMachine()
    assert sm.apply(1, cmd("a", 1, "put", value="1")) == {"prev": None}
    assert sm.apply(2, cmd("a", 2, "put", value="2")) == {"prev": "1"}
    assert sm.apply(3, cmd("a", 3, "cas", expected="1", value="x")) == {
        "swapped": False,
        "value": "2",
    }
    assert sm.apply(4, cmd("a", 4, "cas", expected="2", value="3")) == {
        "swapped": True,
        "value": "3",
    }
    assert sm.apply(5, cmd("a", 5, "delete")) == {"deleted": True}
    assert sm.apply(6, cmd("a", 6, "delete")) == {"deleted": False}
    assert sm.apply(7, cmd("a", 7, "cas", expected=None, value="new")) == {
        "swapped": True,
        "value": "new",
    }
    assert sm.get("k") == {"value": "new"}
    assert sm.last_applied == 7


def test_duplicate_write_applied_once() -> None:
    sm = KVStateMachine()
    first = sm.apply(1, cmd("a", 1, "put", value="v1"))
    sm.apply(2, cmd("b", 1, "put", value="other"))
    # Client a's retry of seq 1 is committed again (e.g. reply was lost).
    again = sm.apply(3, cmd("a", 1, "put", value="v1"))
    assert again == first == {"prev": None}
    assert sm.data == {"k": "other"}  # the retry did not re-execute
    assert sm.cached_result("a", 1) == first


def test_stale_request_rejected() -> None:
    sm = KVStateMachine()
    sm.apply(1, cmd("a", 2, "put", value="new"))
    assert sm.apply(2, cmd("a", 1, "put", value="old")) == {"error": "stale_request"}
    assert sm.data == {"k": "new"}


def test_sessions_are_per_client() -> None:
    sm = KVStateMachine()
    sm.apply(1, cmd("a", 1, "put", value="a1"))
    assert sm.apply(2, cmd("b", 1, "put", value="b1")) == {"prev": "a1"}


def test_unknown_op() -> None:
    assert "error" in KVStateMachine().execute({"kind": "nope"})


def test_snapshot_round_trip() -> None:
    sm = KVStateMachine()
    sm.apply(1, cmd("a", 1, "put", key="x", value="1"))
    sm.apply(2, cmd("b", 4, "cas", key="x", expected="1", value="2"))
    other = KVStateMachine()
    other.restore(sm.snapshot())
    assert other.data == {"x": "2"}
    assert other.sessions == sm.sessions
    assert other.last_applied == 2
    # Dedup still works after restore.
    assert other.apply(3, cmd("b", 4, "cas", key="x", expected="1", value="2")) == {
        "swapped": True,
        "value": "2",
    }
    assert other.data == {"x": "2"}
