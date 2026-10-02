"""The linearizability checker, validated against hand-written cases and a brute-force oracle."""

from __future__ import annotations

import math
import random
import time
from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st

from raftkv.linearizability import (
    Operation,
    RegisterModel,
    brute_force_check,
    check_history,
    check_operations,
)

INF = math.inf


def op(
    kind: str,
    call: float,
    ret: float,
    out: dict[str, Any] | None,
    key: str = "k",
    client: str = "c",
    **args: Any,
) -> Operation:
    return Operation(client, kind, key, args, call, ret, out)


def put(v: str, call: float, ret: float, prev: str | None = None, **kw: Any) -> Operation:
    return op("put", call, ret, None if ret == INF else {"prev": prev}, value=v, **kw)


def get(v: str | None, call: float, ret: float, **kw: Any) -> Operation:
    return op("get", call, ret, {"value": v}, **kw)


def ok(ops: list[Operation]) -> bool:
    return check_operations(ops, RegisterModel())[0]


class TestHandWritten:
    def test_empty_and_sequential(self) -> None:
        assert ok([])
        assert ok([put("a", 0, 1), get("a", 2, 3), put("b", 4, 5, prev="a"), get("b", 6, 7)])

    def test_stale_read_is_detected(self) -> None:
        # The write completed before the read began, yet the read missed it.
        assert not ok([put("a", 0, 1), get(None, 2, 3)])

    def test_concurrent_read_may_see_either(self) -> None:
        assert ok([put("a", 0, 10), get(None, 1, 2)])
        assert ok([put("a", 0, 10), get("a", 1, 2)])

    def test_new_old_inversion_is_detected(self) -> None:
        # Write is concurrent with both reads, but once one read observed the
        # new value a later read may not observe the old one.
        assert not ok([put("a", 0, 10), get("a", 1, 2), get(None, 3, 4)])
        assert ok([put("a", 0, 10), get(None, 1, 2), get("a", 3, 4)])

    def test_put_prev_value_is_checked(self) -> None:
        assert not ok([put("a", 0, 1), put("b", 2, 3, prev=None)])

    def test_cas(self) -> None:
        cas_ok = op("cas", 2, 3, {"swapped": True, "value": "b"}, expected="a", value="b")
        cas_bad = op("cas", 2, 3, {"swapped": True, "value": "b"}, expected="z", value="b")
        assert ok([put("a", 0, 1), cas_ok, get("b", 4, 5)])
        assert not ok([put("a", 0, 1), cas_bad])

    def test_delete(self) -> None:
        d = op("delete", 2, 3, {"deleted": True})
        assert ok([put("a", 0, 1), d, get(None, 4, 5)])
        assert not ok([op("delete", 0, 1, {"deleted": True})])

    def test_pending_write_may_take_effect_late_or_never(self) -> None:
        pending = put("a", 0, INF)
        assert ok([pending, get(None, 1, 2)])
        assert ok([put("a", 0, INF), get(None, 1, 2), get("a", 5, 6)])

    def test_pending_write_cannot_be_unseen(self) -> None:
        assert not ok([put("a", 0, INF), get("a", 1, 2), get(None, 3, 4)])

    def test_value_never_written_is_detected(self) -> None:
        assert not ok([put("a", 0, 1), get("ghost", 2, 3)])

    def test_keys_are_independent(self) -> None:
        history = [
            put("a", 0, 1, key="x"),
            put("b", 0, 1, key="y"),
            get("a", 2, 3, key="x"),
            get("b", 2, 3, key="y"),
        ]
        assert check_history(history).ok
        bad = [*history, get("a", 4, 5, key="y")]
        result = check_history(bad)
        assert not result.ok
        assert result.failed_key == "y"


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------

VALUES = [None, "a", "b"]


@st.composite
def linearizable_histories(draw: st.DrawFn) -> list[Operation]:
    """Execute random ops sequentially at distinct linearization points, then
    widen each into a random interval that contains its point."""
    n = draw(st.integers(1, 40))
    rng = random.Random(draw(st.integers(0, 2**32)))
    points = sorted(rng.sample(range(1, 10 * n + 10), n))
    state: str | None = None
    ops: list[Operation] = []
    for i, p in enumerate(points):
        kind = rng.choice(["get", "put", "delete", "cas"])
        v = f"v{i}"
        if kind == "get":
            o = Operation("c", kind, "k", {}, 0, 0, {"value": state})
        elif kind == "put":
            o = Operation("c", kind, "k", {"value": v}, 0, 0, {"prev": state})
            state = v
        elif kind == "delete":
            o = Operation("c", kind, "k", {}, 0, 0, {"deleted": state is not None})
            state = None
        else:
            expected = rng.choice([state, None, "nope"])
            swapped = expected == state
            state = v if swapped else state
            o = Operation(
                "c",
                kind,
                "k",
                {"expected": expected, "value": v},
                0,
                0,
                {"swapped": swapped, "value": state},
            )
        o.call = p - rng.uniform(0, 15)
        if kind != "get" and rng.random() < 0.1:
            o.ret, o.output = INF, None  # outcome unknown to the client
        else:
            o.ret = p + rng.uniform(0, 15)
        ops.append(o)
    rng.shuffle(ops)
    return ops


@settings(max_examples=300, deadline=None)
@given(linearizable_histories())
def test_generated_linearizable_histories_pass(ops: list[Operation]) -> None:
    assert ok(ops)


@st.composite
def arbitrary_histories(draw: st.DrawFn) -> list[Operation]:
    """Small histories with arbitrary (often impossible) outputs."""
    n = draw(st.integers(1, 6))
    ops = []
    for _ in range(n):
        call = draw(st.integers(0, 10))
        dur = draw(st.integers(1, 6))
        pending = draw(st.booleans()) and draw(st.booleans())
        kind = draw(st.sampled_from(["get", "put", "delete", "cas"]))
        args: dict[str, Any] = {}
        out: dict[str, Any] | None
        if kind == "get":
            out, pending = {"value": draw(st.sampled_from(VALUES))}, False
        elif kind == "put":
            args = {"value": draw(st.sampled_from(VALUES[1:]))}
            out = {"prev": draw(st.sampled_from(VALUES))}
        elif kind == "delete":
            out = {"deleted": draw(st.booleans())}
        else:
            args = {
                "expected": draw(st.sampled_from(VALUES)),
                "value": draw(st.sampled_from(VALUES[1:])),
            }
            out = {"swapped": draw(st.booleans()), "value": draw(st.sampled_from(VALUES))}
        ops.append(
            Operation(
                "c", kind, "k", args, call, INF if pending else call + dur, None if pending else out
            )
        )
    return ops


@settings(max_examples=1500, deadline=None)
@given(arbitrary_histories())
def test_checker_agrees_with_brute_force_oracle(ops: list[Operation]) -> None:
    assert ok(ops) == brute_force_check(ops, RegisterModel())


@settings(max_examples=200, deadline=None)
@given(linearizable_histories(), st.data())
def test_corrupted_read_is_rejected(ops: list[Operation], data: st.DataObject) -> None:
    gets = [o for o in ops if o.kind == "get"]
    if not gets:
        return
    victim = data.draw(st.sampled_from(gets))
    victim.output = {"value": "never-written"}
    assert not ok(ops)


def test_large_concurrent_history_checks_quickly() -> None:
    """2,500 ops from 8 overlapping clients on 4 keys."""
    rng = random.Random(42)
    state: dict[str, str | None] = {}
    ops: list[Operation] = []
    t = 0.0
    for i in range(2500):
        t += rng.uniform(0.5, 1.5)
        key = f"k{rng.randrange(4)}"
        cur = state.get(key)
        if rng.random() < 0.5:
            o = Operation(f"c{i % 8}", "get", key, {}, 0, 0, {"value": cur})
        else:
            o = Operation(f"c{i % 8}", "put", key, {"value": f"v{i}"}, 0, 0, {"prev": cur})
            state[key] = f"v{i}"
        o.call, o.ret = t - rng.uniform(0, 6), t + rng.uniform(0, 6)
        ops.append(o)
    start = time.perf_counter()
    assert check_history(ops).ok
    assert time.perf_counter() - start < 10
