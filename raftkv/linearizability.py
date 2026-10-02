"""Linearizability checker for key-value histories.

Implements the Wing & Gong search with the state-memoization improvement of
Lowe ("Testing for linearizability", 2017), as popularised by Knossos and
Porcupine:

1. The history becomes a doubly linked list of call/return events in time
   order. An operation must take effect somewhere between its call and its
   return.
2. Depth-first search: repeatedly pick an operation whose *call* appears
   before the first remaining *return* (i.e. it is minimal in real-time
   order), apply it to the sequential model, and if the model accepts the
   observed output, "lift" it out of the list and continue. If we hit a
   return event whose operation has not been linearized, backtrack.
3. Memoize ``(set of linearized ops, model state)`` pairs already explored;
   reaching one again cannot lead anywhere new. This makes the search
   practical on histories of thousands of operations.

Because linearizability is *local* (Herlihy & Wing), a KV history is
linearizable iff each per-key sub-history is, so keys are checked
independently -- an exponential saving.

Operations that never returned (client timed out, outcome unknown) are given
a return time of +infinity and accept any output: they may take effect at any
point after their call, or effectively never (by being linearized last).
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Hashable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class Operation:
    client: str
    kind: str  # get | put | delete | cas
    key: str
    args: dict[str, Any]
    call: float
    ret: float = math.inf  # inf => never returned (indeterminate)
    output: dict[str, Any] | None = None  # None => unknown
    id: int = -1

    @property
    def completed(self) -> bool:
        return self.output is not None


class Model(Protocol):
    def init(self) -> Hashable: ...

    def step(self, state: Hashable, op: Operation) -> tuple[bool, Hashable]:
        """Apply ``op``; return whether its observed output is legal, and the new state."""
        ...


class RegisterModel:
    """Sequential spec of a single KV key: state is the value or ``None``."""

    def init(self) -> Hashable:
        return None

    def step(self, state: Hashable, op: Operation) -> tuple[bool, Hashable]:
        out = op.output
        if op.kind == "get":
            return (out is None or out.get("value") == state), state
        if op.kind == "put":
            return (out is None or out.get("prev") == state), op.args["value"]
        if op.kind == "delete":
            return (out is None or out.get("deleted") == (state is not None)), None
        if op.kind == "cas":
            swapped = state == op.args["expected"]
            new = op.args["value"] if swapped else state
            if out is None:
                return True, new
            return (out.get("swapped") == swapped and out.get("value") == new), new
        raise ValueError(f"unknown op kind {op.kind!r}")


@dataclass
class _Event:
    op: Operation
    is_call: bool
    match: _Event | None = None
    prev: _Event | None = None
    next: _Event | None = None


@dataclass
class CheckResult:
    ok: bool
    ops_checked: int
    failed_key: str | None = None
    # Longest linearizable prefix found for the failing key (for debugging).
    partial: list[Operation] = field(default_factory=list)
    unresolved: list[Operation] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.ok


def _build_list(ops: Sequence[Operation]) -> _Event:
    events: list[tuple[float, int, int, _Event]] = []
    for i, op in enumerate(ops):
        call = _Event(op, True)
        ret = _Event(op, False)
        call.match = ret
        # Sort key: time, then calls before returns at equal timestamps (which
        # treats touching intervals as concurrent -- the permissive choice).
        events.append((op.call, 0, i, call))
        events.append((op.ret, 1, i, ret))
    events.sort(key=lambda t: (t[0], t[1], t[2]))
    head = _Event(ops[0], True)  # sentinel
    cur = head
    for *_, ev in events:
        cur.next = ev
        ev.prev = cur
        cur = ev
    return head


def _lift(call: _Event) -> None:
    assert call.prev is not None and call.match is not None
    call.prev.next = call.next
    if call.next is not None:
        call.next.prev = call.prev
    ret = call.match
    assert ret.prev is not None
    ret.prev.next = ret.next
    if ret.next is not None:
        ret.next.prev = ret.prev


def _unlift(call: _Event) -> None:
    ret = call.match
    assert ret is not None and ret.prev is not None and call.prev is not None
    ret.prev.next = ret
    if ret.next is not None:
        ret.next.prev = ret
    call.prev.next = call
    if call.next is not None:
        call.next.prev = call


def check_operations(
    ops: Sequence[Operation], model: Model, *, max_states: int = 5_000_000
) -> tuple[bool, list[Operation]]:
    """Check a single-object history. Returns (ok, best linearization prefix)."""
    if not ops:
        return True, []
    ops = list(ops)
    for i, op in enumerate(ops):
        op.id = i
    head = _build_list(ops)
    state: Hashable = model.init()
    linearized = 0  # bitset of op ids
    cache: set[tuple[int, Hashable]] = set()
    stack: list[tuple[_Event, Hashable]] = []
    best: list[Operation] = []
    entry = head.next
    while head.next is not None:
        assert entry is not None
        if entry.is_call:
            ok, new_state = model.step(state, entry.op)
            if ok:
                new_lin = linearized | (1 << entry.op.id)
                key = (new_lin, new_state)
                if key not in cache:
                    if len(cache) >= max_states:
                        raise RuntimeError("linearizability search exceeded max_states")
                    cache.add(key)
                    stack.append((entry, state))
                    state, linearized = new_state, new_lin
                    _lift(entry)
                    if len(stack) > len(best):
                        best = [e.op for e, _ in stack]
                    entry = head.next
                    continue
            entry = entry.next
        else:
            # A return event whose op is still unlinearized: dead end.
            if not stack:
                return False, best
            call, state = stack.pop()
            linearized &= ~(1 << call.op.id)
            _unlift(call)
            entry = call.next
    return True, [e.op for e, _ in stack]


def check_history(
    ops: Iterable[Operation], model_factory: type[Model] = RegisterModel
) -> CheckResult:
    """Check a multi-key KV history by checking each key independently."""
    by_key: dict[str, list[Operation]] = defaultdict(list)
    total = 0
    for op in ops:
        by_key[op.key].append(op)
        total += 1
    for key in sorted(by_key):
        key_ops = by_key[key]
        ok, best = check_operations(key_ops, model_factory())
        if not ok:
            done = {id(o) for o in best}
            return CheckResult(False, total, key, best, [o for o in key_ops if id(o) not in done])
    return CheckResult(True, total)


def brute_force_check(ops: Sequence[Operation], model: Model) -> bool:
    """Exponential reference checker (try every real-time-respecting order).

    Only for cross-validating :func:`check_operations` on tiny histories.
    """
    ops = list(ops)
    n = len(ops)

    def search(remaining: frozenset[int], state: Hashable) -> bool:
        if not remaining:
            return True
        for i in remaining:
            # i may go next only if no other remaining op returned before i was called.
            if any(ops[j].ret < ops[i].call for j in remaining if j != i):
                continue
            ok, new_state = model.step(state, ops[i])
            if ok and search(remaining - {i}, new_state):
                return True
        return False

    return search(frozenset(range(n)), model.init())
