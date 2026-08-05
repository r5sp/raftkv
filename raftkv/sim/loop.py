"""A deterministic asyncio event loop that runs on virtual time.

``VirtualTimeLoop`` is a normal :class:`asyncio.SelectorEventLoop` except that
``time()`` returns a simulated clock and, whenever the loop would block
waiting for the next timer, the clock simply jumps forward instead of
sleeping. The *real* runtime code (timers, ``asyncio.sleep``, ``wait_for``,
the client library) therefore runs unmodified, but a 30-second simulated
fault-injection run finishes in well under a second of wall time, and --
given the same seed -- executes exactly the same interleaving every time.

Requirements for determinism: no real I/O or threads inside the simulation,
and all randomness drawn from seeded ``random.Random`` instances.
"""

from __future__ import annotations

import asyncio
import selectors
from collections.abc import Callable, Coroutine, Mapping
from typing import Any, TypeVar

T = TypeVar("T")


class SimulationDeadlock(RuntimeError):
    """The loop has nothing scheduled and nothing can ever wake it up."""


class _VirtualSelector(selectors.BaseSelector):
    """Wraps a real selector; replaces blocking waits with clock jumps."""

    def __init__(self, advance: Callable[[float], None]) -> None:
        self._real = selectors.DefaultSelector()
        self._advance = advance

    def register(self, fileobj: Any, events: int, data: Any = None) -> selectors.SelectorKey:
        return self._real.register(fileobj, events, data)

    def unregister(self, fileobj: Any) -> selectors.SelectorKey:
        return self._real.unregister(fileobj)

    def modify(self, fileobj: Any, events: int, data: Any = None) -> selectors.SelectorKey:
        return self._real.modify(fileobj, events, data)

    def select(self, timeout: float | None = None) -> list[tuple[selectors.SelectorKey, int]]:
        ready = self._real.select(0)
        if ready or timeout == 0:
            return ready
        if timeout is None:
            # Nothing scheduled at all. Give real I/O (e.g. the loop's
            # self-pipe) a brief chance, then declare deadlock.
            ready = self._real.select(0.05)
            if ready:
                return ready
            raise SimulationDeadlock("virtual-time loop has no pending timers or I/O")
        self._advance(timeout)
        return []

    def get_map(self) -> Mapping[Any, selectors.SelectorKey]:
        return self._real.get_map()

    def close(self) -> None:
        self._real.close()


class VirtualTimeLoop(asyncio.SelectorEventLoop):
    def __init__(self, start: float = 0.0) -> None:
        self._virtual_now = start
        super().__init__(selector=_VirtualSelector(self._advance))

    def _advance(self, dt: float) -> None:
        self._virtual_now += dt

    def time(self) -> float:
        return self._virtual_now


def run_simulation(main: Callable[[], Coroutine[Any, Any, T]]) -> T:
    """Run ``main()`` to completion on a fresh :class:`VirtualTimeLoop`."""
    with asyncio.Runner(loop_factory=VirtualTimeLoop) as runner:
        return runner.run(main())
