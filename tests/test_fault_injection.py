"""Randomized fault injection: many seeds, every history must be linearizable.

Each run starts a cluster on a network that drops, duplicates and reorders
messages, runs concurrent clients, and lets a nemesis partition the network,
isolate leaders, and crash/restart nodes (see ``raftkv.sim.workload``).
Throughout, the InvariantChecker enforces Raft's safety properties; at the
end the recorded client history is checked for linearizability and all
replicas must have converged to identical state.

Every run is deterministic given its seed, so any failure is reproducible
with ``pytest tests/test_fault_injection.py -k "seed-<n>"``.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from pathlib import Path

import pytest

from raftkv.linearizability import check_history
from raftkv.messages import NodeId
from raftkv.sim import NetworkConfig, run_simulation
from raftkv.sim.workload import ScenarioConfig, ScenarioResult, run_fault_scenario
from raftkv.storage import FileStorage, Storage

FIVE = ScenarioConfig(nodes=5, clients=5, keys=3, duration=15.0)
THREE = ScenarioConfig(nodes=3, clients=4, keys=2, duration=15.0)
HOSTILE = ScenarioConfig(
    nodes=5,
    clients=5,
    keys=2,
    duration=15.0,
    crashes=False,
    network=NetworkConfig(drop_rate=0.25, duplicate_rate=0.25, min_delay=0.001, max_delay=0.08),
)


def run_and_check(
    seed: int | str,
    config: ScenarioConfig,
    storage_factory: Callable[[NodeId], Storage] | None = None,
    min_ops: int = 40,
) -> ScenarioResult:
    result = run_simulation(lambda: run_fault_scenario(seed, config, storage_factory))
    cluster = result.cluster
    context = f"seed={seed}\nfaults:\n  " + "\n  ".join(result.faults)
    assert cluster.errors == [], context
    assert cluster.checker.violations == [], context
    check = check_history(result.history)
    assert check.ok, (
        f"non-linearizable history on key {check.failed_key!r}\n{context}\n"
        f"unresolved: {check.unresolved[:10]}"
    )
    # Guard against vacuous passes: the workload must actually have run.
    assert result.completed_ops >= min_ops, f"only {result.completed_ops} ops completed\n{context}"
    assert len(cluster.checker.committed) > 0
    return result


SNAPSHOTS_SEEN: list[int] = []


@pytest.mark.parametrize("seed", range(100), ids=lambda s: f"seed-{s}")
def test_five_nodes_partitions_and_crashes(seed: int) -> None:
    result = run_and_check(seed, FIVE)
    SNAPSHOTS_SEEN.append(result.cluster.snapshots_installed())


@pytest.mark.parametrize("seed", range(50), ids=lambda s: f"seed-{s}")
def test_three_nodes_partitions_and_crashes(seed: int) -> None:
    run_and_check(f"three-{seed}", THREE)


@pytest.mark.parametrize("seed", range(30), ids=lambda s: f"seed-{s}")
def test_hostile_network_without_crashes(seed: int) -> None:
    run_and_check(f"hostile-{seed}", HOSTILE, min_ops=15)


@pytest.mark.parametrize("seed", range(3), ids=lambda s: f"seed-{s}")
def test_crashes_with_real_disk_storage(seed: int, tmp_path: Path) -> None:
    run_and_check(
        f"disk-{seed}",
        dataclasses.replace(THREE, duration=6.0),
        lambda i: FileStorage(tmp_path / i),
    )


def test_snapshots_were_exercised_under_faults() -> None:
    """The fault runs above must have driven InstallSnapshot at least sometimes."""
    if len(SNAPSHOTS_SEEN) < 100:
        pytest.skip("run together with test_five_nodes_partitions_and_crashes")
    assert sum(1 for n in SNAPSHOTS_SEEN if n > 0) >= 10


def test_runs_are_deterministic() -> None:
    def fingerprint(seed: int) -> list[tuple[object, ...]]:
        r = run_simulation(
            lambda: run_fault_scenario(seed, dataclasses.replace(FIVE, duration=4.0))
        )
        return [(o.client, o.kind, o.key, o.call, o.ret, str(o.output)) for o in r.history] + [
            tuple(r.faults)
        ]

    assert fingerprint(11) == fingerprint(11)
    assert fingerprint(11) != fingerprint(12)


@pytest.mark.slow
@pytest.mark.parametrize("seed", range(1000, 3000), ids=lambda s: f"seed-{s}")
def test_soak_five_nodes(seed: int) -> None:
    run_and_check(seed, dataclasses.replace(FIVE, duration=30.0))
