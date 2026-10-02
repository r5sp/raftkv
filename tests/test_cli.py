"""CLI entry points: command parsing in-process, and a real multi-process smoke test."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pytest

from raftkv.cli import run_command
from raftkv.local_cluster import member_spec, spawn_node
from raftkv.server import build_parser
from raftkv.sim import SimCluster, run_simulation


def test_repl_commands_against_simulated_cluster() -> None:
    async def main() -> list[str]:
        cluster = SimCluster(3, seed=3)
        await cluster.start()
        await cluster.wait_for_leader()
        client = cluster.client("cli")
        out = [
            await run_command(client, ["put", "k", "v"]),
            await run_command(client, ["get", "k"]),
            await run_command(client, ["cas", "k", "v", "w"]),
            await run_command(client, ["cas", "k", "-", "z"]),
            await run_command(client, ["del", "k"]),
            await run_command(client, ["get", "k"]),
            await run_command(client, ["status"]),
        ]
        with pytest.raises(ValueError, match="bad command"):
            await run_command(client, ["frobnicate"])
        await cluster.stop()
        return out

    out = run_simulation(main)
    assert out[:6] == [
        "OK (previous: (nil))",
        "v",
        "swapped; current = w",
        "not swapped; current = w",
        "deleted",
        "(nil)",
    ]
    assert "leader" in out[6]
    assert out[6].count("follower") == 2


def test_server_arg_parsing() -> None:
    args = build_parser().parse_args(
        ["--id", "2", "--peers", "1=a:1,2=b:2", "--election-timeout-ms", "200", "400"]
    )
    assert args.id == "2"
    assert args.election_timeout_ms == [200, 400]


def test_processes_end_to_end(tmp_path: Path) -> None:
    """Three `python -m raftkv.server` processes driven by `python -m raftkv.cli`."""
    from test_tcp import free_ports

    ports = free_ports(3)
    spec = ",".join(f"{i}=127.0.0.1:{p}" for i, p in enumerate(ports, 1))
    assert member_spec(3, 7001).startswith("1=127.0.0.1:7001")
    procs = [spawn_node(i, spec, tmp_path, ["--log-level", "WARNING"]) for i in (1, 2, 3)]

    def cli(*words: str) -> str:
        res = subprocess.run(
            [sys.executable, "-m", "raftkv.cli", "--peers", spec, "--deadline", "15", *words],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert res.returncode == 0, res.stderr
        return res.stdout.strip()

    try:
        time.sleep(0.3)
        assert cli("put", "greeting", "hello") == "OK (previous: (nil))"
        assert cli("get", "greeting") == "hello"
        status = cli("status")
        assert status.count("leader") >= 1
    finally:
        for p in procs:
            p.terminate()
        for p in procs:
            p.wait(timeout=10)
    assert all((tmp_path / str(i) / "log.bin").exists() for i in (1, 2, 3))
