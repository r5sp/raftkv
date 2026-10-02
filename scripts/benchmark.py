#!/usr/bin/env python3
"""Closed-loop throughput/latency benchmark against a real local 3-node cluster.

    python scripts/benchmark.py [--duration 5] [--concurrency 1 8 32]

Starts three server processes on localhost (fsync enabled, temporary data
directory), then for each concurrency level runs N independent clients, each
issuing requests back-to-back for --duration seconds: first writes (put),
then linearizable reads (get via ReadIndex). Prints a Markdown table.
"""

from __future__ import annotations

import argparse
import asyncio
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from raftkv.client import ClusterUnavailable, RaftClient, connect
from raftkv.local_cluster import member_spec, spawn_node

BASE_PORT = 7201


async def wait_ready(spec: str) -> None:
    client = connect(spec, request_timeout=0.3, deadline=20.0)
    try:
        await client.put("__warmup__", "1")
    finally:
        await client.close()


async def worker(
    client: RaftClient, kind: str, stop_at: float, value: str, lat: list[float], wid: int
) -> None:
    i = 0
    while time.perf_counter() < stop_at:
        key = f"w{wid}-k{i % 100}"
        t0 = time.perf_counter()
        if kind == "put":
            await client.put(key, value)
        else:
            await client.get(key)
        lat.append(time.perf_counter() - t0)
        i += 1


async def run_level(
    spec: str, kind: str, concurrency: int, duration: float, value: str
) -> dict[str, float]:
    clients = [connect(spec, request_timeout=2.0, deadline=30.0) for _ in range(concurrency)]
    try:
        for c in clients:  # connect + discover leader before timing
            await c.put("__warmup__", "1")
        lats: list[list[float]] = [[] for _ in clients]
        start = time.perf_counter()
        stop_at = start + duration
        await asyncio.gather(
            *(worker(c, kind, stop_at, value, lats[i], i) for i, c in enumerate(clients))
        )
        elapsed = time.perf_counter() - start
    finally:
        for c in clients:
            await c.close()
    all_lat = sorted(x for lat in lats for x in lat)
    n = len(all_lat)
    return {
        "ops": n,
        "throughput": n / elapsed,
        "p50": 1000 * all_lat[n // 2],
        "p99": 1000 * all_lat[min(n - 1, int(n * 0.99))],
        "mean": 1000 * statistics.fmean(all_lat),
    }


async def bench(args: argparse.Namespace, data_dir: Path) -> None:
    spec = member_spec(3, BASE_PORT)
    procs = [
        spawn_node(
            i,
            spec,
            data_dir,
            ["--log-level", "WARNING", "--snapshot-threshold", "10000"]
            + (["--full-fsync"] if args.full_fsync else []),
        )
        for i in (1, 2, 3)
    ]
    try:
        await wait_ready(spec)
        value = "x" * args.value_size
        sync = "F_FULLFSYNC" if args.full_fsync else "fsync"
        print(
            f"\nraftkv benchmark: 3 nodes on localhost, {sync} on every log write, {args.value_size}-byte values, "
            f"{args.duration:.0f}s per row"
        )
        print(
            f"host: {platform.machine()} / {platform.system()} {platform.release()} / "
            f"Python {platform.python_version()}\n"
        )
        print("| operation | clients | ops/s | p50 ms | p99 ms |")
        print("|---|---:|---:|---:|---:|")
        for kind in ("put", "get"):
            for conc in args.concurrency:
                r = await run_level(spec, kind, conc, args.duration, value)
                label = "put (write)" if kind == "put" else "get (ReadIndex read)"
                print(
                    f"| {label} | {conc} | {r['throughput']:,.0f} | {r['p50']:.2f} | {r['p99']:.2f} |",
                    flush=True,
                )
    except ClusterUnavailable as exc:
        print(f"cluster unavailable: {exc}", file=sys.stderr)
    finally:
        for p in procs:
            p.terminate()
        for p in procs:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--duration", type=float, default=5.0)
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 8, 32])
    ap.add_argument("--value-size", type=int, default=64)
    ap.add_argument("--full-fsync", action="store_true", help="servers use F_FULLFSYNC (macOS)")
    args = ap.parse_args()
    with tempfile.TemporaryDirectory(prefix="raftkv-bench-") as d:
        asyncio.run(bench(args, Path(d)))


if __name__ == "__main__":
    main()
