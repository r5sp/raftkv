#!/usr/bin/env python3
"""Failover demo: 3 real server processes, kill the leader, keep serving.

    python scripts/demo_cluster.py

1. starts a 3-node cluster on localhost (temporary data directory)
2. writes and reads some keys
3. SIGKILLs the leader process and measures time until a new leader is elected
4. shows the data survived and the cluster still accepts writes
5. restarts the killed node and shows it catches up from its own disk + the leader
"""

from __future__ import annotations

import asyncio
import contextlib
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from raftkv.client import ClusterUnavailable, RaftClient, connect
from raftkv.local_cluster import member_spec, spawn_node

BASE_PORT = 7101


def say(msg: str) -> None:
    print(f"\n==> {msg}", flush=True)


async def statuses(client: RaftClient) -> dict[str, dict[str, object]]:
    out = {}
    for server in client.servers:
        with contextlib.suppress(ClusterUnavailable):
            out[server] = await client.status(server)
    return out


async def wait_for_leader(
    client: RaftClient, exclude: str | None = None, timeout: float = 10.0
) -> str:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        for sid, st in (await statuses(client)).items():
            if st["role"] == "leader" and sid != exclude:
                return sid
        await asyncio.sleep(0.05)
    raise TimeoutError("no leader")


def print_table(sts: dict[str, dict[str, object]], ids: list[str]) -> None:
    print(f"    {'id':>3} {'role':>10} {'term':>5} {'commit':>7} {'applied':>8}")
    for sid in ids:
        st = sts.get(sid)
        if st is None:
            print(f"    {sid:>3} {'DOWN':>10}")
        else:
            print(
                f"    {sid:>3} {st['role']!s:>10} {st['term']!s:>5} {st['commit_index']!s:>7} {st['last_applied']!s:>8}"
            )


async def demo(data_dir: Path) -> None:
    spec = member_spec(3, BASE_PORT)
    ids = ["1", "2", "3"]
    quiet = ["--log-level", "WARNING"]
    procs = {i: spawn_node(int(i), spec, data_dir, quiet) for i in ids}
    client = connect(spec, request_timeout=0.3, deadline=15.0)
    probe = connect(spec, request_timeout=0.2, deadline=1.0)
    try:
        say(f"started 3 servers ({spec}); waiting for an election")
        leader = await wait_for_leader(probe)
        print(f"    leader is node {leader}")
        print_table(await statuses(probe), ids)

        say("writing 5 keys")
        for i in range(5):
            await client.put(f"city{i}", ["Lisbon", "Osaka", "Lagos", "Quito", "Oslo"][i])
        print("    city2 =", await client.get("city2"))
        swapped, cur = await client.cas("city2", "Lagos", "Nairobi")
        print(f"    cas(city2, Lagos -> Nairobi): swapped={swapped}, now {cur}")

        say(f"SIGKILL leader (node {leader})")
        t0 = time.monotonic()
        procs[leader].send_signal(signal.SIGKILL)
        procs[leader].wait()
        new_leader = await wait_for_leader(probe, exclude=leader)
        print(
            f"    new leader: node {new_leader}, observed {1000 * (time.monotonic() - t0):.0f} ms after the kill"
        )
        print_table(await statuses(probe), ids)

        say("reading through the new leader; writing more")
        print("    city2 =", await client.get("city2"))
        await client.put("after-failover", "still-serving")
        print("    after-failover =", await client.get("after-failover"))

        say(f"restarting node {leader} from its data directory")
        procs[leader] = spawn_node(int(leader), spec, data_dir, quiet)
        end = time.monotonic() + 10
        while time.monotonic() < end:
            sts = await statuses(probe)
            if len(sts) == 3 and len({s["last_applied"] for s in sts.values()}) == 1:
                break
            await asyncio.sleep(0.1)
        print_table(await statuses(probe), ids)
        print(f"    node {leader} rejoined as a follower and caught up.")
    finally:
        await client.close()
        await probe.close()
        for p in procs.values():
            if p.poll() is None:
                p.terminate()
        for p in procs.values():
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="raftkv-demo-") as d:
        asyncio.run(demo(Path(d)))
    say("done")


if __name__ == "__main__":
    main()
