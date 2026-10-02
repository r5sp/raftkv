"""Launch an N-node raftkv cluster on localhost (one process per node).

    python -m raftkv.local_cluster --nodes 3      # or 5
    python -m raftkv.cli                          # in another terminal

Ctrl-C stops every node. Data is kept in --data-dir between runs unless
--clean is given.
"""

from __future__ import annotations

import argparse
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path


def member_spec(nodes: int, base_port: int, host: str = "127.0.0.1") -> str:
    return ",".join(f"{i}={host}:{base_port + i - 1}" for i in range(1, nodes + 1))


def spawn_node(
    node_id: int, spec: str, data_dir: Path, extra: list[str] | None = None
) -> subprocess.Popen[bytes]:
    cmd = [
        sys.executable,
        "-m",
        "raftkv.server",
        "--id",
        str(node_id),
        "--peers",
        spec,
        "--data-dir",
        str(data_dir / str(node_id)),
        *(extra or []),
    ]
    return subprocess.Popen(cmd)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        prog="python -m raftkv.local_cluster", description="run a local raftkv cluster"
    )
    p.add_argument("--nodes", type=int, default=3, choices=[1, 3, 5, 7])
    p.add_argument("--base-port", type=int, default=7001)
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--clean", action="store_true", help="delete --data-dir first")
    args = p.parse_args(argv)

    if args.clean and args.data_dir.exists():
        shutil.rmtree(args.data_dir)
    spec = member_spec(args.nodes, args.base_port)
    procs = [spawn_node(i, spec, args.data_dir) for i in range(1, args.nodes + 1)]
    print(f"started {args.nodes} nodes: {spec}")
    print(f"connect with: python -m raftkv.cli --peers {spec}")

    def shutdown(*_: object) -> None:
        for proc in procs:
            if proc.poll() is None:
                proc.terminate()
        for proc in procs:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
