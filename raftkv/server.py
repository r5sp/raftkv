"""Run one raftkv server process.

Example (three terminals)::

    python -m raftkv.server --id 1 --peers 1=127.0.0.1:7001,2=127.0.0.1:7002,3=127.0.0.1:7003
    python -m raftkv.server --id 2 --peers 1=127.0.0.1:7001,2=127.0.0.1:7002,3=127.0.0.1:7003
    python -m raftkv.server --id 3 --peers 1=127.0.0.1:7001,2=127.0.0.1:7002,3=127.0.0.1:7003

``--peers`` lists the *full* static membership, including this server; this
server listens on its own entry's address.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import signal
from pathlib import Path

from raftkv.core import RaftConfig, Role
from raftkv.node import RaftNode
from raftkv.storage import FileStorage
from raftkv.transport import TcpTransport, parse_members

log = logging.getLogger("raftkv.server")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m raftkv.server", description=__doc__.split("\n\n")[0]
    )
    p.add_argument("--id", required=True, help="this server's id (must appear in --peers)")
    p.add_argument("--peers", required=True, help="full membership: id=host:port,id=host:port,...")
    p.add_argument(
        "--data-dir", type=Path, help="directory for durable state (default: ./data/<id>)"
    )
    p.add_argument(
        "--snapshot-threshold",
        type=int,
        default=10_000,
        help="entries between snapshots (0 = never)",
    )
    p.add_argument(
        "--election-timeout-ms", type=int, nargs=2, default=[150, 300], metavar=("MIN", "MAX")
    )
    p.add_argument("--heartbeat-ms", type=int, default=50)
    p.add_argument(
        "--full-fsync",
        action="store_true",
        help="use F_FULLFSYNC on macOS (slower, survives power loss)",
    )
    p.add_argument("--log-level", default="INFO")
    return p


async def serve(args: argparse.Namespace) -> None:
    members = parse_members(args.peers)
    if args.id not in members:
        raise SystemExit(f"--id {args.id!r} is not listed in --peers")
    config = RaftConfig(
        election_timeout_min=args.election_timeout_ms[0] / 1000,
        election_timeout_max=args.election_timeout_ms[1] / 1000,
        heartbeat_interval=args.heartbeat_ms / 1000,
        snapshot_threshold=args.snapshot_threshold,
    )
    data_dir = args.data_dir or Path("data") / args.id
    storage = FileStorage(data_dir, full_fsync=args.full_fsync)
    transport = TcpTransport(args.id, members, listen=members[args.id])
    node = RaftNode(args.id, list(members), transport, storage, config=config)
    await node.start()
    host, port = members[args.id]
    log.info("server %s listening on %s:%d, data in %s", args.id, host, port, data_dir)

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, stop.set)

    last: tuple[Role, int, str | None] | None = None
    while not stop.is_set():
        core = node.core
        now = (core.role, core.current_term, core.leader_id)
        if now != last:
            if core.role is Role.LEADER:
                log.info("became LEADER for term %d", core.current_term)
            else:
                log.info(
                    "%s in term %d (leader: %s)", core.role.value, core.current_term, core.leader_id
                )
            last = now
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(stop.wait(), 0.1)
    log.info("shutting down")
    await node.stop()
    storage.close()


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format=f"%(asctime)s [node {args.id}] %(levelname)s %(message)s",
    )
    asyncio.run(serve(args))


if __name__ == "__main__":
    main()
