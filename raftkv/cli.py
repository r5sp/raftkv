"""Interactive client for a raftkv cluster.

python -m raftkv.cli                         # REPL against the default 3-node local cluster
python -m raftkv.cli --peers 1=host:7001,... # custom cluster
python -m raftkv.cli put greeting hello      # one-shot command
"""

from __future__ import annotations

import argparse
import asyncio
import shlex
import sys

from raftkv.client import ClusterUnavailable, RaftClient, RaftKVError, connect

DEFAULT_PEERS = "1=127.0.0.1:7001,2=127.0.0.1:7002,3=127.0.0.1:7003"

HELP = """commands:
  get <key>                      read (linearizable, via ReadIndex)
  put <key> <value>              write; prints the previous value
  del <key>                      delete
  cas <key> <expected> <new>     compare-and-swap; use - for "absent"
  status                         role/term/commit index of every server
  help | quit"""


async def run_command(client: RaftClient, words: list[str]) -> str:
    cmd, args = words[0].lower(), words[1:]
    if cmd == "get" and len(args) == 1:
        value = await client.get(args[0])
        return "(nil)" if value is None else value
    if cmd == "put" and len(args) == 2:
        prev = await client.put(args[0], args[1])
        return f"OK (previous: {'(nil)' if prev is None else prev})"
    if cmd in ("del", "delete") and len(args) == 1:
        return "deleted" if await client.delete(args[0]) else "(not found)"
    if cmd == "cas" and len(args) == 3:
        expected = None if args[1] == "-" else args[1]
        swapped, current = await client.cas(args[0], expected, args[2])
        return f"{'swapped' if swapped else 'not swapped'}; current = {current if current is not None else '(nil)'}"
    if cmd == "status":
        rows = [
            f"{'id':>4} {'role':>10} {'term':>5} {'leader':>6} {'commit':>7} {'applied':>7} {'snap':>6}"
        ]
        for server in client.servers:
            try:
                s = await client.status(server)
                rows.append(
                    f"{s['id']:>4} {s['role']:>10} {s['term']:>5} {s['leader'] or '-':>6} "
                    f"{s['commit_index']:>7} {s['last_applied']:>7} {s['snapshot_index']:>6}"
                )
            except ClusterUnavailable:
                rows.append(f"{server:>4} {'(down)':>10}")
        return "\n".join(rows)
    if cmd == "help":
        return HELP
    raise ValueError(f"bad command: {' '.join(words)!r} (try 'help')")


async def repl(client: RaftClient) -> None:
    print(f"raftkv client {client.client_id[:8]} -> {', '.join(client.servers)}. Type 'help'.")
    while True:
        try:
            line = await asyncio.to_thread(input, "raftkv> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return
        words = shlex.split(line)
        if not words:
            continue
        if words[0] in ("quit", "exit"):
            return
        try:
            print(await run_command(client, words))
        except (ValueError, RaftKVError, ClusterUnavailable) as exc:
            print(f"error: {exc}")


async def amain(args: argparse.Namespace) -> int:
    client = connect(args.peers, request_timeout=args.timeout, deadline=args.deadline)
    try:
        if args.command:
            try:
                print(await run_command(client, args.command))
            except (ValueError, RaftKVError, ClusterUnavailable) as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 1
        else:
            await repl(client)
    finally:
        await client.close()
    return 0


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m raftkv.cli", description="raftkv client")
    p.add_argument("--peers", default=DEFAULT_PEERS, help=f"cluster spec (default {DEFAULT_PEERS})")
    p.add_argument("--timeout", type=float, default=0.5, help="per-attempt timeout, seconds")
    p.add_argument(
        "--deadline", type=float, default=10.0, help="overall deadline per command, seconds"
    )
    p.add_argument("command", nargs=argparse.REMAINDER, help="optional one-shot command")
    sys.exit(asyncio.run(amain(p.parse_args(argv))))


if __name__ == "__main__":
    main()
