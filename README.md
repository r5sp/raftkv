# raftkv

Raft-backed key-value store in Python 3.11+ asyncio. No runtime deps.

It does the stuff from the Raft paper (elections, replication, persistence, snapshots) plus exactly-once client sessions and linearizable reads via ReadIndex. The part I actually care about is the testing. There's a deterministic simulator that runs the real node code over a network that drops, duplicates, reorders and partitions messages while nodes crash and restart, and a linearizability checker that looks at every client history afterwards.

```
$ python scripts/demo_cluster.py
==> SIGKILL leader (node 1)
    new leader: node 2, observed 203 ms after the kill
==> reading through the new leader; writing more
    city2 = Nairobi
==> restarting node 1 from its data directory
    node 1 rejoined as a follower and caught up.
```

## running it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# whole cluster, one process per node
python -m raftkv.local_cluster --nodes 3          # Ctrl-C to stop

# or by hand
PEERS=1=127.0.0.1:7001,2=127.0.0.1:7002,3=127.0.0.1:7003
python -m raftkv.server --id 1 --peers $PEERS --data-dir data/1
python -m raftkv.server --id 2 --peers $PEERS --data-dir data/2
python -m raftkv.server --id 3 --peers $PEERS --data-dir data/3
```

`--peers` is the full static membership, including the node itself. Then:

```text
$ python -m raftkv.cli
raftkv> put greeting hello
OK (previous: (nil))
raftkv> cas greeting hello hi
swapped; current = hi
raftkv> status
  id       role  term leader  commit applied   snap
   1     leader     1      1       3       3      0
   2   follower     1      1       3       3      0
   3   follower     1      1       3       3      0
```

One-shot works too: `python -m raftkv.cli get greeting`. From code:

```python
from raftkv.client import connect

client = connect("1=127.0.0.1:7001,2=127.0.0.1:7002,3=127.0.0.1:7003")
await client.put("k", "v")
assert await client.get("k") == "v"
swapped, current = await client.cas("k", expected="v", value="w")
```

## how it works

```
  RaftClient / cli
        |
    Transport  (TcpTransport, or SimTransport in tests)
        |
    RaftNode   node.py   asyncio timers, client bookkeeping
        |
    RaftCore   core.py   pure consensus, no I/O
      /    \
  KVStateMachine   FileStorage (fsync before reply)
```

`core.py` is Raft as a plain state machine, etcd/raft style. You feed it `step(src, msg, now)` and `tick(now)` and it fills an outbox. No sockets, no clocks, no tasks. `node.py` drives it from an asyncio loop. `storage.py` has `FileStorage` and a `MemoryStorage` the simulator uses as a disk that survives crashes. Client is in `client.py`, simulator stuff in `raftkv/sim/`, checker in `linearizability.py`.

Some notes, assuming you've read the paper:

Election timeouts are random in `[150, 300]` ms. `votedFor` is persisted before replying. Two leaders in one term is an assertion failure in the core.

Followers only truncate on a real conflict, never just because an incoming message is shorter than their log (it might be an old reordered one). Rejections carry `conflictTerm` hints so the leader skips a whole term per round trip. There's a test that repairs a 40-entry divergent suffix with at most 2 rejections.

Commit needs a quorum and `log[N].term == currentTerm`. `tests/test_core_replication.py` replays Figure 8 (c), (d) and (e) message by message. New leaders append a no-op.

On disk: `state.json` (atomic replace), `log.bin` (append-only `[len][crc32][json]` records, torn tail gets truncated on load), `snapshot.bin` (atomic replace). Everything goes to storage synchronously before the core emits any message that depends on it. Recovery also handles a crash between writing a snapshot and rewriting the log.

Sessions: each client has a `client_id` and a `seq`, retries reuse the `seq`, and the state machine caches the last result per client. One test does 60 back-to-back CAS increments with 20% drop and 30% duplication, and every CAS has to succeed. Without sessions a retried CAS would re-run and return `swapped=False`.

Reads use ReadIndex, not the log. Leader waits for its no-op to commit, records `commitIndex`, confirms with a quorum heartbeat round, then serves locally. So a partitioned ex-leader never answers a read (`test_partitioned_leader_never_serves_read`). I picked ReadIndex over leases because it doesn't depend on bounded clock drift.

## testing

```
pytest          # 311 tests, about 15 s on an Apple Silicon laptop
pytest -m slow  # +2,000 fault-injection seeds of 30 simulated seconds (about 4 min)
```

`raftkv.sim.VirtualTimeLoop` is a normal `asyncio.SelectorEventLoop` whose clock jumps forward whenever it would sleep. So the real runtime, client, `asyncio.sleep` and `wait_for` timeouts all run unmodified, a 15-second scenario with 5 nodes and 5 clients takes about 50 ms, and everything is reproducible from the seed (`test_runs_are_deterministic`).

Each scenario runs 3 or 5 nodes for 15 simulated seconds. Network has 5% loss, 5% duplication and 1-30 ms random delay; the "hostile" variant is 25% / 25% / up to 80 ms. A nemesis fires every 0.3-1.2 s and does partitions, leader isolation, bridge topologies, crashes (minority at most) and restarts. 4-5 clients hammer 2-3 keys with `get`/`put`/`delete`/`cas`. Three seeds use real `FileStorage` on disk.

A run passes if the Figure 3 invariants hold the whole time, the history is linearizable, the cluster elects a leader and finishes a write once faults stop, replicas converge, and at least 40 ops completed (so a stuck cluster can't pass by doing nothing).

The checker is Wing & Gong backtracking with Lowe's memoization, same idea as Knossos and Porcupine, checked per key since linearizability is local. A 2,500-operation history with 8 overlapping clients checks in about 10 ms. It's also tested against a brute-force oracle on 1,500 random small histories.

To make sure the harness can actually fail, `test_bug_detection.py` monkeypatches six classic Raft bugs in and requires each one to get caught:

- reads served locally without ReadIndex (checker catches a stale read)
- `votedFor`/`currentTerm` not persisted
- no up-to-date check in `RequestVote`
- no session dedup
- committing old-term entries by counting replicas (Figure 8 test)
- follower truncating on every `AppendEntries`

## numbers

`scripts/benchmark.py` starts three local server processes and runs N closed-loop clients for 5 seconds per row, 64-byte values. Apple Silicon Mac, macOS 26.4, Python 3.14, single run:

| operation | clients | ops/s | p50 ms | p99 ms |
|---|---:|---:|---:|---:|
| put (write) | 1 | 3,586 | 0.27 | 0.35 |
| put (write) | 8 | 11,063 | 0.71 | 0.98 |
| put (write) | 32 | 21,397 | 1.46 | 2.02 |
| get (ReadIndex read) | 1 | 4,102 | 0.24 | 0.27 |
| get (ReadIndex read) | 8 | 21,525 | 0.37 | 0.47 |
| get (ReadIndex read) | 32 | 44,756 | 0.71 | 0.89 |

A repeat came within about 10%. Caveat: on macOS plain `fsync()` doesn't flush the drive cache. With `--full-fsync` (`F_FULLFSYNC`, survives power loss) writes drop to 90 ops/s at 1 client (11.05 ms p50) and 467 ops/s at 32 clients. Reads barely change since they never touch disk.

The leader does group commit, but followers still fsync once per `AppendEntries` with one batch in flight, which is why durable writes scale badly. Also every node is one Python process on one core, over loopback. Don't read too much into it.

## not done yet

- static membership, no config changes
- no PreVote / CheckQuorum, so a rejoining node with an inflated term forces a pointless election (availability issue, not safety)
- `InstallSnapshot` sends the whole thing in one message, and snapshotting blocks the event loop
- sessions never expire
- one outstanding request per client
- disk I/O is synchronous on the event loop, no replication pipelining, no TLS or auth
- simulator doesn't inject disk faults mid-run (torn writes and corruption are covered in `test_storage.py`)

Would like to get to pipelining, PreVote, leadership transfer, membership changes and maybe sharding at some point.

## references

- Ongaro & Ousterhout, *In Search of an Understandable Consensus Algorithm (Extended Version)*, 2014
- Ongaro, *Consensus: Bridging Theory and Practice*, PhD dissertation, Stanford, 2014
- Wing & Gong, *Testing and Verifying Concurrent Objects*, JPDC, 1993
- Lowe, *Testing for Linearizability*, Concurrency and Computation, 2017

MIT, see [LICENSE](LICENSE).
