# SuperBrain Subnet — CLAUDE.md
# Auto-loaded project context for Claude Code sessions
# Last updated: 2026-03-25

---

## Project Identity

**SuperBrain** — Local-First Anonymous Knowledge Network on Bittensor
**Author:** Lys-David Louis-Charles (KatchDaVizion) — Ottawa, Canada
**Repo:** https://github.com/KatchDaVizion/superbrain-subnet (MUST stay private)
**Competition:** Bittensor Subnet Ideathon 2026

> "Private by default. Public by choice. Wherever one SuperBrain remains, knowledge shall survive."

---

## Architecture Overview

```
┌────────────────────────────────────────────────────────────────────┐
│                        SUPERBRAIN ECOSYSTEM                         │
│                                                                     │
│  ┌──────────────────────────┐    ┌──────────────────────────┐      │
│  │     DESKTOP APP          │    │     BITTENSOR SUBNET      │      │
│  │  (Electron + React)      │    │     (SN442 Testnet)       │      │
│  │                          │    │                           │      │
│  │  Ollama ──► Local RAG    │    │  Validator ◄──► Miner    │      │
│  │  Qdrant ──► Vector Store │    │  RAGSynapse  SyncSynapse  │      │
│  │  Chat   ──► Cited Answer │    │  4-Factor scoring (both)  │      │
│  │                          │    │                           │      │
│  │  [Private Pool]          │    │                           │      │
│  │  [Public Pool] ──────────────►│  KnowledgeSyncSynapse    │      │
│  └──────────────────────────┘    └──────────────────────────┘      │
│                │                             │                      │
│                ▼                             ▼                      │
│  ┌──────────────────────────┐    ┌──────────────────────────┐      │
│  │   THREE-LAYER SYNC       │    │  INTELLIGENCE ROUTING    │      │
│  │  LAN  · BT  · I2P        │    │  CrossSubnetRouter       │      │
│  │  TransportLayer ABC      │    │  coding→SN64 Chutes      │      │
│  │  delta_sync() protocol   │    │  knowledge→SN442 home    │      │
│  └──────────────────────────┘    └──────────────────────────┘      │
└────────────────────────────────────────────────────────────────────┘
```

---

## Key Paths

| Path | Purpose |
|------|---------|
| `neurons/miner.py` | Bittensor miner (Ollama + extractive RAG + sync queue) |
| `neurons/validator.py` | Bittensor validator (scoring + sync forward) |
| `superbrain/protocol.py` | RAGSynapse + KnowledgeSyncSynapse definitions |
| `superbrain/validator/reward.py` | RAG 4-factor scoring V2 |
| `superbrain/validator/embeddings.py` | 3-tier embedding (Ollama→TF-IDF→word overlap) + FAISS |
| `superbrain/validator/forward.py` | RAG forward pass (dynamic KB from SyncQueue) |
| `superbrain/validator/sync_forward.py` | Sync forward pass + Ed25519 enforcement |
| `superbrain/routing/cross_subnet_router.py` | CrossSubnetRouter — Phase 2 intelligence routing |
| `superbrain/routing/__init__.py` | Exports: CrossSubnetRouter, QueryType, RouteDecision |
| `sync/protocol/pool_model.py` | KnowledgeChunk, SyncManifest, Ed25519, compute_content_hash |
| `sync/protocol/delta_sync.py` | TransportLayer ABC + run_sync() 5-step protocol |
| `sync/protocol/batch.py` | Store-and-forward batch (zlib + Ed25519 signed) |
| `sync/queue/sync_queue.py` | SQLite WAL-mode queue (thread-safe, check_same_thread=False) |
| `sync/query/network_rag.py` | Network-wide RAG search against public pool |
| `sync/ingestion/document_ingestor.py` | PDF/markdown/text → KnowledgeChunks |
| `sync/lan/` | LAN transport (WebSocket + mDNS) — proven E2E |
| `sync/bluetooth/` | Bluetooth transport (RFCOMM) — mock-tested only |
| `sync/i2p/` | I2P transport (SAM v3.1) — mock-tested only |
| `scripts/ingest.py` | CLI: ingest files into SyncQueue |
| `scripts/query.py` | CLI: query the network knowledge pool |
| `scripts/network_query_ipc.py` | IPC bridge: desktop → network RAG (stdin/stdout JSON) |
| `scripts/share_to_network.py` | IPC bridge: desktop → miner SyncQueue (subprocess) |
| `scripts/start_miner.sh` | PM2 entrypoint: kills port 8384, starts miner |
| `run_sync_node.py` | Standalone sync node (seed peer) |
| `ecosystem.config.js` | PM2 process config (miner + validator + sync-node) |
| `data/sync_queue.db` | SQLite WAL database (gitignored) |
| `test_routing.py` | Phase 2 routing layer test (46 assertions) |
| `tests/` | All test suites (see Test Suite section below) |

---

## Wallet Configuration

| Role | Wallet Name | Hotkey | Network |
|------|-------------|--------|---------|
| Miner | `sb_miner` | `default` | testnet (SN442) |
| Validator | `sb_validator` | `default` | testnet (SN442) |
| Mainnet miner | UID 190 on SN65 | — | mainnet (superbrain-pool repo, private) |

Wallets are stored in `~/.bittensor/wallets/`.

---

## PM2 Process Setup

```bash
# Status
pm2 status

# Start all (first time or after reboot)
pm2 start ecosystem.config.js

# Save process list across reboots
pm2 save
pm2 startup   # generates systemd unit (run the printed command as root)

# Logs
pm2 logs superbrain-miner
pm2 logs superbrain-validator
pm2 logs superbrain-sync-node
```

### ecosystem.config.js summary

| PM2 name | Script | Args |
|----------|--------|------|
| `superbrain-miner` | `scripts/start_miner.sh` | kills port 8384, starts miner wallet=sb_miner |
| `superbrain-validator` | `neurons/validator.py` | netuid=442 network=test wallet=sb_validator |
| `superbrain-sync-node` | `run_sync_node.py` | --seed --db data/sync_queue.db --port 8385 |

**Python binary:** `venv/bin/python` (venv at `/root/superbrain-subnet/venv/`)
**PYTHONPATH:** `/root/superbrain-subnet`

### Known fixes applied to PM2 config
- **SQLite threading fix:** `SyncQueue.__init__` now uses `check_same_thread=False` + `threading.Lock()` — fixes `ProgrammingError: SQLite objects created in a thread can only be used in that same thread` when PM2 forks the process.
- **Port conflict fix:** `scripts/start_miner.sh` runs `fuser -k 8384/tcp` before starting — prevents `OSError: [Errno 98] Address already in use` on PM2 restart.

---

## Bittensor Network Config

| Setting | Value |
|---------|-------|
| Testnet netuid | 442 |
| Network | `test` |
| Subtensor endpoint | default (bittensor test endpoint) |
| Mainnet netuid | 65 (SN65, mining pool only — separate repo) |

---

## Phase 2 — Intelligence Routing Layer

`superbrain/routing/cross_subnet_router.py`

```python
from superbrain.routing import CrossSubnetRouter, QueryType, RouteDecision

router = CrossSubnetRouter()
decision = router.route("Write a Python function to sort a list")
# → RouteDecision(type=coding, → SN64 [Chutes], conf=0.78)
```

### Routing table

| QueryType | Target Subnet | Rationale |
|-----------|---------------|-----------|
| CODING | SN64 Chutes | Code execution + coding assistance |
| MATH | SN64 Chutes | Numerical computation |
| KNOWLEDGE | SN442 SuperBrain | Local knowledge retrieval (home) |
| GENERAL | SN442 SuperBrain | Default home fallback |
| IMAGE | SN18 Cortex | General AI inference |

### Key behaviours
- `prefer_home=True` + confidence < 0.6 → always routes to SN442
- Fallback on every non-home route = SN442
- `route_batch(queries)` → `List[RouteDecision]`
- No API keys required for classification (pure regex scoring)
- OpenAI-compatible REST endpoints reserved per SubnetSpec (empty = use metagraph)

### Classification patterns
- **CODING:** Python/JS/TS/Go/Rust/Java keywords, git/docker/api, code block markers, write-a-function phrases
- **MATH:** solve/equation/integral/derivative/arithmetic expressions
- **KNOWLEDGE:** what is/who is/who invented/explain/define/history/research
- **IMAGE:** image/photo/render/stable diffusion/vision/OCR

---

## RAG Scoring (V2)

```
RAG Score = 0.40 × Supportedness + 0.25 × Relevance + 0.20 × Novelty + 0.15 × Latency
```

V2 enhancements: embedding cosine similarity (Ollama→TF-IDF→word overlap), FAISS vector index, length penalty, citation quality bonus.

---

## Sync Scoring

```
Sync Score = 0.35 × Validity + 0.25 × Freshness + 0.25 × Quantity + 0.15 × Latency
```

---

## Two-Pool Privacy Model

Every `KnowledgeChunk` has `pool_visibility`: `private` (default) or `public`.
- **Private pool:** local only, no telemetry, your eyes only
- **Public pool:** Ed25519 signed, content-addressed, synced to peers, TAO-incentivized
- Revocation stops future distribution; already-synced copies persist (distributed system semantics)

---

## Test Suite

```
tests/unit/test_knowledge_sync.py     ~159 assertions — synapse + sync scoring
tests/unit/test_scoring_v2.py          56 assertions — embedding similarity
tests/unit/test_pool_model.py          58 assertions — data model + crypto
tests/unit/test_sync_queue.py          44 assertions — SQLite queue
tests/unit/test_batch.py               37 assertions — batch format
tests/unit/test_delta_sync.py          49 assertions — sync protocol
tests/unit/test_lan_sync.py            54 assertions — LAN transport
tests/unit/test_bluetooth_sync.py      55 assertions — Bluetooth transport
tests/unit/test_i2p_sync.py           100 assertions — I2P transport
tests/unit/test_document_ingestor.py  ~150 assertions — PDF/text ingestion
tests/unit/test_dynamic_forward.py    ~120 assertions — dynamic KB validator
tests/unit/test_network_query_ipc.py  ~100 assertions — IPC bridge
tests/unit/test_network_rag.py        ~165 assertions — network RAG search
tests/test_integration.py              38 assertions — full LAN→miner→validator loop
test_routing.py                        46 assertions — Phase 2 routing layer
─────────────────────────────────────────────────────
TOTAL                                ~1,200+ assertions
```

Run all unit tests:
```bash
for f in tests/unit/test_*.py; do venv/bin/python "$f" 2>&1 | tail -3; done
venv/bin/python test_routing.py
venv/bin/python tests/test_integration.py
```

---

## Known Limitations (Honest)

- **Hardcoded validator KB:** 5 static queries on testnet — allows caching by gaming miners. Production needs rotating ground-truth datasets.
- **Mock-tested transports:** Bluetooth (RFCOMM) and I2P (SAM v3.1) fully implemented but NOT tested on real hardware. LAN is the only proven-E2E transport.
- **Ed25519 not enforced in sync scoring:** Signatures implemented and attached, but validator sync path checks hash integrity only — `verify()` not called on incoming chunks.
- **Desktop UI not wired:** React components (ShareToPoolButton, PoolOverview, SyncStatusBadge) render with demo data. `share_to_network.py` IPC bridge is ready but Electron app not yet wired to it.
- **Single-machine demo:** Multi-node testing across real networks not yet done.
- **Revocation is local-only:** Marking a chunk private stops future sync but cannot recall copies already distributed.

---

## Related Repos (all private)

| Repo | Purpose |
|------|---------|
| `KatchDaVizion/superbrain-pool` | SN65 mainnet mining pool (UID 190, WireGuard, 253 peers) |
| `KatchDaVizion/superbrain-desktop-work` | Electron + React desktop app |
| `KatchDaVizion/superbrain-dev` | Dev sandbox |

---

## Current Phase Status (2026-03-25)

| Phase | Status |
|-------|--------|
| Phase 1 — Core subnet (RAG + Sync synapses, 3 transports, scoring) | Complete |
| Phase 1 — Production hardening (Ed25519, FAISS, SQLite threading, port fix) | Complete |
| Phase 1 — DevOps (venv, DB init, PM2 ecosystem.config.js, start_miner.sh) | Complete |
| Phase 2 — Intelligence Routing Layer (CrossSubnetRouter, 46 assertions) | Complete |
| Phase 3 — Desktop UI wiring (share_to_network.py bridge ready, Electron pending) | Next |
| Phase 3 — Real hardware testing (Bluetooth + I2P) | Pending |
| Phase 3 — Multi-node testnet (real multi-machine) | Pending |

---

## Quick Commands

```bash
# Check everything is running
pm2 status

# Restart a process
pm2 restart superbrain-miner

# Run routing tests
venv/bin/python test_routing.py

# Ingest a document
venv/bin/python scripts/ingest.py --file /path/to/doc.pdf --public

# Query the network pool
venv/bin/python scripts/query.py "What is yuma consensus?"

# Check sync queue
venv/bin/python -c "
from sync.queue.sync_queue import SyncQueue
q = SyncQueue('data/sync_queue.db')
print(q.stats())
"
```
