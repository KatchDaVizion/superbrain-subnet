# SuperBrain — Local-First Anonymous Knowledge Network on Bittensor

**Working Prototype | Bittensor Subnet Ideathon 2026 | by Lys-David Louis-Charles (KatchDaVizion)**

SuperBrain is a working prototype for a decentralized knowledge network where your data stays private by default and public by choice. Users run local AI (Ollama + Qdrant) for private RAG, then optionally share knowledge chunks to a network-wide pool — incentivized by Bittensor's TAO mining rewards. Three sync transports (LAN, Bluetooth, I2P) keep knowledge alive even without internet.

This is not a finished product — it's a functional proof-of-concept with running code on Bittensor testnet, 677 passing test assertions, and a live 4-process demo. The architecture works. What remains is hardening, real-world testing, and production deployment.

> *"Private by default. Public by choice. Wherever one SuperBrain remains, knowledge shall survive."*

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        SUPERBRAIN ECOSYSTEM                          │
│                                                                      │
│  ┌──────────────────────────┐    ┌──────────────────────────┐       │
│  │     DESKTOP APP          │    │     BITTENSOR SUBNET      │       │
│  │  (Electron + React)      │    │     (SN442 Testnet)       │       │
│  │                          │    │                            │       │
│  │  Ollama ──► Local RAG    │    │  Validator ◄──► Miner     │       │
│  │  Qdrant ──► Vector Store │    │     │              │      │       │
│  │  Chat   ──► Cited Answer │    │  RAGSynapse   SyncSynapse │       │
│  │                          │    │     │              │      │       │
│  │  [Private Pool]          │    │  4-Factor      4-Factor   │       │
│  │  [Public Pool] ──────────────►│  Scoring       Scoring    │       │
│  └──────────────────────────┘    └──────────────────────────┘       │
│                │                                                     │
│                ▼                                                     │
│  ┌──────────────────────────────────────────────────────────┐       │
│  │           THREE-LAYER SYNC PROTOCOL                       │       │
│  │                                                            │       │
│  │  Layer 1: LAN Sync     (mDNS + WebSocket)                │       │
│  │  Layer 2: Bluetooth     (RFCOMM + 4-byte framing)         │       │
│  │  Layer 3: I2P Anonymous (SAM v3.1 + framing)              │       │
│  │                                                            │       │
│  │  All share: TransportLayer ABC + delta_sync()             │       │
│  └──────────────────────────────────────────────────────────┘       │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────┐       │
│  │              MINING POOL (SN65)                            │       │
│  │  FastAPI + PostgreSQL + WireGuard + Docker                 │       │
│  │  UID 190 • Auto-healing • 3 systemd watchdogs             │       │
│  └──────────────────────────────────────────────────────────┘       │
└──────────────────────────────────────────────────────────────────────┘
```

## Prototype Status

| Feature | Status | Details |
|---------|--------|---------|
| RAG Synapse (query → cited answer) | Live on testnet | SN442, miner + validator running |
| 4-Factor RAG Scoring | Complete | Supportedness/Relevance/Novelty/Latency |
| V2 Embedding Scoring | Complete | Ollama → TF-IDF → word overlap (3-tier fallback) |
| KnowledgeSyncSynapse | Complete | Bittensor-native knowledge chunk exchange |
| Sync Scoring Engine | Complete | Validity/Freshness/Quantity/Latency |
| Two-Pool Privacy Model | Complete | Private by default, public by choice |
| Delta Sync Protocol | Complete | Transport-agnostic 5-step handshake |
| LAN Sync | Complete | mDNS discovery + WebSocket transport (proven end-to-end) |
| Bluetooth Sync | Protocol complete (mock-tested) | RFCOMM + 4-byte framing — not tested on real BT hardware |
| I2P Anonymous Sync | Protocol complete (mock-tested) | SAM v3.1 streams — not tested on live I2P router |
| Store-and-Forward Batch | Complete | zlib compressed, Ed25519 signed |
| Desktop UI (Share to Pool) | UI components ready (demo data) | React + shadcn/ui — not wired to sync backend |
| Mining Pool (SN65 mainnet) | Live | UID 190, auto-healing, 253 WireGuard peers |
| **Test Suite** | **677 assertions** | **11 test files, all passing** |

## Two-Pool Privacy Model

```
┌─────────────────────┐          ┌─────────────────────┐
│   PRIVATE POOL       │  Share   │   PUBLIC POOL        │
│                     │ ──────► │                     │
│ Your personal       │          │ Shared to network   │
│ knowledge base      │ ◄────── │ Incentivized by TAO │
│                     │  Revoke  │                     │
│ • Local only        │          │ • Synced to peers   │
│ • No telemetry      │          │ • Ed25519 signed    │
│ • Your eyes only    │          │ • Content-addressed  │
└─────────────────────┘          └─────────────────────┘
```

Every knowledge chunk has a `pool_visibility` field: `private` (default) or `public`. Users explicitly choose what to share. Revocation stops future sync distribution; copies already on remote nodes persist (standard distributed system behavior — you can't un-send data).

## Three-Layer Sync Protocol

All three transports share the same `TransportLayer` ABC and `delta_sync()` protocol:

| Layer | Transport | Discovery | Status |
|-------|-----------|-----------|--------|
| **LAN** | WebSocket | mDNS/zeroconf | Proven end-to-end (real WebSocket, real mDNS) |
| **Bluetooth** | RFCOMM | PyBluez scan | Protocol complete, mock-tested (needs real BT hardware) |
| **I2P** | SAM v3.1 streams | Address book | Protocol complete, mock-tested (needs live I2P router) |

**Delta sync flow:** Handshake → Manifest exchange → Diff computation → Chunk transfer → Ingestion

## RAG Scoring Engine

```
RAG Score = (0.40 × Supportedness) + (0.25 × Relevance) + (0.20 × Novelty) + (0.15 × Latency)
```

| Factor | Weight | V2 Enhancement |
|--------|--------|----------------|
| **Supportedness** | 40% | Embedding cosine similarity + citation quality sub-factor |
| **Relevance** | 25% | Mean embedding similarity per cited chunk |
| **Novelty** | 20% | SHA-256 dedup + word-overlap paraphrase detection |
| **Latency** | 15% | Linear: `1 - (time / max_time)` |

V2 adds: embedding-based scoring (Ollama → TF-IDF → word overlap fallback chain), length penalty multiplier, citation quality bonus.

## KnowledgeSyncSynapse

A second Bittensor synapse for knowledge chunk exchange:

```
Validator                              Miner
   │                                     │
   │  KnowledgeSyncSynapse               │
   │  known_hashes: [h1, h2, ...]        │
   │  max_chunks: 50                     │
   │ ──────────────────────────────────► │
   │                                     │  Filter queue, exclude known
   │                 batch_data (base64) │  Compress + encode
   │                 chunk_count: 12     │
   │ ◄────────────────────────────────── │
   │                                     │
   │  Validate hashes, ingest, score     │
```

**Sync Scoring:** `0.35 × Validity + 0.25 × Freshness + 0.25 × Quantity + 0.15 × Latency`

## Quick Start

### Install
```bash
git clone https://github.com/KatchDaVizion/superbrain-subnet.git
cd superbrain-subnet
python -m pip install -e .
```

### Run Tests
```bash
# Full integration test (LAN sync → miner queue → validator)
python tests/test_integration.py

# RAG scoring tests
python tests/test_scoring.py

# All unit tests
for f in tests/unit/test_*.py; do python3 -u "$f" 2>&1 | tail -3; done
```

### Run with Mock Network
```bash
# Terminal 1 — Miner
python neurons/miner.py --netuid 1 --mock --wallet.name miner --wallet.hotkey default

# Terminal 2 — Validator
python neurons/validator.py --netuid 1 --mock --wallet.name validator --wallet.hotkey default
```

### Deploy to Testnet
```bash
# Get test TAO: join Bittensor Discord → #testnet channel
chmod +x scripts/deploy_testnet.sh
./scripts/deploy_testnet.sh
```

## Project Structure

```
superbrain-subnet/
├── superbrain/                          # Main package
│   ├── protocol.py                      # RAGSynapse + KnowledgeSyncSynapse
│   ├── mock.py                          # MockSubtensor/Dendrite for testing
│   ├── base/                            # Base neuron classes (from template)
│   │   ├── miner.py, validator.py, neuron.py
│   │   └── utils/weight_utils.py
│   ├── utils/                           # Shared utilities
│   │   ├── config.py, misc.py, uids.py, logging.py
│   └── validator/                       # Scoring engines
│       ├── reward.py                    # RAG 4-factor scoring (V2)
│       ├── embeddings.py                # 3-tier embedding fallback
│       ├── sync_reward.py               # Sync 4-factor scoring
│       ├── sync_forward.py              # Validator sync forward pass
│       └── forward.py                   # RAG forward pass
├── neurons/
│   ├── miner.py                         # Miner (Ollama + extractive + sync)
│   └── validator.py                     # Validator (RAG scoring + sync)
├── sync/                                # Three-layer sync protocol
│   ├── protocol/                        # Transport-agnostic core
│   │   ├── delta_sync.py                # TransportLayer ABC + run_sync()
│   │   ├── pool_model.py                # KnowledgeChunk, SyncManifest, Ed25519
│   │   └── batch.py                     # Store-and-forward batch format
│   ├── queue/
│   │   └── sync_queue.py                # SQLite WAL-mode queue
│   ├── lan/                             # LAN transport (WebSocket + mDNS)
│   ├── bluetooth/                       # Bluetooth transport (RFCOMM)
│   └── i2p/                             # I2P transport (SAM v3.1)
├── tests/
│   ├── unit/                            # 639 unit test assertions across 10 files
│   │   ├── test_knowledge_sync.py       # 159 assertions — synapse + sync scoring
│   │   ├── test_scoring_v2.py           # 56 assertions — embedding scoring
│   │   ├── test_pool_model.py           # 58 assertions — data model + crypto
│   │   ├── test_sync_queue.py           # 44 assertions — SQLite queue
│   │   ├── test_batch.py                # 37 assertions — batch format
│   │   ├── test_delta_sync.py           # 49 assertions — sync protocol
│   │   ├── test_lan_sync.py             # 54 assertions — LAN transport
│   │   ├── test_bluetooth_sync.py       # 55 assertions — Bluetooth transport
│   │   └── test_i2p_sync.py            # 100 assertions — I2P transport
│   ├── test_integration.py              # 38 assertions — full-loop: LAN→miner→validator
│   └── mock/                            # Mock transport + socket layers
├── scripts/deploy_testnet.sh
├── setup.py, requirements.txt
└── LICENSE (MIT)
```

## Test Suite — 677 Assertions (11 Files, All Passing)

```
test_scoring.py           27 assertions — RAG scoring pipeline, anti-gaming, edge cases
test_pool_model.py        58 assertions — KnowledgeChunk, SyncManifest, crypto, diff
test_sync_queue.py        44 assertions — SQLite queue operations, persistence
test_scoring_v2.py        56 assertions — embedding similarity, 3-tier fallback
test_batch.py             37 assertions — batch create/extract, compression, signatures
test_delta_sync.py        49 assertions — wire format, handshake, manifest exchange
test_lan_sync.py          54 assertions — WebSocket transport, mDNS discovery
test_bluetooth_sync.py    55 assertions — RFCOMM framing, mock sockets, manager
test_i2p_sync.py         100 assertions — SAM protocol, mock bridge, full lifecycle
test_knowledge_sync.py   159 assertions — synapse, scoring, miner/validator flow, E2E
test_integration.py       38 assertions — full-loop: LAN sync → miner queue → validator
─────────────────────────────────────
TOTAL                    677 assertions   All passing
```

## Anti-Gaming (Prototype)

Current measures implemented in scoring:

- **SHA-256 content-addressing** — exact duplicate detection across miners
- **Word-overlap paraphrase detection** — catches rephrased copies
- **Challenge queries (20%)** — known-answer queries with keyword verification
- **Hallucination penalty** — cross-references claims against source chunks
- **Embedding cosine similarity** — semantic-level plagiarism detection

See "Known Limitations" below for honest assessment of what these cover and what they don't.

## Known Limitations

This is a working prototype. We're transparent about what's done and what isn't:

- **Hardcoded knowledge base**: The validator uses a static set of 5 sample queries and 2 challenge queries for testnet demonstration. A production validator would maintain curated, rotating ground-truth datasets. Static queries allow answer caching by gaming miners.
- **Per-miner timing not precise**: Response times are estimated by dividing total batch time equally across miners. Production would use Bittensor's per-response `process_time` for accurate latency scoring.
- **Signature verification not enforced**: KnowledgeChunks support Ed25519 signatures, but the validator sync path checks hash integrity only — it does not yet call `verify()` on incoming chunks. Cryptographic attribution is implemented but not enforced in scoring.
- **Local-only revocation**: Making a chunk private stops future sync distribution, but copies already synced to remote nodes persist. This matches standard distributed system behavior, but users should understand sharing is effectively permanent once synced.
- **Mock-tested transports**: Bluetooth (RFCOMM) and I2P (SAM v3.1) protocols are fully implemented and tested with mock sockets/bridges. They have not been tested on real Bluetooth hardware or a live I2P router. LAN sync is the only transport proven end-to-end.
- **Desktop UI not wired**: React components (ShareToPoolButton, PoolOverview, SyncStatusBadge) render correctly with demo data but are not yet connected to the sync backend.
- **Chunk spam vector**: The quantity scoring factor could be gamed by generating high volumes of structurally valid but semantically useless chunks. Future work includes logarithmic diminishing returns and semantic deduplication.
- **Single-machine demo**: The 4-process demo runs on one machine. Multi-node testing across real networks is next.

## Existing Infrastructure

| System | Status | Link |
|--------|--------|------|
| **Testnet Subnet (SN442)** | Running | This repo |
| **Mainnet Mining Pool (SN65)** | Live, UID 190 | [superbrain-pool](https://github.com/KatchDaVizion/superbrain-pool) (private) |
| **Desktop App** | UI ready (demo data) | [superbrain-desktop-work](https://github.com/KatchDaVizion/superbrain-desktop-work) |
| **Dev Sandbox** | 677 assertions | [superbrain-dev](https://github.com/KatchDaVizion/superbrain-dev) (private) |

## Author

**Lys-David Louis-Charles** (KatchDaVizion) — Ottawa, Canada
- Currently UID 190 on Subnet 65 (TPN) mainnet
- GitHub: [@KatchDaVizion](https://github.com/KatchDaVizion)

## License

MIT — see [LICENSE](LICENSE)
