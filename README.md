# SuperBrain Subnet (SN442)

> Decentralized, privacy-first AI knowledge synchronization on Bittensor.

SuperBrain is a Bittensor subnet (**SN442**) where every contributor's laptop becomes a knowledge node. Users share validated knowledge chunks, the network scores them via the Yuma consensus, and retrievals earn TAO. The full stack runs locally — your memory never leaves your machine unless you choose to share it.

This repository contains the **subnet protocol code**: LAN sync server/client, demo API, and intelligence routing layer.

The **desktop app** lives at [`superbrain-desktop-work`](https://github.com/KatchDaVizion/superbrain-desktop-work).

The **SDK** ships with the desktop monorepo under `sdk/` (4-function Node client for `query`, `share`, `earnings`, `peers`).

---

## What's in this repo

```
superbrain-subnet/
├── superbrain/              # LAN-sync subprotocol (Ed25519-signed gossip)
│   └── sync/
│       ├── lan_sync_server.py
│       └── lan_sync_client.py
├── superbrain-demo/         # Frankfurt seed node API + dashboard
│   ├── api.py               # FastAPI service (port 8400)
│   ├── start.sh
│   ├── deploy_live.sh
│   └── static/              # Dashboard UI
└── superbrain-routing/      # Cross-subnet intelligence routing
    ├── superbrain/routing/  # SN64 / SN18 / SN442 fallback chain
    └── test_routing.py
```

---

## Architecture (5-layer stack)

```
Layer 0  MemPalace cross-session memory     (ChromaDB-backed verbatim drawers)
Layer 1  ZIM offline encyclopedia           (Wikipedia via kiwix-serve)
Layer 2  Qdrant document store              (hybrid BM25 + vector RAG)
Layer 3  SN442 Bittensor network            (this repo)
Layer 4  Ollama local inference             (qwen2.5:0.5b, no GPU required)
```

Each layer is gracefully optional. If any layer fails, the others fill in. The user owns all data.

---

## Subnet status

| | |
|---|---|
| **Subnet** | SN442 (testnet) |
| **Validator step** | 33,000+ |
| **Validator EMA** | 0.575 |
| **Knowledge pool** | 50+ chunks |
| **Scoring interval** | ~12s |
| **Miner UID 1 incentive** | 1.0000 |
| **Seed node** | Frankfurt, Germany |
| **License** | MIT |

Mainnet registration is the next milestone (682–1,027 TAO required).

---

## Quick start &mdash; query the live testnet

```bash
# From any Node 18+ environment
cat > test.js <<'EOF'
const sb = require('superbrain-sdk') // or relative path to ../sdk/index.js
sb.query('What is SuperBrain?').then(r => console.log(r.answer))
EOF
node test.js
```

Or with `curl`:

```bash
curl -X POST http://46.225.114.202:8400/query \
  -H "Content-Type: application/json" \
  -d '{"question":"What is SuperBrain?"}'
```

---

## Components

### LAN Sync (Ed25519-signed gossip)

Located at `superbrain/sync/`. Production-ready peer-to-peer chunk gossip with cryptographic signatures.

- **Server:** `lan_sync_server.py` &mdash; listens on configurable port, accepts signed chunks, broadcasts to peers
- **Client:** `lan_sync_client.py` &mdash; connects to peers, fetches chunks, verifies signatures

Generates Ed25519 keypairs locally. **Wallets and private keys never leave your machine.**

### Demo API (Frankfurt seed)

Located at `superbrain-demo/`. The FastAPI service that runs on the Frankfurt seed node and serves the demo dashboard.

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Health check |
| `/query` | POST | Ask the network a question |
| `/knowledge/share` | POST | File a chunk to the network |
| `/knowledge/list` | GET | List all chunks in the pool |
| `/peers` | GET | Registered peer registry |
| `/announce` | POST | Register as a peer |
| `/earnings/{hotkey}` | GET | Retrieval-based earnings for a hotkey |
| `/network-map` | GET | Live network topology |

### Intelligence Routing

Located at `superbrain-routing/`. Cross-subnet query router. Detects intent and routes to the best subnet:

- Coding questions &rarr; SN64 Chutes
- Vision/image &rarr; SN18 Cortex
- General knowledge &rarr; SN442 (this subnet)

Falls back through a 3-tier chain. API keys for SN64/SN18 are read from env vars (`CHUTES_API_KEY`, `TARGON_API_KEY`) &mdash; **never committed.**

---

## Running a node

Detailed instructions live in `superbrain-demo/README.md`. The TL;DR:

```bash
cd superbrain-demo
./start.sh
```

This launches the FastAPI service on port 8400. You'll need Python 3.11+, FastAPI, uvicorn, and the Bittensor SDK installed in a venv.

---

## Security disclosure

This repository is the **subnet code** &mdash; no secrets, no wallets, no API keys. If you find one, please open an issue immediately.

- All `.env*` files are gitignored except `.env.example`
- Bittensor wallets are gitignored
- Ed25519 keys are generated locally per node
- The Frankfurt seed node IP is public information

---

## Contributing

This is a solo-built project moving fast. Contributions welcome via PR. Open an issue first for non-trivial changes.

**License:** MIT &mdash; see `LICENSE`

---

## Links

- **Desktop app:** github.com/KatchDaVizion/superbrain-desktop-work
- **SuperBrain on TaoStats:** taostats.io/subnets/442
- **Founder:** [@KatchDaVizion](https://github.com/KatchDaVizion) &mdash; David Louis-Charles, 17779011 Canada Inc.
