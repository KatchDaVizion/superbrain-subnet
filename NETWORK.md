# SuperBrain — SN442 Network Guide

SuperBrain is a decentralized knowledge network on Bittensor subnet 442.
Every participating node is a **mini server**: it stores knowledge locally, runs
its own language model (Ollama), answers queries, and earns TAO through Yuma
consensus scoring.

## Quick links

- **Seed node:** http://46.225.114.202:8400
- **Live feed:** http://46.225.114.202:8400/feed.html
- **Bootstrap:** http://46.225.114.202:8400/bootstrap
- **Health:** http://46.225.114.202:8400/health

## How SN442 works

### The chain
- Bittensor testnet netuid **442**.
- **Validator** (UID 1 on Frankfurt) queries miners with RAG prompts every ~5s,
  scores responses via a 4-factor mechanism (Supportedness, Relevance, Novelty,
  Latency), EMA over ~20 rounds, and writes weights to chain.
- **Miners** (UID 2 + new joiners) serve `RAGSynapse` and `KnowledgeSyncSynapse`
  over their Bittensor axon. Each miner holds a local SQLite knowledge DB.
- **Yuma consensus** distributes TAO emissions per-tempo (360 blocks ≈ 72 min)
  based on the validator-assigned weights.

### The knowledge layer
- **Chunks** are stored in SQLite (`sync_queue.db`). Fields include
  `content_hash`, `content`, `contributor_hotkey`, `shared_at`, `metadata`,
  `retrieval_count`. WAL mode, `isolation_level=None`.
- **Sharing** goes through the seed's `POST /knowledge/share` endpoint, which
  writes to the chunk DB and triggers validator scoring.
- **Retrieval** happens implicitly when the validator picks random chunks for
  its forward-pass queries. Each retrieval bumps `retrieval_count` and
  eventually your EMA score.

### Peer discovery
- Frankfurt is **bootstrap-only**. Nodes call `/bootstrap` once on first boot,
  cache the seed + current peer list locally, then sync peer-to-peer.
- LAN sync protocol lives on each miner's `--lan-port` (default 8384).
- Hyperswarm DHT is also live on `46.225.114.202:49737` for the desktop app's
  direct mesh layer (topic `sha256("superbrain-sn442-v1")`).
- I2P SAM bridge is live on `127.0.0.1:7656` (tested on the seed; `i2p_active`
  now reflects real handshake state in `/peers` and `/bootstrap` responses).

## How to become a miner

### 1. Create a wallet
```bash
pip3 install bittensor bittensor-cli --break-system-packages
btcli wallet new-coldkey --wallet-name my_miner --wallet-path /root/.bittensor/wallets/ --n-words 12 --no-use-password
btcli wallet new-hotkey  --wallet-name my_miner --hotkey default --wallet-path /root/.bittensor/wallets/ --n-words 12 --no-use-password
btcli wallet list
```

### 2. Fund the coldkey
Testnet registration is cheap (currently `τ 0.0005` recycle). Get testnet TAO
from the Bittensor Discord `#testnet` faucet, or from an existing holder.

### 3. Register on SN442
```bash
btcli subnet register --netuid 442 --wallet-name my_miner --wallet-hotkey default --subtensor.network test
```

### 4. Clone this repo + install deps
```bash
git clone <this-repo-url> /root/superbrain-subnet
cd /root/superbrain-subnet
pip3 install -r requirements.txt --break-system-packages
```

### 5. Install Ollama (local inference — never external APIs)
```bash
curl -fsSL https://ollama.com/install.sh | sh
# Bind to localhost only for security
mkdir -p /etc/systemd/system/ollama.service.d
cat > /etc/systemd/system/ollama.service.d/override.conf <<EOF
[Service]
Environment="OLLAMA_HOST=127.0.0.1:11434"
Environment="OLLAMA_KEEP_ALIVE=-1"
EOF
systemctl daemon-reload && systemctl restart ollama
ollama pull qwen2.5:0.5b
ollama pull nomic-embed-text
```

### 6. Start the miner
```bash
python3 neurons/miner.py \
  --netuid 442 \
  --subtensor.network test \
  --wallet.name my_miner \
  --wallet.hotkey default \
  --axon.port 8091 \
  --axon.external_ip $(curl -s ifconfig.me) \
  --lan-sync --lan-port 8384 \
  --lan-static 46.225.114.202:8384 \
  --logging.debug
```

The miner will:
1. Register its axon IP on chain (one-time, rate-limited).
2. Open port 8091 to accept RAGSynapse queries from validators.
3. Gossip chunks with Frankfurt via LAN-sync over port 8384.
4. Log progress every 5s: `Miner running... chunks=N lan_peers=M`.

## How to run a validator

You need **1,024 TAO stake** minimum for validator permit (vpermit) on mainnet.
On testnet the current limit is set high (`vpermit_tao_limit 1000000`) for
development.

```bash
btcli subnet register --netuid 442 --wallet-name my_validator --wallet-hotkey default --subtensor.network test
python3 neurons/validator.py \
  --netuid 442 \
  --subtensor.network test \
  --wallet.name my_validator \
  --wallet.hotkey default \
  --axon.port 8092 \
  --axon.external_ip $(curl -s ifconfig.me) \
  --neuron.vpermit_tao_limit 1000000 \
  --logging.debug
```

**Operational note:** the validator's scoring work happens in a background
thread. If the Bittensor testnet RPC returns HTTP 503 during a forward pass,
the thread can die silently while the PM2 keepalive loop continues logging
`Validator running...`. Verify live scoring by tailing logs for the
`step(N) block(M)` cadence, not the keepalive line.

## How to contribute knowledge via SDK

The SuperBrain SDK exposes a 4-function Node interface:

```bash
npm install superbrain-sdk
```

```typescript
import { query, share, earnings, peers } from 'superbrain-sdk'

// Ask the network a question
const answer = await query("How does Yuma consensus work?")

// Share validated knowledge with your hotkey
const chunkId = await share(
  "Yuma consensus is ... (validated content)",
  "5DABRZwS4tmJnbjkoKFi7PqAamSdCdztwciw7QFwbRwgWkJX"
)

// Check your earnings
const e = await earnings("5DABRZwS4tmJnbjkoKFi7PqAamSdCdztwciw7QFwbRwgWkJX")
// e.total_chunks, e.total_retrievals, e.estimated_tao
```

All calls default to `http://46.225.114.202:8400` and can be overridden via
the `SB_API_URL` env var.

## Privacy & data-flow rules

1. **Local inference only.** No node in this network calls external AI APIs.
   Every prompt is answered by a local Ollama instance.
2. **Frankfurt is bootstrap-only.** Desktop clients discover peers via
   `/bootstrap`, then sync directly from each other — not back through Frankfurt.
3. **Wallets are private.** Mnemonics never leave the machine where they were
   generated. Public ss58 addresses are the only wallet data shared.
4. **I2P optional.** When `i2p_active` is true (seed runs i2pd with SAM bridge),
   peers can opt into I2P tunnels for traffic anonymization. Default remains
   clearnet for performance.

## Live state (updated 2026-04-17)

| Component | State |
|---|---|
| Frankfurt seed | UID 1 validator + UID 2 miner, step(236+), 5/5 PM2 online |
| Helsinki miner | Wallet created, repo deployed, Ollama installed, **not yet registered** |
| Total chunks | 130+ (growing ~15/run via mega_agent) |
| i2p_active | true (SAM handshake verified) |
| Knowledge feed | http://46.225.114.202:8400/feed.html |

## Repo map

- `neurons/miner.py` — Bittensor miner neuron (416 LOC)
- `neurons/validator.py` — Bittensor validator neuron (71 LOC + `BaseValidatorNeuron`)
- `superbrain/base/` — base neuron / miner / validator classes
- `superbrain/validator/forward.py` — 4-factor scoring
- `superbrain/validator/sync_forward.py` — periodic chunk sync
- `sync/lan/lan_sync.py` — LAN peer-to-peer chunk sync
- `sync/queue/sync_queue.py` — SQLite chunk queue
- `run_sync_node.py` — standalone sync-node seed
- `scripts/start_miner.sh` / `start_miner_helsinki.sh` — PM2/systemd-friendly starters
- `scripts/deploy_testnet.sh` — end-to-end testnet bootstrap

## Contact

For onboarding help, check the `/bootstrap` endpoint — it returns the current
peer list. Ping any online miner with a `RAGSynapse` query to verify the
network is answering.
