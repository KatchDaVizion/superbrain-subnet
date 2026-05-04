# SuperBrain I2P Bootstrap Peers

Connect to the SuperBrain knowledge mesh over I2P — censorship-resistant, anonymous, always-on.

> **Prerequisite:** I2P mesh sync requires a local i2pd router with SAM API enabled.
> ```bash
> apt install i2pd && systemctl enable --now i2pd
> ```
> If SAM is unreachable (port 7656), the node starts anyway with LAN sync only and logs install instructions.

## Quick start

```bash
# Clone and run (auto-connects to both bootstrap nodes)
git clone https://github.com/KatchDaVizion/superbrain-subnet
cd superbrain-subnet
pip install -r requirements.txt
python3 run_sync_node.py --seed
```

Both bootstrap nodes are built into `run_sync_node.py`. Your node will:
1. Generate a **persistent I2P identity** (saved to `data/i2p-sync.keys` — same address across restarts)
2. Sync knowledge chunks with Frankfurt + Helsinki over I2P
3. Accept inbound connections from any other node that knows your address

## Bootstrap nodes

| Node | Location | I2P Destination (first 32 chars) |
|---|---|---|
| Frankfurt | 46.225.114.202 | `B9gQNaaB6-uXkWW8FH56EHixv8zXUId...` |
| Helsinki | 89.167.61.241 | `diGWuHmkh0W56Oauw2n7TAVFZh1AnvV...` |

Full base64 destinations are in `run_sync_node.py` as `FRANKFURT_I2P_DEST` and `HELSINKI_I2P_DEST`.

## Adding your node as a peer

After your node starts, the log will show:

```
I2P FULL DEST: <your-full-base64-dest>
```

Share this with other node operators or open a PR to add it to this file.

## Advanced: add extra I2P peers

```bash
python3 run_sync_node.py --seed --i2p-peers "base64dest:nodename,base64dest2:nodename2"
```

## LAN-only mode (no I2P)

```bash
python3 run_sync_node.py --seed --no-i2p
```

## Requirements

- Python 3.10+
- i2pd running locally with SAM API enabled on port 7656
  - Ubuntu/Debian: `apt install i2pd && systemctl enable --now i2pd`
  - i2pd.conf: ensure `[sam]` section has `enabled = true`
