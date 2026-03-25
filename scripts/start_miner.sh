#!/bin/bash
# Kill any process holding the LAN sync port before starting
fuser -k 8384/tcp 2>/dev/null || true
exec /root/superbrain-subnet/venv/bin/python neurons/miner.py \
  --netuid 442 \
  --subtensor.network test \
  --wallet.name sb_miner \
  --wallet.hotkey default \
  --lan-sync \
  --lan-port 8384 \
  --logging.debug
