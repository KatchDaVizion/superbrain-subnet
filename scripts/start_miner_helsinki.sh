#!/bin/bash
# Helsinki miner start script — talks to Frankfurt seed sync node
fuser -k 8384/tcp 2>/dev/null || true
cd /root/superbrain-subnet
export PYTHONPATH=/root/superbrain-subnet
exec /usr/bin/python3 neurons/miner.py \
  --netuid 442 \
  --subtensor.network test \
  --wallet.name helsinki_miner \
  --wallet.hotkey default \
  --axon.port 8091 \
  --axon.external_ip 89.167.61.241 \
  --lan-sync \
  --lan-port 8384 \
  --lan-static 46.225.114.202:8385 \
  --logging.debug
