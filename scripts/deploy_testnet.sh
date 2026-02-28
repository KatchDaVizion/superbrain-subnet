#!/bin/bash
# SuperBrain Testnet Deployment — follows official docs exactly
set -e
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'

echo -e "${GREEN}══════════════════════════════════════════════${NC}"
echo -e "${GREEN}  SuperBrain Testnet Deployment${NC}"
echo -e "${GREEN}══════════════════════════════════════════════${NC}"

echo -e "\n${YELLOW}[1/7] Checking dependencies...${NC}"
python3 -c "import bittensor" 2>/dev/null || { echo -e "${RED}Install bittensor: pip install bittensor${NC}"; exit 1; }
btcli --version
echo -e "${GREEN}✅ OK${NC}"

echo -e "\n${YELLOW}[2/7] Creating wallets...${NC}"
for name in sb_owner sb_miner sb_validator; do
    if btcli wallet overview --wallet.name "$name" 2>/dev/null | grep -q "ss58"; then
        echo "  $name exists"
    else
        echo "  Creating $name..."
        btcli wallet new_coldkey --wallet.name "$name" --no_password
        btcli wallet new_hotkey --wallet.name "$name" --wallet.hotkey default
    fi
done

echo -e "\n${YELLOW}[3/7] Check owner balance...${NC}"
btcli wallet balance --wallet.name sb_owner --subtensor.network test 2>/dev/null || true
echo -e "${YELLOW}  Need ~100 test TAO. Get from Bittensor Discord #testnet${NC}"
read -p "Press Enter when funded..."

echo -e "\n${YELLOW}[4/7] Subnet lock cost...${NC}"
btcli subnet lock_cost --subtensor.network test

echo -e "\n${YELLOW}[5/7] Creating subnet...${NC}"
btcli subnet create --subtensor.network test --wallet.name sb_owner
read -p "Enter your netuid: " NETUID
echo "NETUID=$NETUID" > .env

echo -e "\n${YELLOW}[6/7] Registering miner + validator...${NC}"
btcli subnet register --netuid "$NETUID" --subtensor.network test --wallet.name sb_miner --wallet.hotkey default
btcli subnet register --netuid "$NETUID" --subtensor.network test --wallet.name sb_validator --wallet.hotkey default

echo -e "\n${GREEN}══════════════════════════════════════════════${NC}"
echo -e "${GREEN}  ✅ DEPLOYMENT COMPLETE${NC}"
echo -e "${GREEN}══════════════════════════════════════════════${NC}"
echo -e "\n${YELLOW}Terminal 1 — Miner:${NC}"
echo "  python neurons/miner.py --netuid $NETUID --subtensor.network test --wallet.name sb_miner --wallet.hotkey default --logging.debug"
echo -e "\n${YELLOW}Terminal 2 — Validator:${NC}"
echo "  python neurons/validator.py --netuid $NETUID --subtensor.network test --wallet.name sb_validator --wallet.hotkey default --logging.debug"
echo -e "\n${YELLOW}Monitor:${NC}"
echo "  btcli wallet overview --wallet.name sb_miner --subtensor.network test"
echo "  btcli subnet metagraph --netuid $NETUID --subtensor.network test"
