const PYTHON   = "/root/superbrain-subnet/venv/bin/python";
const CWD      = "/root/superbrain-subnet";
const NETUID   = "442";
const NETWORK  = "test";

module.exports = {
  apps: [
    {
      name: "superbrain-miner",
      script: `${CWD}/scripts/start_miner.sh`,
      args: "",
      cwd: CWD,
      interpreter: "/bin/bash",
      autorestart: true,
      watch: false,
      max_restarts: 10,
      restart_delay: 10000,
      max_memory_restart: "512M",
      env: { PYTHONPATH: CWD, PYTHONUNBUFFERED: "1" },
      error_file: "logs/miner-error.log",
      out_file: "logs/miner-out.log",
    },
    {
      name: "superbrain-validator",
      script: PYTHON,
      args: "neurons/validator.py --netuid 442 --subtensor.network test --wallet.name sb_validator --wallet.hotkey default --axon.port 8092 --axon.external_ip 46.225.114.202 --neuron.vpermit_tao_limit 1000000 --logging.debug",
      cwd: CWD,
      interpreter: "none",
      autorestart: true,
      watch: false,
      max_restarts: 10,
      restart_delay: 10000,
      max_memory_restart: "512M",
      env: { PYTHONPATH: CWD, PYTHONUNBUFFERED: "1" },
      error_file: "logs/validator-error.log",
      out_file: "logs/validator-out.log",
    },
    {
      name: "superbrain-sync-node",
      script: PYTHON,
      args: "run_sync_node.py --seed --db data/sync_queue.db --port 8385",
      cwd: CWD,
      interpreter: "none",
      autorestart: true,
      watch: false,
      max_restarts: 10,
      restart_delay: 10000,
      max_memory_restart: "512M",
      env: { PYTHONPATH: CWD },
      error_file: "logs/sync-node-error.log",
      out_file: "logs/sync-node-out.log",
    },
  ],
};
