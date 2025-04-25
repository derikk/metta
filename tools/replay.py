"""
Generate (and optionally upload) a replay for a single episode.

Simply set `sim.replay_path` to either a local path or an S3 URI:
    - "/tmp/replay.json.z"
    - "s3://softmax-public/replays/my_run/ep0.json.z"
and this script will run one episode, save the file, upload if needed, and
log a WandB link when a WandB run is active.
"""

import sys
from dataclasses import dataclass

import hydra
from omegaconf import DictConfig

from metta.agent.policy_store import PolicyStore
from metta.sim.simulation import Simulation
from metta.sim.simulation_config import SimulationConfig
from metta.util.config import Config
from metta.util.runtime_configuration import setup_mettagrid_environment
from metta.util.wandb.wandb_context import WandbContext


class ReplayJob(Config):
    sim: SimulationConfig  # must have num_envs = num_episodes = 1
    policy_uri: str


@hydra.main(version_base=None, config_path="../configs", config_name="replay_job")
def main(cfg: DictConfig):
    setup_metta_environment(cfg)
    setup_mettagrid_environment(cfg)

    job = ReplayJob(cfg.replay_job)
    job.sim.replay_path = f"{cfg.run_dir}/replays/replay.json.z"
    job.sim.num_envs = 1
    job.sim.num_episodes = 1

    with WandbContext(cfg) as run:
        store = PolicyStore(cfg, run)
        pr = store.policy(job.policy_uri)

        sim = Simulation(job.sim, pr, policy_store=store, wandb_run=run)
        sim.simulate()  # replay is emitted/uploaded inside simulate()

        print(f"Replay complete → {job.sim.replay_path}")


if __name__ == "__main__":
    sys.exit(main())
