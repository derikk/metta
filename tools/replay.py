# Generate a graphical trace of multiple runs.

import platform
import webbrowser

import platform
import webbrowser

import hydra

from metta.agent.policy_store import PolicyStore
from metta.sim.simulation import Simulation
from metta.sim.simulation_config import SimulationConfig
from metta.util.config import Config, setup_metta_environment
from metta.util.runtime_configuration import setup_mettagrid_environment
from metta.util.wandb.wandb_context import WandbContext


class ReplayJob(Config):
    sim: SimulationConfig
    policy_uri: str


@hydra.main(version_base=None, config_path="../configs", config_name="replay_job")
def main(cfg):
    setup_metta_environment(cfg)
    setup_mettagrid_environment(cfg)

    with WandbContext(cfg) as wandb_run:
        policy_store = PolicyStore(cfg, wandb_run)
        replay_job = ReplayJob(cfg.replay_job)
        replay_job.sim.replay_path = f"s3://softmax-public/replays/runs/{cfg.run}/replay.json.z"
        policy_record = policy_store.policy(replay_job.policy_uri)
<<<<<<< HEAD
        replay_helper = ReplayHelper(replay_job.sim, policy_record, wandb_run)
        epoch = policy_record.metadata.get("epoch", 0)
        replay_helper.generate_and_upload_replay(
            epoch,
            cfg.run_dir,
            cfg.run,
            dry_run=cfg.trainer.get("replay_dry_run", False),
        )

        # Only on macos open a browser to the replay
        if platform.system() == "Darwin":
            replay_url = f"https://softmax-public.s3.us-east-1.amazonaws.com/replays/{cfg.run}/replay.{epoch}.json.z"
=======

        sim = Simulation(replay_job.sim, policy_record, policy_store, wandb_run=wandb_run)
        sim.simulate()
        if platform.system() == "Darwin":
            replay_url = f"https://softmax-public.s3.us-east-1.amazonaws.com/replays/runs/{cfg.run}/replay.json.z"
>>>>>>> 46ba7f78 (cp)
            webbrowser.open(f"https://metta-ai.github.io/metta/?replayUrl={replay_url}")


if __name__ == "__main__":
    main()
