import logging
import os
import signal
import sys
from dataclasses import dataclass
from datetime import datetime

import hydra
from rich.logging import RichHandler

from metta.agent.policy_store import PolicyStore
from metta.sim.simulation import Simulation
from metta.sim.simulation_config import SimulationConfig
from metta.util.config import Config
from metta.util.runtime_configuration import setup_mettagrid_environment
from metta.util.wandb.wandb_context import WandbContext

signal.signal(signal.SIGINT, lambda *_: os._exit(0))  # immediate exit on Ctrl-C


# ─── rich logging ────────────────────────────────────────────────────────────
class _MilliFmt(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        created = datetime.fromtimestamp(record.created)
        msec = created.microsecond // 1000
        datefmt = "[%H:%M:%S.%f]" if datefmt is None else datefmt.replace("%f", f"{msec:03d}")
        return created.strftime(datefmt)


handler = RichHandler(rich_tracebacks=True)
handler.setFormatter(_MilliFmt("%(message)s"))
logging.basicConfig(level="DEBUG", handlers=[handler])
logger = logging.getLogger(__name__)


class PlayJob(Config):
    sim: SimulationConfig
    policy_uri: str


@hydra.main(version_base=None, config_path="../configs", config_name="play_job")
def main(cfg):
    # reset logging after Hydra init
    root = logging.getLogger()
    for h in root.handlers[:]:
        root.removeHandler(h)
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)

    setup_mettagrid_environment(cfg)

    with WandbContext(cfg) as wandb_run:
        policy_store = PolicyStore(cfg, wandb_run)
        play_job = PlayJob(cfg.play_job)
        sim = Simulation(play_job.sim, policy_store.policy(play_job.policy_uri), policy_store=policy_store, render_mode="human", wandb_run=wandb_run)
        sim.play()


if __name__ == "__main__":
    sys.exit(main())
