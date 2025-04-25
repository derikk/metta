"""
Unified simulation driver for MettaGrid.

* `simulate()` – batch evaluation / training, *or* single-env run that can
                 emit a compressed replay (local file *or* S3 URI).
* `play()`     – interactive GUI session (Raylib).
"""

from __future__ import annotations

import json
import logging
import tempfile
import time
import zlib
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import boto3
import numpy as np
import torch
import wandb
from omegaconf import OmegaConf

from metta.agent.policy_store import PolicyRecord, PolicyStore
from metta.sim.simulation_config import SimulationConfig, SimulationSuiteConfig
from metta.sim.vecenv import make_vecenv
from metta.util.config import config_from_path
from metta.util.datastruct import flatten_config

logger = logging.getLogger(__name__)


class Simulation:
    def __init__(
        self,
        cfg: SimulationConfig,
        policy_pr: PolicyRecord,
        policy_store: Optional[PolicyStore] = None,
        name: str = "",
        *,
        wandb_run=None,
        render_mode: Optional[str] = None,  # use "human" for GUI play
    ):
        self.cfg = cfg
        self.name = name
        self.device = cfg.device
        self.wandb_run = wandb_run

        # Build environment
        self.env_cfg = config_from_path(cfg.env, cfg.env_overrides)
        self.vecenv = make_vecenv(
            self.env_cfg,
            cfg.vectorization,
            num_envs=cfg.num_envs,
            render_mode=render_mode,
        )

        # Single-env invariant for play / replay
        if cfg.replay_path or render_mode == "human":
            assert cfg.num_envs == cfg.num_episodes == 1, "Play / replay require num_envs = num_episodes = 1"

        # Candidate policy
        self.policy_pr = policy_pr
        self.policy = policy_pr.policy()

        # Optional NPC policy
        if cfg.npc_policy_uri:
            assert policy_store, "npc_policy_uri needs PolicyStore"
            self.npc_pr = policy_store.policy(cfg.npc_policy_uri)
            self.npc_policy = self.npc_pr.policy()
            share = cfg.policy_agents_pct
        else:
            self.npc_pr = None
            self.npc_policy = None
            share = 1.0

        # Agent index bookkeeping
        agents = self.env_cfg.game.num_agents
        self.policy_agents = max(1, int(agents * share))
        self.npc_agents = agents - self.policy_agents

        grid = torch.arange(self.vecenv.num_agents).reshape(cfg.num_envs, agents).to(self.device)
        self.policy_idx = grid[:, : self.policy_agents].reshape(-1)
        self.npc_idx = grid[:, self.policy_agents :].reshape(-1) if self.npc_agents else []

        # Episode bookkeeping
        self.completed = 0
        self.total_rewards = np.zeros(cfg.num_envs * agents)

    # ------------------------------------------------------------------ #
    # simulate() -- primary API for batch simulation
    # ------------------------------------------------------------------ #
    def simulate(self):
        """
        Returns nested list of per-agent episode dicts.

        If `cfg.replay_path` is defined (and we are single-env / episode),
        writes a compressed replay file.  An *s3://bucket/key* URI uploads
        automatically and logs a WandB link if `wandb_run` is provided.
        """
        capture_replay = self.cfg.replay_path is not None
        if capture_replay:
            grid_hist: list[dict] = []
            env = self.vecenv.envs[0]

        obs, _ = self.vecenv.reset()
        pol_state = npc_state = None
        start = time.time()
        episodes: list = []

        while self.completed < self.cfg.num_episodes and time.time() - start < self.cfg.max_time_s:
            # choose actions
            with torch.no_grad():
                o = torch.as_tensor(obs, device=self.device)
                pa, *_p = self.policy(o[self.policy_idx], pol_state)
                pol_state = _p[3]

                if self.npc_policy:
                    na, *_n = self.npc_policy(o[self.npc_idx], npc_state)
                    npc_state = _n[3]

            # snapshot grid (pre-step)
            if capture_replay:
                self._snap_grid(env, grid_hist)

            # merge / step
            acts = pa
            if self.npc_policy:
                acts = torch.cat(
                    [
                        pa.view(self.cfg.num_envs, self.policy_agents, -1),
                        na.view(self.cfg.num_envs, self.npc_agents, -1),
                    ],
                    1,
                ).reshape(self.vecenv.num_agents, -1)

            obs, rew, done, trunc, infos = self.vecenv.step(acts.cpu().numpy())
            self.total_rewards += rew
            self.completed += sum(e.done for e in self.vecenv.envs)

            # episode-level stats
            meta = flatten_config(OmegaConf.to_container(self.env_cfg.game, resolve=False), parent_key="game")
            meta.update(
                eval_name=self.name or "",
                timestamp=datetime.now().isoformat(),
                npc=self.npc_pr.uri if self.npc_pr else None,
            )

            for env_i, info in enumerate(infos):
                if "agent_raw" not in info:
                    continue
                for a_i, d in enumerate(info["agent_raw"]):
                    idx = env_i * self.env_cfg.game.num_agents + a_i
                    d["policy_name"] = self.policy_pr.name if idx in self.policy_idx else self.npc_pr.name
                    d["episode_reward"] = info["episode_rewards"][a_i].tolist()
                    d.update(meta)
                episodes.append(info["agent_raw"])

        # replay output (if requested)
        if capture_replay:
            local_file = self._write_replay_file(env, grid_hist, self.cfg.replay_path)
            if self.cfg.replay_path.startswith("s3://"):
                self._upload_to_s3_and_log(local_file, self.cfg.replay_path)

        self.vecenv.close()
        return episodes

    # ------------------------------------------------------------------ #
    # Replay helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _append(seq: list, step: int, val):
        if not seq or seq[-1][1] != val:
            seq.append([step, val])

    def _snap_grid(self, env, hist):
        t = env._c_env.current_timestep()
        for i, obj in enumerate(env.grid_objects.values()):
            if len(hist) <= i:
                hist.append({})
            for k, v in obj.items():
                self._append(hist[i].setdefault(k, []), t, v)

    def _write_replay_file(self, env, hist, path):
        parsed = urlparse(path)
        local_path = tempfile.mktemp(suffix=".json.z") if parsed.scheme == "s3" else Path(path).expanduser()
        if parsed.scheme != "s3":
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)

        data = {
            "version": 1,
            "action_names": env.action_names(),
            "object_types": env.object_type_names(),
            "map_size": [env.map_width, env.map_height],
            "num_agents": env.num_agents,
            "max_steps": env._c_env.current_timestep(),
            "grid_objects": [{k: v if len(v) > 1 else v[0][1] for k, v in g.items()} for g in hist],
        }
        with open(local_path, "wb") as f:
            f.write(zlib.compress(json.dumps(data).encode("utf-8")))
        logger.info("Replay written to %s", local_path)
        return local_path

    def _upload_to_s3_and_log(self, local, s3_uri):
        parsed = urlparse(s3_uri)
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")

        boto3.client("s3").upload_file(
            Filename=local,
            Bucket=bucket,
            Key=key,
            ExtraArgs={"ContentType": "application/x-compress"},
        )
        logger.info("Replay uploaded to s3://%s/%s", bucket, key)

        if self.wandb_run:
            url = f"https://{bucket}.s3.amazonaws.com/{key}"
            viewer = f"https://metta-ai.github.io/metta/?replayUrl={url}"
            self.wandb_run.log({"replays/link": wandb.Html(f'<a href="{viewer}">Replay</a>')})

    # ------------------------------------------------------------------ #
    # play() – GUI
    # ------------------------------------------------------------------ #
    def play(self):
        """
        Launches an interative simulation window
        """
        from mettagrid.renderer.raylib.raylib_renderer import MettaGridRaylibRenderer

        env = self.vecenv.envs[0]
        assert self.policy_pr.metadata["action_names"] == env._c_env.action_names(), (
            "Policy and environment action sets differ"
        )

        renderer = MettaGridRaylibRenderer(env._c_env, env._env_cfg.game)

        obs, _ = self.vecenv.reset()
        policy_rnn_state = None
        rewards = np.zeros(self.vecenv.num_agents)
        total_rewards = np.zeros(self.vecenv.num_agents)

        while True:
            # ── forward policy ─────────────────────────────────────────────
            with torch.no_grad():
                obs_t = torch.as_tensor(obs).to(device=self.device)
                actions, _, _, _, policy_rnn_state, _, _, _ = self.policy(obs_t, policy_rnn_state)
                if actions.dim() == 0:
                    actions = torch.tensor([actions.item()])

            # ── render frame (exact legacy signature) ─────────────────────
            renderer.update(
                actions.cpu().numpy(),
                obs,
                rewards,
                total_rewards,
                env._c_env.current_timestep(),
            )
            renderer.render_and_wait()
            actions = renderer.get_actions()  # user overrides

            # ── step environment ──────────────────────────────────────────
            obs, rewards, done, trunc, _ = self.vecenv.step(actions)
            total_rewards += rewards

            if done.any() or trunc.any():
                logger.info("Play finished – total rewards: %s", total_rewards)
                break

        self.vecenv.close()


class SimulationSuite:
    def __init__(self, cfg: SimulationSuiteConfig, pr: PolicyRecord, store: PolicyStore, wandb_run=None):
        self._sims = {
            n: Simulation(s_cfg, pr, store, name=n, wandb_run=wandb_run) for n, s_cfg in cfg.simulations.items()
        }

    def simulate(self):
        return {n: s.simulate() for n, s in self._sims.items()}
