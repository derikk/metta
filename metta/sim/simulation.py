import logging
import os
import time
from datetime import datetime

import numpy as np
import torch
from omegaconf import OmegaConf

from metta.agent.policy_store import PolicyRecord, PolicyStore
from metta.sim.replay_recorder import ReplayRecorder
from metta.sim.simulation_config import SimulationConfig, SimulationSuiteConfig
from metta.sim.vecenv import make_vecenv
from metta.util.config import config_from_path
from metta.util.datastruct import flatten_config

logger = logging.getLogger(__name__)


class Simulation:
    """
    A simulation is any process of stepping through a Mettagrid environment.
    Simulations configure things likes how the policies are mapped to the a
    agents, as well as which environments to run in.

    Simulations are used by training, evaluation and .
    """

    def __init__(
        self,
        config: SimulationConfig,
        policy_pr: PolicyRecord,
        policy_store: PolicyStore,
        name: str = "",
        wandb_run=None,
    ):
        self._config = config
        # TODO: Replace with typed EnvConfig
        self._env_cfg = config_from_path(config.env, config.env_overrides)
        self._env_name = config.env

        self._npc_policy_uri = config.npc_policy_uri
        self._policy_agents_pct = config.policy_agents_pct
        self._policy_store = policy_store
        self._wandb_run = wandb_run

        self._device = config.device

        self._num_envs = config.num_envs
        self._min_episodes = config.num_episodes
        self._max_time_s = config.max_time_s

        # Replay configuration
        self._replay_path = config.replay_path
        self._replay_recorders = None

        # load candidate policy
        self._policy_pr = policy_pr
        self._name = name
        # load npc policy
        self._npc_pr = None
        if self._npc_policy_uri is None:
            self._policy_agents_pct = 1.0
        else:
            self._npc_pr = self._policy_store.policy(self._npc_policy_uri)

        self._agents_per_env = self._env_cfg.game.num_agents
        self._policy_agents_per_env = max(1, int(self._agents_per_env * self._policy_agents_pct))
        self._npc_agents_per_env = self._agents_per_env - self._policy_agents_per_env
        self._total_agents = self._num_envs * self._agents_per_env

        self._vecenv = make_vecenv(self._env_cfg, config.vectorization, num_envs=self._num_envs)

        # each index is an agent, and we reshape it into a matrix of num_envs x agents_per_env
        slice_idxs = (
            torch.arange(self._vecenv.num_agents).reshape(self._num_envs, self._agents_per_env).to(device=self._device)
        )

        self._policy_idxs = slice_idxs[:, : self._policy_agents_per_env].reshape(
            self._policy_agents_per_env * self._num_envs
        )

        self._npc_idxs = []
        if self._npc_agents_per_env > 0:
            self._npc_idxs = slice_idxs[:, self._policy_agents_per_env :].reshape(
                self._num_envs * self._npc_agents_per_env
            )

        self._completed_episodes = 0
        self._total_rewards = np.zeros(self._total_agents)
        self._agent_stats = [{} for a in range(self._total_agents)]

        # Create mapping from metta.agent index to policy name
        self._agent_idx_to_policy_name = {}
        for agent_idx in self._policy_idxs:
            self._agent_idx_to_policy_name[agent_idx.item()] = self._policy_pr.name

        for agent_idx in self._npc_idxs:
            self._agent_idx_to_policy_name[agent_idx.item()] = self._npc_pr.name

        # Initialize replay recorders if a path is specified
        if self._replay_path:
            self._init_replay_recorders()

    def _get_replay_path(self, base_path, env_idx=None, episode=None):
        """
        Convert base replay path to environment/episode specific path.

        Args:
            base_path: The original replay path
            env_idx: Optional environment index
            episode: Optional episode number

        Returns:
            Path with environment/episode prefixes added to the filename
        """
        # Parse the base path
        base_dir = os.path.dirname(base_path)
        base_filename = os.path.basename(base_path)

        # Build prefixes if needed
        prefixes = []
        if env_idx is not None and self._num_envs > 1:
            prefixes.append(f"env{env_idx}")
        if episode is not None and self._min_episodes > 1:
            prefixes.append(f"ep{episode}")

        # Add prefixes to filename
        if prefixes:
            prefix_str = "_".join(prefixes) + "_"
            new_filename = f"{prefix_str}{base_filename}"
        else:
            new_filename = base_filename

        # Handle S3 paths
        if base_path.startswith("s3://"):
            bucket_and_key = base_path.split("s3://")[1]
            parts = bucket_and_key.split("/", 1)

            if len(parts) > 1:
                bucket, key = parts
                key_parts = key.rsplit("/", 1)

                if len(key_parts) > 1:
                    key_dir, _ = key_parts
                    return f"s3://{bucket}/{key_dir}/{new_filename}"
                else:
                    return f"s3://{bucket}/{new_filename}"
            else:
                return f"s3://{parts[0]}/{new_filename}"

        # Local path
        return os.path.join(base_dir, new_filename)

    def _init_replay_recorders(self):
        """Initialize one replay recorder per environment."""
        self._replay_recorders = []

        for env_idx in range(self._num_envs):
            # Get environment-specific path
            env_path = self._get_replay_path(self._replay_path, env_idx=env_idx)

            # Initialize recorder for this environment
            recorder = ReplayRecorder(env_path, self._wandb_run)
            recorder.initialize(self._vecenv.envs[env_idx])
            self._replay_recorders.append(recorder)

    def simulate(self):
        logger.info(
            f"Simulating {self._name} policy: {self._policy_pr.name} "
            + f"in {self._env_name} with {self._policy_agents_per_env} agents"
        )
        if self._npc_pr is not None:
            logger.debug(f"Against npc policy: {self._npc_pr.name} with {self._npc_agents_per_env} agents")

        logger.info(f"Simulation settings: {self._config}")

        obs, _ = self._vecenv.reset()
        policy_rnn_state = None
        npc_rnn_state = None

        game_stats = []
        start = time.time()
        step = 0
        episode_count = 0

        # set of episodes that parallelize the environments
        while self._completed_episodes < self._min_episodes and time.time() - start < self._max_time_s:
            with torch.no_grad():
                obs = torch.as_tensor(obs).to(device=self._device)
                # observavtions that correspond to policy agent
                my_obs = obs[self._policy_idxs]

                # Parallelize across opponents
                policy = self._policy_pr.policy()  # policy to evaluate
                policy_actions, _, _, _, policy_rnn_state, _, _, _ = policy(my_obs, policy_rnn_state)

                # Iterate opponent policies
                if self._npc_pr is not None:
                    npc_obs = obs[self._npc_idxs]
                    npc_rnn_state = npc_rnn_state

                    npc_policy = self._npc_pr.policy()
                    npc_action, _, _, _, npc_rnn_state, _, _, _ = npc_policy(npc_obs, npc_rnn_state)

            actions = policy_actions
            if self._npc_agents_per_env > 0:
                actions = torch.cat(
                    [
                        policy_actions.view(self._num_envs, self._policy_agents_per_env, -1),
                        npc_action.view(self._num_envs, self._npc_agents_per_env, -1),
                    ],
                    dim=1,
                )

            actions = actions.view(self._num_envs * self._agents_per_env, -1)

        if self._replay_recorders:
            for env_idx, env in enumerate(self._vecenv.envs):
                # Calculate the agent offset for this environment
                agent_offset = env_idx * self._agents_per_env

                # This matches the original ReplayHelper loop more closely
                self._replay_recorders[env_idx].update(
                    step=step,
                    env=env,
                    actions=actions,
                    rewards=self._total_rewards,
                    total_rewards=self._total_rewards,
                    action_success=env.action_success,
                    agent_offset=agent_offset,
                )

            obs, rewards, dones, truncated, infos = self._vecenv.step(actions.cpu().numpy())

            self._total_rewards += rewards
            completed_episodes_before = self._completed_episodes
            self._completed_episodes += sum([e.done for e in self._vecenv.envs])

            # Check for newly completed episodes to save replays
            if self._replay_recorders and self._completed_episodes > completed_episodes_before:
                for env_idx, env in enumerate(self._vecenv.envs):
                    if env.done:
                        # Save current episode replay
                        self._replay_recorders[env_idx].save()

                        # Reset recorder for next episode
                        new_path = self._get_replay_path(self._replay_path, env_idx=env_idx, episode=episode_count + 1)
                        self._replay_recorders[env_idx] = ReplayRecorder(new_path, self._wandb_run)
                        self._replay_recorders[env_idx].initialize(env)

            # Convert the environment configuration to a dictionary and flatten it.
            game_cfg = OmegaConf.to_container(self._env_cfg.game, resolve=False)
            flattened_env = flatten_config(game_cfg, parent_key="game")
            flattened_env["eval_name"] = self._name
            flattened_env["timestamp"] = datetime.now().isoformat()
            flattened_env["npc"] = self._npc_policy_uri

            for n in range(len(infos)):
                if "agent_raw" in infos[n]:
                    agent_episode_data = infos[n]["agent_raw"]
                    episode_reward = infos[n]["episode_rewards"]
                    for agent_i in range(len(agent_episode_data)):
                        agent_idx = agent_i + n * self._agents_per_env

                        if agent_idx in self._agent_idx_to_policy_name:
                            agent_episode_data[agent_i]["policy_name"] = self._agent_idx_to_policy_name[
                                agent_idx
                            ].replace("file://", "")
                        else:
                            agent_episode_data[agent_i]["policy_name"] = "No Name Found"
                        agent_episode_data[agent_i]["episode_reward"] = episode_reward[agent_i].tolist()
                        agent_episode_data[agent_i].update(flattened_env)

                    game_stats.append(agent_episode_data)

                    # If this is an episode completion, increment counter
                    if n in [env_idx for env_idx, env in enumerate(self._vecenv.envs) if env.done]:
                        episode_count += 1

            step += 1

        logger.debug(f"Simulation time: {time.time() - start}")

        # Save final replay states for any environments that haven't completed
        if self._replay_recorders:
            for env_idx, env in enumerate(self._vecenv.envs):
                if not env.done:
                    self._replay_recorders[env_idx].save()

        self._vecenv.close()
        return game_stats


class SimulationSuite:
    def __init__(
        self,
        config: SimulationSuiteConfig,
        policy_pr: PolicyRecord,
        policy_store: PolicyStore,
        wandb_run=None,
    ):
        logger.debug(f"Building Simulation suite from config:{config}")
        self._simulations = dict()

        for name, sim_config in config.simulations.items():
            # Create a Simulation object for each config
            sim = Simulation(
                config=sim_config, policy_pr=policy_pr, policy_store=policy_store, name=name, wandb_run=wandb_run
            )
            self._simulations[name] = sim

    def simulate(self):
        # Run all simulations and gather results by name
        return {name: sim.simulate() for name, sim in self._simulations.items()}
