"""
This file implements a PolicyStore class that manages loading and caching of trained policies.
It provides functionality to:
- Load policies from local files or remote URIs
- Cache loaded policies to avoid reloading
- Select policies based on metadata filters
- Track policy metadata and versioning

The PolicyStore is used by the training system to manage opponent policies and checkpoints.
"""

import logging
import os
from typing import Dict, Optional, Union

import torch
from omegaconf import DictConfig, ListConfig

from metta.agent.metta_agent import MettaAgent
from mettagrid.mettagrid_env import MettaGridEnv

logger = logging.getLogger("policy_store")


class PolicyStore:
    """Manages loading and caching of trained policies."""

    def __init__(self, env: MettaGridEnv, cfg: Union[DictConfig, ListConfig], device: str = "cpu"):
        self.env = env
        self.cfg = cfg
        self.device = device
        self.cache: Dict[str, MettaAgent] = {}

    def load_policy(self, policy_path: str) -> MettaAgent:
        """Load a policy from a path."""
        if policy_path in self.cache:
            return self.cache[policy_path]

        # Create a new agent
        agent = MettaAgent(self.env, self.cfg, self.device)

        # Load the policy weights
        if os.path.exists(policy_path):
            state_dict = torch.load(policy_path, map_location=self.device)
            agent.load_state_dict(state_dict)
            agent.local_path = policy_path
            agent.uri = policy_path
            agent.name = os.path.basename(policy_path)
        else:
            raise FileNotFoundError(f"Policy file not found: {policy_path}")

        # Cache the loaded policy
        self.cache[policy_path] = agent
        return agent

    def load_policy_from_uri(self, uri: str) -> MettaAgent:
        """Load a policy from a URI."""
        if uri in self.cache:
            return self.cache[uri]

        # Create a new agent
        agent = MettaAgent(self.env, self.cfg, self.device)

        # Load the policy weights
        if os.path.exists(uri):
            state_dict = torch.load(uri, map_location=self.device)
            agent.load_state_dict(state_dict)
            agent.local_path = uri
            agent.uri = uri
            agent.name = os.path.basename(uri)
        else:
            raise FileNotFoundError(f"Policy file not found: {uri}")

        # Cache the loaded policy
        self.cache[uri] = agent
        return agent

    def get_policy(self, policy_path: str) -> MettaAgent:
        """Get a policy from the cache or load it if not cached."""
        if policy_path in self.cache:
            return self.cache[policy_path]
        return self.load_policy(policy_path)

    def clear_cache(self) -> None:
        """Clear the policy cache."""
        self.cache.clear()

    def remove_from_cache(self, policy_path: str) -> None:
        """Remove a policy from the cache."""
        if policy_path in self.cache:
            del self.cache[policy_path]

    def get_cached_policies(self) -> Dict[str, MettaAgent]:
        """Get all cached policies."""
        return self.cache.copy()

    def get_policy_info(self, policy_path: str) -> Optional[Dict]:
        """Get information about a policy."""
        if policy_path not in self.cache:
            return None

        agent = self.cache[policy_path]
        return {
            "uri": agent.uri,
            "local_path": agent.local_path,
            "name": agent.name,
            "epoch": agent.epoch,
            "agent_step": agent.agent_step,
            "generation": agent.generation,
            "train_time": agent.train_time,
            "score": agent.score,
            "eval_scores": agent.eval_scores,
        }

    def save(
        self, path: str, agent: MettaAgent, metadata: Optional[dict] = None, env: Optional["MettaGridEnv"] = None
    ) -> None:
        """Save a MettaAgent to the specified path."""
        logger.info(f"Saving MettaAgent to {path}")
        torch.save(agent, path)
        self.cache[path] = agent

    def policy(self, path_or_uri: Union[str, DictConfig]) -> Optional[MettaAgent]:
        """Load a policy from a path or URI."""
        if isinstance(path_or_uri, dict):
            uri = path_or_uri.get("uri")
            if not uri:
                return None
            path = str(uri)
        else:
            path = str(path_or_uri)
        return self.get_policy(path)

    def create(self, env: "MettaGridEnv") -> MettaAgent:
        """Create a new policy for the given environment."""
        from metta.agent.metta_agent import make_agent

        return make_agent(env, self.cfg)

    def make_model_name(self, epoch: int) -> str:
        """Generate a model name for the given epoch."""
        return f"model_{epoch:06d}.pt"

    def add_to_wandb_run(self, run_name: str, agent: Optional[MettaAgent]) -> None:
        """Add a policy to a wandb run."""
        if agent is None:
            return
        # Implementation depends on wandb integration
        pass

    def _get_uri_type(self, uri: str) -> str:
        """Get the type of a URI."""
        if uri.startswith("file://"):
            return "file"
        elif uri.startswith("wandb://"):
            return "wandb"
        else:
            raise ValueError(f"Invalid URI: {uri}")
