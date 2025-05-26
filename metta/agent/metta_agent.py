"""
This file implements the MettaAgent class, which is a wrapper around either BrainPolicy or PufferPolicy.
It provides a unified interface for both policy types and handles the conversion between them.
"""

import logging
from typing import List, Optional, Tuple, Union

import torch
from omegaconf import DictConfig, ListConfig
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from metta.agent.brain_policy import BrainPolicy
from metta.agent.policy_state import PolicyState
from metta.agent.puffer_policy import PufferPolicy
from mettagrid.mettagrid_env import MettaGridEnv

logger = logging.getLogger("metta_agent")


def make_agent(env: MettaGridEnv, cfg: Union[DictConfig, ListConfig], device: str = "cpu") -> "MettaAgent":
    """Create a new MettaAgent instance."""
    return MettaAgent(env=env, cfg=cfg, device=device)


class DistributedMettaAgent(DistributedDataParallel):
    def __init__(self, agent, device):
        super().__init__(agent, device_ids=[device], output_device=device)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


class MettaAgent(nn.Module):
    """Base class for all policy types."""

    def __init__(self, env: MettaGridEnv, cfg: Union[DictConfig, ListConfig], device: str = "cpu"):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.env = env
        self.obs_space = env.single_observation_space
        self.action_space = env.single_action_space

        # Initialize the appropriate policy
        if hasattr(cfg.agent, "puffer"):
            self.policy = PufferPolicy(env, cfg.agent, device)
        else:
            self.policy = BrainPolicy(env, cfg.agent, device)

        # Store agent attributes for compatibility
        self.clip_range = cfg.agent.get("clip_range", 0.2)
        self.hidden_size = self.policy.hidden_size
        self.core_num_layers = self.policy.core_num_layers if hasattr(self.policy, "core_num_layers") else 1

        # Set up action space
        self.action_max_params = torch.tensor(env.single_action_space.nvec, device=device)
        self.action_index_tensor = torch.arange(int(self.action_max_params.prod()), device=device)
        self.action_names = None
        self.active_actions = None

        # Set up L2 regularization
        self.l2_init_weight_copy = None
        self.update_l2_init_weight_copy()

        # Policy metadata
        self._uri = None
        self._local_path = None
        self._name = None
        self._epoch = 0
        self._agent_step = 0
        self._generation = 0
        self._train_time = 0
        self._score = None
        self._eval_scores = {}

    @property
    def uri(self) -> Optional[str]:
        return self._uri

    @uri.setter
    def uri(self, value: str):
        self._uri = value

    @property
    def local_path(self) -> Optional[str]:
        return self._local_path

    @local_path.setter
    def local_path(self, value: str):
        self._local_path = value

    @property
    def name(self) -> Optional[str]:
        return self._name

    @name.setter
    def name(self, value: str):
        self._name = value

    @property
    def epoch(self) -> int:
        return self._epoch

    @epoch.setter
    def epoch(self, value: int):
        self._epoch = value

    @property
    def agent_step(self) -> int:
        return self._agent_step

    @agent_step.setter
    def agent_step(self, value: int):
        self._agent_step = value

    @property
    def generation(self) -> int:
        return self._generation

    @generation.setter
    def generation(self, value: int):
        self._generation = value

    @property
    def train_time(self) -> float:
        return self._train_time

    @train_time.setter
    def train_time(self, value: float):
        self._train_time = value

    @property
    def score(self) -> Optional[float]:
        return self._score

    @score.setter
    def score(self, value: float):
        self._score = value

    @property
    def eval_scores(self) -> dict:
        return self._eval_scores

    @eval_scores.setter
    def eval_scores(self, value: dict):
        self._eval_scores = value

    def key_and_version(self) -> tuple[str, int]:
        """Extract the policy key and version from the URI."""
        if not self.uri:
            return "", 0

        # Get the last part after splitting by slash
        base_name = self.uri.split("/")[-1]

        # Check if it has a version number in format ":vNUM"
        if ":" in base_name and ":v" in base_name:
            parts = base_name.split(":v")
            key = parts[0]
            try:
                version = int(parts[1])
            except ValueError:
                version = 0
        else:
            # No version, use the whole thing as key and version = 0
            key = base_name
            version = 0

        return key, version

    def key(self) -> str:
        return self.key_and_version()[0]

    def version(self) -> int:
        return self.key_and_version()[1]

    def expected_observation_channels(self) -> int:
        """Get the expected number of observation channels."""
        if hasattr(self.policy, "components") and isinstance(self.policy.components, dict):
            cnn1 = self.policy.components.get("cnn1")
            if cnn1 is not None and hasattr(cnn1, "_net") and len(cnn1._net) > 0:
                weight = cnn1._net[0].weight
                if isinstance(weight, torch.Tensor):
                    return weight.shape[1]
        return 0

    def __repr__(self):
        """Generate a detailed representation of the MettaAgent with weight shapes."""
        # Basic agent info
        lines = [f"MettaAgent(name={self.name}, uri={self.uri})"]

        # Add key metadata
        metadata_items = []
        if self.epoch is not None:
            metadata_items.append(f"epoch={self.epoch}")
        if self.agent_step is not None:
            metadata_items.append(f"agent_step={self.agent_step}")
        if self.generation is not None:
            metadata_items.append(f"generation={self.generation}")
        if self.score is not None:
            metadata_items.append(f"score={self.score}")

        if metadata_items:
            lines.append(f"Metadata: {', '.join(metadata_items)}")

        # Add total parameter count
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        lines.append(f"Total parameters: {total_params:,} (trainable: {trainable_params:,})")

        # Add module structure with detailed weight shapes
        lines.append("\nModule Structure with Weight Shapes:")

        for name, module in self.named_modules():
            # Skip top-level module
            if name == "":
                continue

            # Create indentation based on module hierarchy
            indent = "  " * name.count(".")

            # Get module type
            module_type = module.__class__.__name__

            # Start building the module info line
            module_info = f"{indent}{name}: {module_type}"

            # Get parameters for this module (non-recursive)
            params = list(module.named_parameters(recurse=False))

            # Add detailed parameter information
            if params:
                # For common layer types, add specialized shape information
                if isinstance(module, torch.nn.Conv2d):
                    weight = next((p for name, p in params if name == "weight"), None)
                    if weight is not None:
                        out_channels, in_channels, kernel_h, kernel_w = weight.shape
                        module_info += " ["
                        module_info += f"out_channels={out_channels}, "
                        module_info += f"in_channels={in_channels}, "
                        module_info += f"kernel=({kernel_h}, {kernel_w})"
                        module_info += "]"

                elif isinstance(module, torch.nn.Linear):
                    weight = next((p for name, p in params if name == "weight"), None)
                    if weight is not None:
                        out_features, in_features = weight.shape
                        module_info += f" [in_features={in_features}, out_features={out_features}]"

                elif isinstance(module, torch.nn.LSTM):
                    module_info += " ["
                    module_info += f"input_size={module.input_size}, "
                    module_info += f"hidden_size={module.hidden_size}, "
                    module_info += f"num_layers={module.num_layers}"
                    module_info += "]"

                elif isinstance(module, torch.nn.Embedding):
                    weight = next((p for name, p in params if name == "weight"), None)
                    if weight is not None:
                        num_embeddings, embedding_dim = weight.shape
                        module_info += f" [num_embeddings={num_embeddings}, embedding_dim={embedding_dim}]"

                # Add all parameter shapes
                param_shapes = []
                for param_name, param in params:
                    param_shapes.append(f"{param_name}={list(param.shape)}")

                if param_shapes and not any(
                    x in module_info for x in ["out_channels", "in_features", "hidden_size", "num_embeddings"]
                ):
                    module_info += f" ({', '.join(param_shapes)})"

            # Add formatted module info to output
            lines.append(module_info)

        # Add section for buffer shapes (non-parameter tensors like running_mean in BatchNorm)
        buffers = list(self.named_buffers())
        if buffers:
            lines.append("\nBuffer Shapes:")
            for name, buffer in buffers:
                lines.append(f"  {name}: {list(buffer.shape)}")

        return "\n".join(lines)

    def forward(
        self, x: torch.Tensor, state: Optional[PolicyState] = None, action: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the policy."""
        if state is None:
            state = PolicyState()
        return self.policy(x, state, action)

    def clip_weights(self) -> None:
        """Clip the weights of the policy."""
        if not self.clip_range:
            return

        if hasattr(self.policy, "clip_weights"):
            self.policy.clip_weights(self.clip_range)

    def l2_reg_loss(self) -> torch.Tensor:
        """Compute the L2 regularization loss."""
        if hasattr(self.policy, "l2_reg_loss"):
            return self.policy.l2_reg_loss()
        return torch.tensor(0.0, device=self.device)

    def l2_init_loss(self) -> torch.Tensor:
        """Compute the L2 initialization loss."""
        if self.l2_init_weight_copy is None:
            return torch.tensor(0.0, device=self.device)

        if hasattr(self.policy, "l2_init_loss"):
            return self.policy.l2_init_loss(self.l2_init_weight_copy)
        return torch.tensor(0.0, device=self.device)

    def update_l2_init_weight_copy(self) -> None:
        """Update the L2 initialization weight copy."""
        if hasattr(self.policy, "state_dict"):
            self.l2_init_weight_copy = {k: v.clone() for k, v in self.policy.state_dict().items()}

    def convert_action_to_logit_index(self, actions: torch.Tensor) -> torch.Tensor:
        """Convert actions to logit indices."""
        if actions.dim() == 1:
            actions = actions.unsqueeze(0)
        return torch.sum(actions * self.action_index_tensor, dim=1)

    def convert_logit_index_to_action(self, logit_indices: torch.Tensor) -> torch.Tensor:
        """Convert logit indices to actions."""
        if logit_indices.dim() == 0:
            logit_indices = logit_indices.unsqueeze(0)
        actions = []
        for idx in logit_indices:
            action = []
            remaining = idx
            for max_param in reversed(self.action_max_params):
                action.insert(0, remaining % max_param)
                remaining = remaining // max_param
            actions.append(action)
        return torch.tensor(actions, device=self.device)

    def activate_actions(self, action_names: List[str], action_max_params: List[int], device: str):
        """Initialize action space for the policy"""
        assert isinstance(action_max_params, list), "action_max_params must be a list"

        self.device = device
        self.action_max_params = torch.tensor(action_max_params, device=device)
        self.action_names = action_names
        self.active_actions = list(zip(action_names, action_max_params, strict=False))

        # Precompute cumulative sums for faster conversion
        self.cum_action_max_params = torch.cumsum(torch.tensor([0] + action_max_params, device=self.device), dim=0)

        # Create action_index tensor
        action_index = []
        for action_type_idx, max_param in enumerate(action_max_params):
            for j in range(max_param + 1):
                action_index.append([action_type_idx, j])

        self.action_index_tensor = torch.tensor(action_index, device=self.device)

        # Activate actions in policy if supported
        if hasattr(self.policy, "activate_actions"):
            if callable(self.policy.activate_actions):
                self.policy.activate_actions(action_names, action_max_params, device)

        logger.info(f"MettaAgent actions activated with: {self.active_actions}")

    @property
    def lstm(self):
        return self.policy.lstm

    @property
    def total_params(self):
        return self.policy.total_params

    def compute_weight_metrics(self, delta: float = 0.01) -> List[dict]:
        """Compute weight metrics"""
        if hasattr(self.policy, "compute_weight_metrics"):
            return self.policy.compute_weight_metrics(delta)
        return []

    def forward_training(self, *args, **kwargs):
        fn = getattr(self.policy, "forward_training", None)
        if callable(fn):
            return fn(*args, **kwargs)
        raise NotImplementedError("forward_training not available for this policy type")
