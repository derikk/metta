"""
This file implements the PufferPolicy class, which is a policy for Metta that uses Puffer's architecture.
"""

import logging
import os
from typing import List, Optional, Tuple, Union

import torch
from omegaconf import DictConfig, ListConfig

from metta.agent.policy_state import PolicyState
from mettagrid.mettagrid_env import MettaGridEnv

logger = logging.getLogger(__name__)


class PufferPolicy(torch.nn.Module):
    """Policy for Metta that uses Puffer's architecture."""

    def __init__(self, env: MettaGridEnv, cfg: Union[DictConfig, ListConfig], device: str = "cpu"):
        super().__init__()
        self.env = env
        self.cfg = cfg
        self.device = device
        self.obs_space = env.single_observation_space
        self.action_space = env.single_action_space

        # Initialize components
        self._initialize_components()

    def _initialize_components(self):
        """Initialize policy components."""
        # Initialize core components
        self.core = torch.nn.LSTM(
            input_size=self.obs_space.shape[0],
            hidden_size=self.cfg.hidden_size,
            num_layers=self.cfg.core_num_layers,
            batch_first=True,
        ).to(self.device)

        # Initialize action components
        self.action_components = torch.nn.ModuleDict()
        for action_name, action_size in zip(self.cfg.action_names, self.cfg.action_max_params, strict=False):
            self.action_components[action_name] = torch.nn.Linear(self.cfg.hidden_size, action_size + 1).to(self.device)

        # Initialize value head
        self.value_head = torch.nn.Linear(self.cfg.hidden_size, 1).to(self.device)

    def forward(
        self, x: torch.Tensor, state: Optional[PolicyState] = None, action: Optional[dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the policy.
        Args:
            x: Input tensor.
            state: PolicyState object.
            action: Optional dict mapping action names to tensors. If None, actions are sampled.
        Returns:
            Tuple of (action_tensor, logprob_tensor, entropy_tensor, value_tensor, log_probs_tensor)
        """
        if state is None:
            state = PolicyState()

        # Process through core
        core_out, (h_n, c_n) = self.core(x, (state.lstm_h, state.lstm_c))

        # Process through action components
        action_logits = {}
        for action_name, component in self.action_components.items():
            action_logits[action_name] = component(core_out)

        # Compute value
        value = self.value_head(core_out)

        # Update state
        state.lstm_h = h_n
        state.lstm_c = c_n
        state.hidden = core_out

        # Compute action probabilities
        action_probs = {}
        for action_name, logits in action_logits.items():
            action_probs[action_name] = torch.softmax(logits, dim=-1)

        # Sample actions if not provided
        if action is None:
            action = {}
            for action_name, probs in action_probs.items():
                action[action_name] = torch.multinomial(probs, 1).squeeze(-1)

        # Compute log probabilities
        log_probs = {}
        for action_name, logits in action_logits.items():
            log_probs[action_name] = torch.log_softmax(logits, dim=-1)

        # Compute entropy
        entropy = {}
        for action_name, probs in action_probs.items():
            entropy[action_name] = -torch.sum(probs * log_probs[action_name], dim=-1)

        # Convert to tensors
        action_names = list(self.action_components.keys())
        action_tensor = torch.stack([action[name] for name in action_names], dim=-1)
        logprob_tensor = torch.stack(
            [log_probs[name].gather(-1, action[name].unsqueeze(-1)).squeeze(-1) for name in action_names], dim=-1
        )
        entropy_tensor = torch.stack([entropy[name] for name in action_names], dim=-1)
        value_tensor = value.squeeze(-1)
        log_probs_tensor = torch.stack([log_probs[name] for name in action_names], dim=-1)

        return action_tensor, logprob_tensor, entropy_tensor, value_tensor, log_probs_tensor

    def clip_weights(self, clip_range: float) -> None:
        """Clip the weights of the policy."""
        if not clip_range:
            return

        for param in self.parameters():
            param.data.clamp_(-clip_range, clip_range)

    def l2_reg_loss(self) -> torch.Tensor:
        """Compute the L2 regularization loss."""
        l2_loss = torch.tensor(0.0, device=self.device)
        for param in self.parameters():
            l2_loss += torch.norm(param)
        return l2_loss

    def l2_init_loss(self, init_weights: dict) -> torch.Tensor:
        """Compute the L2 initialization loss."""
        l2_loss = torch.tensor(0.0, device=self.device)
        for name, param in self.named_parameters():
            if name in init_weights:
                l2_loss += torch.norm(param - init_weights[name])
        return l2_loss

    @property
    def hidden_size(self) -> int:
        return self.cfg.hidden_size

    @property
    def core_num_layers(self) -> int:
        return self.cfg.core_num_layers

    @property
    def lstm(self):
        return self.core

    @property
    def total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def compute_weight_metrics(self, delta: float = 0.01) -> List[dict]:
        """Compute weight metrics"""
        metrics = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                metrics.append(
                    {
                        "name": name,
                        "mean": param.data.mean().item(),
                        "std": param.data.std().item(),
                        "min": param.data.min().item(),
                        "max": param.data.max().item(),
                        "grad_mean": param.grad.mean().item() if param.grad is not None else 0.0,
                        "grad_std": param.grad.std().item() if param.grad is not None else 0.0,
                        "grad_min": param.grad.min().item() if param.grad is not None else 0.0,
                        "grad_max": param.grad.max().item() if param.grad is not None else 0.0,
                    }
                )
        return metrics

    def _load_from_puffer(self, path: str) -> None:
        """Load weights from a Puffer model."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Puffer model not found: {path}")

        # Load the Puffer model weights
        state_dict = torch.load(path, map_location=self.device)

        # Convert Puffer weights to our format
        converted_state_dict = {}
        for key, value in state_dict.items():
            # Convert key names if needed
            if key.startswith("puffer."):
                new_key = key[7:]  # Remove "puffer." prefix
            else:
                new_key = key
            converted_state_dict[new_key] = value

        # Load the converted weights
        self.load_state_dict(converted_state_dict)
        logger.info(f"Loaded Puffer weights from {path}")

    def activate_actions(self, action_names: list[str], action_max_params: list[int], device: str):
        """Initialize action space for the policy"""
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
        logger.info(f"PufferPolicy actions activated with: {self.active_actions}")
