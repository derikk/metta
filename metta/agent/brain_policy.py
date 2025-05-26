"""
This file implements the BrainPolicy class, which is a component-based policy for Metta.
"""

import logging
from typing import List, Optional, Tuple, Union

import torch
from omegaconf import DictConfig, ListConfig

from metta.agent.policy_state import PolicyState

logger = logging.getLogger(__name__)


class BrainPolicy(torch.nn.Module):
    """Component-based policy for Metta."""

    def __init__(self, env, cfg: Union[DictConfig, ListConfig], device: str = "cpu"):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.env = env
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
        self, x: torch.Tensor, state: Optional[PolicyState] = None, action: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the policy."""
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
        action_tensor = torch.stack([action[name] for name in self.action_components.keys()], dim=-1)
        logprob_tensor = torch.stack(
            [
                log_probs[name].gather(-1, action[name].unsqueeze(-1)).squeeze(-1)
                for name in self.action_components.keys()
            ],
            dim=-1,
        )
        entropy_tensor = torch.stack([entropy[name] for name in self.action_components.keys()], dim=-1)
        value_tensor = value.squeeze(-1)
        log_probs_tensor = torch.stack([log_probs[name] for name in self.action_components.keys()], dim=-1)

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
