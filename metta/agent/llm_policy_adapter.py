"""
Policy adapter for LLM agents to integrate with the simulation system.
This creates a policy record that can be used by the PolicyStore and simulation system.
"""

import torch
from torch import nn

from metta.agent.llm_agent import LLMAgent
from metta.agent.policy_store import PolicyRecord
from mettagrid.mettagrid_env import MettaGridEnv


class LLMPolicyAdapter(nn.Module):
    """
    A PyTorch module wrapper around LLMAgent to make it compatible with the policy system.
    This adapter implements the interface expected by the simulation system.
    """

    def __init__(self, env: MettaGridEnv):
        super().__init__()
        self.llm_agent = LLMAgent(env)

    def activate_actions(self, action_names: list[str], action_max_params: list[int], device: torch.device) -> None:
        """Delegate to the underlying LLM agent."""
        self.llm_agent.activate_actions(action_names, action_max_params, device)

    def forward(self, obs, state):
        """Forward pass that delegates to the LLM agent."""
        return self.llm_agent(obs, state)


def create_llm_policy_record(policy_store, env: MettaGridEnv, name: str = "llm_agent") -> PolicyRecord:
    """
    Create a PolicyRecord for an LLM agent.

    Args:
        policy_store: The PolicyStore instance
        env: The MettaGridEnv to create the agent for
        name: Name for the policy record

    Returns:
        PolicyRecord that can be used in simulations
    """
    # Create the LLM policy adapter
    llm_policy = LLMPolicyAdapter(env)

    # Create a minimal PolicyRecord
    metadata = {"type": "llm_agent", "name": name, "description": "Large Language Model agent"}

    # Create a custom policy record that returns our LLM policy
    class LLMPolicyRecord(PolicyRecord):
        def __init__(self, policy_store, name: str, uri: str, metadata: dict, llm_policy):
            super().__init__(policy_store, name, uri, metadata)
            self._llm_policy = llm_policy

        def policy(self) -> nn.Module:
            return self._llm_policy

        def policy_as_metta_agent(self):
            return self._llm_policy

    uri = f"llm://{name}"
    return LLMPolicyRecord(policy_store, name, uri, metadata, llm_policy)
