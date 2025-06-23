#!/usr/bin/env python3
"""
Simple test script to verify LLM agent integration with the simulation system.
"""

import torch
from omegaconf import OmegaConf

from metta.agent.llm_policy_adapter import LLMPolicyAdapter
from metta.agent.policy_state import PolicyState
from mettagrid.curriculum import SingleTaskCurriculum
from mettagrid.mettagrid_env import MettaGridEnv


def test_llm_agent():
    """Test basic LLM agent functionality."""
    print("Testing LLM agent integration...")

    # Create a complete environment config (similar to test_basic.yaml)
    env_config = OmegaConf.create(
        {
            "_target_": "mettagrid.mettagrid_env.MettaGridEnv",
            "sampling": 0,
            "game": {
                "map_builder": {
                    "_target_": "mettagrid.room.maze.MazePrim",
                    "width": 11,
                    "height": 11,
                    "border_width": 0,
                    # "agents": 1,
                    # "objects": {"altar": 1, "generator": 3, "wall": 10},
                    "start_pos": (0, 0),
                    "end_pos": (10, 10),
                },
                "num_agents": 1,
                "obs_width": 11,
                "obs_height": 11,
                "max_steps": 1000,
                "groups": {"agent": {"id": 0, "sprite": 0, "props": {}}},
                "agent": {
                    "default_item_max": 5,
                    "heart_max": 255,
                    "freeze_duration": 10,
                    "energy_reward": 0,
                    "hp": 1,
                    "use_cost": 0,
                    "rewards": {"heart": 1},
                },
                "objects": {
                    "altar": {"hp": 30, "cooldown": 2, "use_cost": 100},
                    "generator.red": {"hp": 30, "cooldown": 5, "initial_resources": 30, "use_cost": 0},
                    "wall": {"hp": 10},
                },
                "actions": {
                    "noop": {"enabled": True},
                    "move": {"enabled": True},
                    "rotate": {"enabled": True},
                    "attack": {"enabled": False},
                    "put_items": {"enabled": False},
                    "get_items": {"enabled": False},
                    "swap": {"enabled": False},
                    "change_color": {"enabled": False},
                },
            },
            "rendering": {},
            "features": {},
        }
    )

    # Create curriculum and environment using SingleTaskCurriculum
    curriculum = SingleTaskCurriculum("test_llm", env_config)
    env = MettaGridEnv(curriculum, render_mode=None)

    # Create LLM policy adapter
    llm_policy = LLMPolicyAdapter(env)

    # Activate actions
    action_names = env.action_names
    action_max_params = env.max_action_args
    device = torch.device("cpu")

    llm_policy.activate_actions(action_names, action_max_params, device)

    # Create dummy observation
    obs_shape = env.single_observation_space.shape
    batch_size = 2
    obs = torch.randn((batch_size, *obs_shape), device=device)

    # Create policy state
    state = PolicyState()

    # Test forward pass
    try:
        actions, log_probs, entropy, value, full_log_probs = llm_policy(obs, state)
        print("✓ Forward pass successful!")
        print(f"  - Actions shape: {actions.shape}")
        print(f"  - Actions: {actions}")
        print(f"  - Log probs shape: {log_probs.shape}")
        print(f"  - Value shape: {value.shape}")
        return True
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_llm_agent()
    if success:
        print("\n✓ LLM agent integration test passed!")
        print("\nYou can now run simulations with:")
        print("  python -m tools.sim_llm run=test_run")
        print("or:")
        print("  python -m tools.sim use_llm_agent=true run=test_run")
    else:
        print("\n✗ LLM agent integration test failed!")
        exit(1)
