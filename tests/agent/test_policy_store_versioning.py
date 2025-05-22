import gymnasium as gym
import numpy as np
import pytest
from omegaconf import OmegaConf

from metta.agent.metta_agent import MettaAgent
from metta.agent.policy_store import PolicyStore


@pytest.fixture
def mock_env():
    """Create a mock environment with known observation and action spaces."""

    class MockEnv:
        def __init__(self):
            self.single_observation_space = gym.spaces.Box(low=0, high=255, shape=(10, 10, 5), dtype=np.uint8)
            self.single_action_space = gym.spaces.MultiDiscrete([5, 5])
            self.grid_features = ["agent", "wall", "mine", "generator"]

        def action_names(self):
            return ["move", "use"]

    return MockEnv()


@pytest.fixture
def mock_policy(mock_env):
    """Create a mock policy with known attributes."""
    cfg = {
        "device": "cpu",
        "agent": {
            "_target_": "metta.agent.metta_agent.MettaAgent",
            "components": {"_core_": {"output_size": 64, "nn_params": {"num_layers": 2}}},
            "observations": {"obs_key": "grid_obs"},
            "clip_range": 0.2,
        },
    }
    return MettaAgent(
        obs_space=mock_env.single_observation_space,
        action_space=mock_env.single_action_space,
        grid_features=mock_env.grid_features,
        device="cpu",
        **cfg["agent"],
    )


@pytest.fixture
def policy_store():
    """Create a policy store for testing."""
    cfg = OmegaConf.create({"device": "cpu", "trainer": {"checkpoint_dir": "test_checkpoints"}})
    return PolicyStore(cfg, None)


def test_versioning_validation_success(policy_store, mock_env, mock_policy, tmp_path):
    """Test successful validation of policy spaces."""
    # Create a policy with matching spaces
    pr = policy_store.save(
        "test_policy",
        str(tmp_path / "test_policy.pt"),
        mock_policy,
        {
            "obs_space_shape": list(mock_env.single_observation_space.shape),
            "action_space_shape": list(mock_env.single_action_space.nvec),
            "grid_features": mock_env.grid_features,
        },
    )

    # Should validate successfully
    assert pr.validate_spaces()


def test_versioning_validation_failure(policy_store, mock_env, mock_policy, tmp_path):
    """Test validation failure with mismatched spaces."""
    # Create a policy with different spaces
    pr = policy_store.save(
        "test_policy",
        str(tmp_path / "test_policy.pt"),
        mock_policy,
        {
            "obs_space_shape": [5, 5, 3],  # Different shape
            "action_space_shape": [3, 3],  # Different actions
            "grid_features": ["agent", "wall"],  # Different features
        },
    )

    # Should fail validation
    assert not pr.validate_spaces()


def test_old_policy_without_versioning(policy_store, mock_env, mock_policy, tmp_path):
    """Test handling of old policies without versioning information."""
    # Create a policy without versioning metadata
    pr = policy_store.save("test_policy", str(tmp_path / "test_policy.pt"), mock_policy, {"epoch": 0, "generation": 0})

    # Should pass validation (backwards compatibility)
    assert pr.validate_spaces()


def test_puffer_policy_loading(policy_store, mock_env, mock_policy, tmp_path):
    """Test that puffer policies skip validation."""
    # Mock loading a puffer policy
    uri = "puffer://test_policy"
    pr = policy_store.load_from_uri(uri)

    # Should have loaded without validation errors
    assert pr is not None


def test_load_policy_with_mismatched_spaces(policy_store, mock_env, mock_policy, tmp_path):
    """Test that loading a policy with mismatched spaces raises an error."""
    # Create a policy with different spaces
    path = str(tmp_path / "test_policy.pt")
    policy_store.save(
        "test_policy",
        path,
        mock_policy,
        {
            "obs_space_shape": [5, 5, 3],  # Different shape
            "action_space_shape": [3, 3],  # Different actions
            "grid_features": ["agent", "wall"],  # Different features
        },
    )

    # Loading should raise a ValueError
    with pytest.raises(ValueError) as exc_info:
        policy_store.load_from_uri(f"file://{path}")
    assert "different observation/action spaces" in str(exc_info.value)
