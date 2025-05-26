import os
import tempfile

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
            "clip_range": 0.2,
            "observations": {"obs_key": "grid_obs"},
            "components": {
                "_obs_": {
                    "_target_": "metta.agent.lib.obs_shaper.ObsShaper",
                    "sources": None,
                },
                "obs_normalizer": {
                    "_target_": "metta.agent.lib.observation_normalizer.ObservationNormalizer",
                    "sources": [{"name": "_obs_"}],
                },
                "obs_flattener": {
                    "_target_": "metta.agent.lib.nn_layer_library.Flatten",
                    "sources": [{"name": "obs_normalizer"}],
                },
                "encoded_obs": {
                    "_target_": "metta.agent.lib.nn_layer_library.Linear",
                    "sources": [{"name": "obs_flattener"}],
                    "nn_params": {"out_features": 64},
                },
                "_core_": {
                    "_target_": "metta.agent.lib.lstm.LSTM",
                    "sources": [{"name": "encoded_obs"}],
                    "output_size": 64,
                    "nn_params": {"num_layers": 2},
                },
                "_action_embeds_": {
                    "_target_": "metta.agent.lib.action.ActionEmbedding",
                    "sources": None,
                    "nn_params": {"num_embeddings": 50, "embedding_dim": 8},
                },
                "actor_layer": {
                    "_target_": "metta.agent.lib.nn_layer_library.Linear",
                    "sources": [{"name": "_core_"}],
                    "nn_params": {"out_features": 128},
                },
                "_action_": {
                    "_target_": "metta.agent.lib.actor.MettaActorBig",
                    "sources": [{"name": "actor_layer"}, {"name": "_action_embeds_"}],
                    "bilinear_output_dim": 32,
                    "mlp_hidden_dim": 128,
                },
                "critic_layer": {
                    "_target_": "metta.agent.lib.nn_layer_library.Linear",
                    "sources": [{"name": "_core_"}],
                    "nn_params": {"out_features": 64},
                    "nonlinearity": "nn.Tanh",
                },
                "_value_": {
                    "_target_": "metta.agent.lib.nn_layer_library.Linear",
                    "sources": [{"name": "critic_layer"}],
                    "nn_params": {"out_features": 1},
                    "nonlinearity": None,
                },
            },
        },
    }

    # Create the agent with minimal config needed for the tests
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
    with tempfile.TemporaryDirectory() as tmp_dir:
        cfg = OmegaConf.create(
            {
                "device": "cpu",
                "trainer": {"checkpoint_dir": tmp_dir},
                "data_dir": tmp_dir,
            }
        )
        yield PolicyStore(cfg, None), tmp_dir


def test_validation_with_matching_spaces(mock_env, mock_policy, policy_store):
    """Test that validation succeeds when all spaces match."""
    store, tmp_dir = policy_store
    path = os.path.join(tmp_dir, "test_policy.pt")

    # Save policy with matching spaces
    pr = store.save(
        "test_policy",
        path,
        mock_policy,
        {
            "obs_space_shape": list(mock_env.single_observation_space.shape),
            "action_space_shape": list(mock_env.single_action_space.nvec),
            "grid_features": mock_env.grid_features,
        },
    )

    # Load and validate
    loaded_pr = store.load_from_uri(f"file://{path}")
    assert loaded_pr.validate_spaces()


def test_validation_with_mismatched_obs_space(mock_env, mock_policy, policy_store):
    """Test that validation fails when observation space doesn't match."""
    store, tmp_dir = policy_store
    path = os.path.join(tmp_dir, "test_policy.pt")

    # Save policy with different observation space
    pr = store.save(
        "test_policy",
        path,
        mock_policy,
        {
            "obs_space_shape": [5, 5, 3],  # Different shape
            "action_space_shape": list(mock_env.single_action_space.nvec),
            "grid_features": mock_env.grid_features,
        },
    )

    # Load and validate
    loaded_pr = store.load_from_uri(f"file://{path}")
    assert not loaded_pr.validate_spaces()


def test_validation_with_mismatched_action_space(mock_env, mock_policy, policy_store):
    """Test that validation fails when action space doesn't match."""
    store, tmp_dir = policy_store
    path = os.path.join(tmp_dir, "test_policy.pt")

    # Save policy with different action space
    pr = store.save(
        "test_policy",
        path,
        mock_policy,
        {
            "obs_space_shape": list(mock_env.single_observation_space.shape),
            "action_space_shape": [3, 3],  # Different shape
            "grid_features": mock_env.grid_features,
        },
    )

    # Load and validate
    loaded_pr = store.load_from_uri(f"file://{path}")
    assert not loaded_pr.validate_spaces()


def test_validation_with_mismatched_grid_features(mock_env, mock_policy, policy_store):
    """Test that validation fails when grid features don't match."""
    store, tmp_dir = policy_store
    path = os.path.join(tmp_dir, "test_policy.pt")

    # Save policy with different grid features
    pr = store.save(
        "test_policy",
        path,
        mock_policy,
        {
            "obs_space_shape": list(mock_env.single_observation_space.shape),
            "action_space_shape": list(mock_env.single_action_space.nvec),
            "grid_features": ["agent", "wall"],  # Different features
        },
    )

    # Load and validate
    loaded_pr = store.load_from_uri(f"file://{path}")
    assert not loaded_pr.validate_spaces()


def test_backward_compatibility(mock_env, mock_policy, policy_store):
    """Test that validation is skipped for old policies without versioning metadata."""
    store, tmp_dir = policy_store
    path = os.path.join(tmp_dir, "test_policy.pt")

    # Save policy without versioning metadata
    pr = store.save(
        "test_policy",
        path,
        mock_policy,
        {
            "epoch": 0,
            "generation": 0,
        },
    )

    # Load and validate - should pass for backward compatibility
    loaded_pr = store.load_from_uri(f"file://{path}")
    assert loaded_pr.validate_spaces()


def test_validation_during_load(mock_env, mock_policy, policy_store):
    """Test that validation is performed during policy loading."""
    store, tmp_dir = policy_store

    # Save policy with mismatched spaces
    path1 = os.path.join(tmp_dir, "test_policy1.pt")
    pr1 = store.save(
        "test_policy1",
        path1,
        mock_policy,
        {
            "obs_space_shape": [5, 5, 3],  # Different shape
            "action_space_shape": list(mock_env.single_action_space.nvec),
            "grid_features": mock_env.grid_features,
        },
    )

    # Load should fail validation but not raise an error
    loaded_pr1 = store.load_from_uri(f"file://{path1}")
    assert not loaded_pr1.validate_spaces()

    # Save policy with matching spaces
    path2 = os.path.join(tmp_dir, "test_policy2.pt")
    pr2 = store.save(
        "test_policy2",
        path2,
        mock_policy,
        {
            "obs_space_shape": list(mock_env.single_observation_space.shape),
            "action_space_shape": list(mock_env.single_action_space.nvec),
            "grid_features": mock_env.grid_features,
        },
    )

    # Load should pass validation
    loaded_pr2 = store.load_from_uri(f"file://{path2}")
    assert loaded_pr2.validate_spaces()
