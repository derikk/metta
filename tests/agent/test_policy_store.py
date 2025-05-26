import os

import gymnasium as gym
import numpy as np
import pytest
from omegaconf import OmegaConf

from metta.agent.metta_agent import make_agent
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
def mock_brain_agent(mock_env):
    """Create a mock policy with BrainPolicy."""
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
    return make_agent(mock_env, OmegaConf.create(cfg))


@pytest.fixture
def mock_puffer_agent(mock_env):
    """Create a mock policy with PufferPolicy."""
    cfg = {
        "device": "cpu",
        "agent": {
            "_target_": "metta.agent.metta_agent.MettaAgent",
            "clip_range": 0.2,
            "observations": {"obs_key": "grid_obs"},
            "puffer": {
                "_target_": "metta.agent.external.example.Recurrent",
                "hidden_size": 512,
                "cnn_channels": 128,
            },
        },
    }

    # Create the agent with minimal config needed for the tests
    return make_agent(mock_env, OmegaConf.create(cfg))


@pytest.fixture
def policy_store(tmp_path):
    """Create a policy store with a temporary directory for testing."""
    cfg = OmegaConf.create(
        {
            "device": "cpu",
            "trainer": {"checkpoint_dir": str(tmp_path)},
            "data_dir": str(tmp_path),
        }
    )
    return PolicyStore(cfg)


def test_save_and_load_brain_agent(policy_store, mock_brain_agent, tmp_path):
    path = str(tmp_path / "test_brain_agent.pt")
    policy_store.save(path, mock_brain_agent)
    assert os.path.exists(path)
    loaded_agent = policy_store.load(path)
    assert isinstance(loaded_agent, type(mock_brain_agent))


def test_save_and_load_puffer_agent(policy_store, mock_puffer_agent, tmp_path):
    path = str(tmp_path / "test_puffer_agent.pt")
    policy_store.save(path, mock_puffer_agent)
    assert os.path.exists(path)
    loaded_agent = policy_store.load(path)
    assert isinstance(loaded_agent, type(mock_puffer_agent))


def test_invalid_uri_loading(policy_store):
    """Test loading from invalid URI."""
    with pytest.raises(ValueError) as exc_info:
        policy_store.load_from_uri("invalid://test")
    assert "Invalid URI" in str(exc_info.value)


def test_policy_store_uri_handling(policy_store):
    """Test handling of different URI types."""
    # Test file URI
    file_uri = "file://test.pt"
    assert policy_store._get_uri_type(file_uri) == "file"

    # Test wandb URI
    wandb_uri = "wandb://test"
    assert policy_store._get_uri_type(wandb_uri) == "wandb"

    # Test invalid URI
    with pytest.raises(ValueError) as exc_info:
        policy_store._get_uri_type("invalid://test")
    assert "Invalid URI" in str(exc_info.value)
