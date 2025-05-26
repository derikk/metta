import pytest
import torch
from omegaconf import OmegaConf

from metta.agent.brain_policy import BrainPolicy
from metta.agent.policy_state import PolicyState


@pytest.fixture
def mock_env():
    """Create a mock environment with known observation and action spaces."""

    class MockEnv:
        def __init__(self):
            self.single_observation_space = torch.zeros((3, 5, 5))  # Example shape
            self.single_action_space = torch.zeros((2,))  # Example shape
            self.grid_features = ["agent", "wall", "mine"]
            self.global_features = []

        def action_names(self):
            return ["move", "use"]

    return MockEnv()


@pytest.fixture
def basic_config():
    """Create a basic configuration for BrainPolicy."""
    return OmegaConf.create(
        {
            "device": "cpu",
            "agent": {
                "observations": {"obs_key": "grid_obs"},
                "clip_range": 0.2,
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
                        "nn_params": {"num_layers": 1},
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
    )


def test_brain_policy_initialization(mock_env, basic_config):
    """Test that BrainPolicy initializes correctly with basic config."""
    policy = BrainPolicy(mock_env, basic_config, "cpu")
    assert policy is not None
    assert hasattr(policy, "components")
    assert "_core_" in policy.components
    assert "_action_" in policy.components
    assert "_value_" in policy.components


def test_brain_policy_forward(mock_env, basic_config):
    """Test the forward pass of BrainPolicy."""
    policy = BrainPolicy(mock_env, basic_config, "cpu")

    # Create sample input
    batch_size = 2
    obs = torch.randn(batch_size, *mock_env.single_observation_space.shape)
    state = PolicyState()

    # Test forward pass without action (inference mode)
    action, logprob, entropy, value, log_probs = policy(obs, state)

    assert action.shape == (batch_size, 2)  # Assuming 2D action space
    assert logprob.shape == (batch_size,)
    assert entropy.shape == (batch_size,)
    assert value.shape == (batch_size, 1)
    assert log_probs.shape == (batch_size, -1)  # Shape depends on action space size


def test_brain_policy_activate_actions(mock_env, basic_config):
    """Test action space activation."""
    policy = BrainPolicy(mock_env, basic_config, "cpu")

    action_names = ["move", "use"]
    action_max_params = [3, 2]  # move: [0,1,2,3], use: [0,1,2]

    policy.activate_actions(action_names, action_max_params, "cpu")

    assert policy.action_names == action_names
    assert policy.action_max_params == action_max_params
    assert policy.action_index_tensor is not None


def test_brain_policy_l2_losses(mock_env, basic_config):
    """Test L2 regularization and initialization losses."""
    policy = BrainPolicy(mock_env, basic_config, "cpu")

    # Test L2 regularization loss
    l2_reg_loss = policy.l2_reg_loss()
    assert isinstance(l2_reg_loss, torch.Tensor)
    assert l2_reg_loss.ndim == 0  # Should be a scalar

    # Test L2 initialization loss
    l2_init_loss = policy.l2_init_loss()
    assert isinstance(l2_init_loss, torch.Tensor)
    assert l2_init_loss.ndim == 0  # Should be a scalar


def test_brain_policy_clip_weights(mock_env, basic_config):
    """Test weight clipping functionality."""
    policy = BrainPolicy(mock_env, basic_config, "cpu")

    # Set a small clip range
    policy.clip_range = 0.1

    # Store original weights
    original_weights = {name: param.clone() for name, param in policy.named_parameters()}

    # Apply some large updates to weights
    for param in policy.parameters():
        param.data += torch.randn_like(param) * 10

    # Clip weights
    policy.clip_weights()

    # Check that weights were clipped
    for name, param in policy.named_parameters():
        if name in original_weights:
            max_diff = torch.max(torch.abs(param - original_weights[name]))
            assert max_diff <= policy.clip_range, f"Weights for {name} were not properly clipped"


def test_brain_policy_compute_weight_metrics(mock_env, basic_config):
    """Test weight metrics computation."""
    policy = BrainPolicy(mock_env, basic_config, "cpu")

    metrics = policy.compute_weight_metrics(delta=0.01)
    assert isinstance(metrics, list)
    assert len(metrics) > 0

    # Check that each metric has the expected structure
    for metric in metrics:
        assert isinstance(metric, dict)
        assert "name" in metric
        assert "value" in metric
