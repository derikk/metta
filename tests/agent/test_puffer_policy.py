import pytest
import torch
from omegaconf import OmegaConf

from metta.agent.policy_state import PolicyState
from metta.agent.puffer_policy import PufferPolicy


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
def puffer_config():
    """Create a basic configuration for PufferPolicy."""
    return OmegaConf.create(
        {
            "device": "cpu",
            "agent": {
                "observations": {"obs_key": "grid_obs"},
                "clip_range": 0.2,
            },
            "puffer": {
                "_target_": "metta.agent.external.example.Recurrent",
                "hidden_size": 512,
                "cnn_channels": 128,
            },
        }
    )


def test_puffer_policy_initialization(mock_env, puffer_config):
    """Test that PufferPolicy initializes correctly with basic config."""
    policy = PufferPolicy(mock_env, puffer_config, "cpu")
    assert policy is not None
    assert hasattr(policy, "policy")
    assert policy.hidden_size == 512
    assert policy.core_num_layers == 1


def test_puffer_policy_forward(mock_env, puffer_config):
    """Test the forward pass of PufferPolicy."""
    policy = PufferPolicy(mock_env, puffer_config, "cpu")

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


def test_puffer_policy_activate_actions(mock_env, puffer_config):
    """Test action space activation."""
    policy = PufferPolicy(mock_env, puffer_config, "cpu")

    action_names = ["move", "use"]
    action_max_params = [3, 2]  # move: [0,1,2,3], use: [0,1,2]

    policy.activate_actions(action_names, action_max_params, "cpu")

    # For puffer policies, we just verify the method exists and doesn't raise errors
    # since the actual action space handling is done by the underlying policy
    assert hasattr(policy, "activate_actions")


def test_puffer_policy_l2_losses(mock_env, puffer_config):
    """Test L2 regularization and initialization losses."""
    policy = PufferPolicy(mock_env, puffer_config, "cpu")

    # Test L2 regularization loss
    l2_reg_loss = policy.l2_reg_loss()
    assert isinstance(l2_reg_loss, torch.Tensor)
    assert l2_reg_loss.ndim == 0  # Should be a scalar

    # Test L2 initialization loss
    l2_init_loss = policy.l2_init_loss()
    assert isinstance(l2_init_loss, torch.Tensor)
    assert l2_init_loss.ndim == 0  # Should be a scalar


def test_puffer_policy_clip_weights(mock_env, puffer_config):
    """Test weight clipping functionality."""
    policy = PufferPolicy(mock_env, puffer_config, "cpu")

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


def test_puffer_policy_compute_weight_metrics(mock_env, puffer_config):
    """Test weight metrics computation."""
    policy = PufferPolicy(mock_env, puffer_config, "cpu")

    metrics = policy.compute_weight_metrics(delta=0.01)
    assert isinstance(metrics, list)

    # For puffer policies, metrics might be empty if the underlying policy
    # doesn't implement compute_weight_metrics
    if len(metrics) > 0:
        for metric in metrics:
            assert isinstance(metric, dict)
            assert "name" in metric
            assert "value" in metric


def test_puffer_policy_load_weights(mock_env, puffer_config, tmp_path):
    """Test loading weights from a file."""
    policy = PufferPolicy(mock_env, puffer_config, "cpu")

    # Save some weights
    weights_path = tmp_path / "test_weights.pt"
    torch.save(policy.state_dict(), weights_path)

    # Create a new policy with the same config
    new_policy = PufferPolicy(mock_env, puffer_config, "cpu")

    # Load the weights
    new_policy.load_state_dict(torch.load(weights_path))

    # Compare weights
    for (name1, param1), (name2, param2) in zip(policy.named_parameters(), new_policy.named_parameters(), strict=False):
        assert name1 == name2
        assert torch.allclose(param1, param2)
