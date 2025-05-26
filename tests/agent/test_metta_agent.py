import gymnasium as gym
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from metta.agent.brain_policy import BrainPolicy
from metta.agent.metta_agent import MettaAgent, make_agent
from metta.agent.puffer_policy import PufferPolicy


@pytest.fixture
def create_brain_agent():
    obs_space = gym.spaces.Dict(
        {
            "grid_obs": gym.spaces.Box(
                low=0,
                high=1,
                shape=(3, 5, 5, 3),
                dtype=np.float32,
            ),
            "global_vars": gym.spaces.Box(low=-np.inf, high=np.inf, shape=[0], dtype=np.int32),
        }
    )
    action_space = gym.spaces.MultiDiscrete([3, 2])
    grid_features = ["agent", "hp", "wall"]
    config_dict = {
        "clip_range": 0.1,
        "observations": {"obs_key": "grid_obs"},
        "components": {
            "_obs_": {"_target_": "metta.agent.lib.obs_shaper.ObsShaper", "sources": None},
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
    }

    class MockEnv:
        def __init__(self):
            self.single_observation_space = obs_space["grid_obs"]
            self.single_action_space = action_space
            self.grid_features = grid_features
            self.global_features = []

        def action_names(self):
            return ["action0", "action1", "action2"]

    mock_env = MockEnv()
    agent = make_agent(mock_env, OmegaConf.create({"device": "cpu", "agent": config_dict}))

    class ClippableComponent(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 10)
            self.ready = True
            self._sources = None
            self.clipped = False

        def setup(self, source_components):
            pass

        def clip_weights(self):
            self.clipped = True
            return True

        def forward(self, x):
            return x

    class MockActionEmbeds(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(50, 8)
            self.ready = True
            self._sources = None
            self.clipped = False
            self.action_names = None
            self.device = None

        def setup(self, source_components):
            pass

        def clip_weights(self):
            self.clipped = True
            return True

        def activate_actions(self, action_names, device):
            self.action_names = action_names
            self.device = device
            self.action_to_idx = {name: i for i, name in enumerate(action_names)}

        def l2_reg_loss(self):
            return torch.tensor(0.0)

        def l2_init_loss(self):
            return torch.tensor(0.0)

        def forward(self, x):
            return x

    comp1 = ClippableComponent()
    comp2 = ClippableComponent()
    action_embeds = MockActionEmbeds()
    agent.components = torch.nn.ModuleDict({"_core_": comp1, "_action_": comp2, "_action_embeds_": action_embeds})
    return agent, comp1, comp2


@pytest.fixture
def create_puffer_agent():
    obs_space = gym.spaces.Dict(
        {
            "grid_obs": gym.spaces.Box(
                low=0,
                high=1,
                shape=(3, 5, 5, 3),
                dtype=np.float32,
            ),
            "global_vars": gym.spaces.Box(low=-np.inf, high=np.inf, shape=[0], dtype=np.int32),
        }
    )
    action_space = gym.spaces.MultiDiscrete([3, 2])
    grid_features = ["agent", "hp", "wall"]
    config_dict = {
        "clip_range": 0.1,
        "observations": {"obs_key": "grid_obs"},
        "puffer": {
            "_target_": "metta.agent.external.example.Recurrent",
            "hidden_size": 512,
            "cnn_channels": 128,
        },
    }

    class MockEnv:
        def __init__(self):
            self.single_observation_space = obs_space["grid_obs"]
            self.single_action_space = action_space
            self.grid_features = grid_features
            self.global_features = []

        def action_names(self):
            return ["action0", "action1", "action2"]

    mock_env = MockEnv()
    agent = make_agent(mock_env, OmegaConf.create({"device": "cpu", "agent": config_dict}))
    return agent


def test_agent_initialization():
    """Test that MettaAgent can be initialized with both BrainPolicy and PufferPolicy."""
    obs_space = gym.spaces.Box(low=0, high=1, shape=(3, 5, 5, 3), dtype=np.float32)
    action_space = gym.spaces.MultiDiscrete([3, 2])
    grid_features = ["agent", "hp", "wall"]

    class MockEnv:
        def __init__(self):
            self.single_observation_space = obs_space
            self.single_action_space = action_space
            self.grid_features = grid_features
            self.global_features = []

        def action_names(self):
            return ["action0", "action1", "action2"]

    # Test BrainPolicy initialization
    brain_config = {
        "clip_range": 0.1,
        "observations": {"obs_key": "grid_obs"},
        "components": {
            "_obs_": {"_target_": "metta.agent.lib.obs_shaper.ObsShaper", "sources": None},
            "obs_normalizer": {
                "_target_": "metta.agent.lib.observation_normalizer.ObservationNormalizer",
                "sources": [{"name": "_obs_"}],
            },
        },
    }
    brain_agent = MettaAgent(
        env=MockEnv(),
        cfg=OmegaConf.create({"device": "cpu", "agent": brain_config}),
        device="cpu",
    )
    assert isinstance(brain_agent.policy, BrainPolicy)

    # Test PufferPolicy initialization
    puffer_config = {
        "clip_range": 0.1,
        "observations": {"obs_key": "grid_obs"},
        "puffer": {
            "_target_": "metta.agent.external.example.Recurrent",
            "hidden_size": 512,
            "cnn_channels": 128,
        },
    }
    puffer_agent = MettaAgent(
        env=MockEnv(),
        cfg=OmegaConf.create({"device": "cpu", "agent": puffer_config}),
        device="cpu",
    )
    assert isinstance(puffer_agent.policy, PufferPolicy)


def test_clip_weights_calls_components(create_brain_agent):
    agent, comp1, comp2 = create_brain_agent

    # Ensure clip_range is positive to enable clipping
    agent.clip_range = 0.1

    # Call the method being tested
    agent.clip_weights()

    # Verify each component's clip_weights was called
    assert comp1.clipped
    assert comp2.clipped


def test_clip_weights_disabled(create_brain_agent):
    agent, comp1, comp2 = create_brain_agent

    # Disable clipping by setting clip_range to 0
    agent.clip_range = 0

    # Call the method being tested
    agent.clip_weights()

    # Verify no component's clip_weights was called
    assert not comp1.clipped
    assert not comp2.clipped


def test_clip_weights_raises_attribute_error(create_brain_agent):
    agent, comp1, comp2 = create_brain_agent

    # Add a component without the clip_weights method
    class IncompleteComponent(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.ready = True
            self._sources = None

        def setup(self, source_components):
            pass

        def forward(self, x):
            return x

    # Add the incomplete component
    agent.components["bad_comp"] = IncompleteComponent()

    # Verify that an AttributeError is raised
    with pytest.raises(AttributeError) as excinfo:
        agent.clip_weights()

    # Check the error message
    assert "bad_comp" in str(excinfo.value)
    assert "clip_weights" in str(excinfo.value)


def test_clip_weights_with_non_callable(create_brain_agent):
    agent, comp1, comp2 = create_brain_agent

    # Make clip_weights non-callable on one component
    comp1.clip_weights = "Not a function"

    # Verify a TypeError is raised
    with pytest.raises(TypeError) as excinfo:
        agent.clip_weights()

    # Check the error message
    assert "not callable" in str(excinfo.value)


def test_l2_reg_loss_sums_component_losses(create_brain_agent):
    agent, comp1, comp2 = create_brain_agent

    # Add l2_reg_loss method to test components with predictable return values
    comp1.l2_reg_loss = lambda: torch.tensor(0.5)
    comp2.l2_reg_loss = lambda: torch.tensor(1.5)

    # Call the method being tested
    result = agent.l2_reg_loss()

    # Verify the result is the sum of component losses
    assert result.item() == 2.0  # 0.5 + 1.5 = 2.0


def test_l2_reg_loss_raises_attribute_error(create_brain_agent):
    agent, comp1, comp2 = create_brain_agent

    # Add a component without the l2_reg_loss method
    class IncompleteComponent(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.ready = True
            self._sources = None

        def setup(self, source_components):
            pass

        def forward(self, x):
            return x

    # Add the incomplete component
    agent.components["bad_comp"] = IncompleteComponent()

    # Verify that an AttributeError is raised
    with pytest.raises(AttributeError) as excinfo:
        agent.l2_reg_loss()

    # Check the error message
    assert "bad_comp" in str(excinfo.value)
    assert "l2_reg_loss" in str(excinfo.value)


def test_l2_reg_loss_raises_error_for_different_shapes(create_brain_agent):
    agent, comp1, comp2 = create_brain_agent

    # Add l2_reg_loss methods that return tensors of different shapes
    comp1.l2_reg_loss = lambda: torch.tensor([0.5, 1.0])
    comp2.l2_reg_loss = lambda: torch.tensor(1.5)

    # Verify that a RuntimeError is raised
    with pytest.raises(RuntimeError) as excinfo:
        agent.l2_reg_loss()

    # Check the error message
    assert "shapes" in str(excinfo.value).lower()


def test_l2_init_loss_raises_error_for_different_shapes(create_brain_agent):
    agent, comp1, comp2 = create_brain_agent

    # Add l2_init_loss methods that return tensors of different shapes
    comp1.l2_init_loss = lambda: torch.tensor([0.5, 1.0])
    comp2.l2_init_loss = lambda: torch.tensor(1.5)

    # Verify that a RuntimeError is raised
    with pytest.raises(RuntimeError) as excinfo:
        agent.l2_init_loss()

    # Check the error message
    assert "shapes" in str(excinfo.value).lower()


def test_l2_reg_loss_with_non_callable(create_brain_agent):
    agent, comp1, comp2 = create_brain_agent

    # Make l2_reg_loss non-callable on one component
    comp1.l2_reg_loss = "Not a function"

    # Verify a TypeError is raised
    with pytest.raises(TypeError) as excinfo:
        agent.l2_reg_loss()

    # Check the error message
    assert "not callable" in str(excinfo.value)


def test_l2_reg_loss_empty_components(create_brain_agent):
    agent, comp1, comp2 = create_brain_agent

    # Clear all components
    agent.components = torch.nn.ModuleDict()

    # Call the method being tested
    result = agent.l2_reg_loss()

    # Verify the result is zero
    assert result.item() == 0.0


def test_convert_action_to_logit_index(create_brain_agent):
    agent, comp1, comp2 = create_brain_agent

    # Test converting actions to logit indices
    actions = torch.tensor([[0, 1], [2, 0], [1, 1]])
    logit_indices = agent.convert_action_to_logit_index(actions)

    # Verify the shape and values
    assert logit_indices.shape == (3,)
    assert torch.all(logit_indices == torch.tensor([1, 6, 4]))


def test_convert_logit_index_to_action(create_brain_agent):
    agent, comp1, comp2 = create_brain_agent

    # Test converting logit indices to actions
    logit_indices = torch.tensor([1, 6, 4])
    actions = agent.convert_logit_index_to_action(logit_indices)

    # Verify the shape and values
    assert actions.shape == (3, 2)
    assert torch.all(actions == torch.tensor([[0, 1], [2, 0], [1, 1]]))


def test_bidirectional_action_conversion(create_brain_agent):
    agent, comp1, comp2 = create_brain_agent

    # Test bidirectional conversion
    original_actions = torch.tensor([[0, 1], [2, 0], [1, 1]])
    logit_indices = agent.convert_action_to_logit_index(original_actions)
    converted_actions = agent.convert_logit_index_to_action(logit_indices)

    # Verify the conversion is bidirectional
    assert torch.all(original_actions == converted_actions)


def test_action_conversion_edge_cases(create_brain_agent):
    agent, comp1, comp2 = create_brain_agent

    # Test edge cases
    edge_cases = [
        torch.tensor([[0, 0]]),  # Minimum values
        torch.tensor([[2, 1]]),  # Maximum values
        torch.tensor([[1, 0]]),  # Mixed values
    ]

    for actions in edge_cases:
        logit_indices = agent.convert_action_to_logit_index(actions)
        converted_actions = agent.convert_logit_index_to_action(logit_indices)
        assert torch.all(actions == converted_actions)


def test_action_use(create_brain_agent):
    agent, comp1, comp2 = create_brain_agent

    # Test action use with sample inputs
    batch_size = 2
    obs = torch.randn(batch_size, *agent.obs_space.shape)
    state = torch.zeros(batch_size, agent.hidden_size)

    # Test forward pass
    action, logprob, entropy, value, log_probs = agent(obs, state)

    # Verify output shapes
    assert action.shape == (batch_size, 2)  # Assuming 2D action space
    assert logprob.shape == (batch_size,)
    assert entropy.shape == (batch_size,)
    assert value.shape == (batch_size, 1)
    assert log_probs.shape == (batch_size, -1)  # Shape depends on action space size
