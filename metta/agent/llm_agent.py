import json
from typing import Optional, Tuple

import numpy as np
import torch

from metta.agent.policy_state import PolicyState
from mettagrid.char_encoder import grid_object_to_char
from mettagrid.mettagrid_env import MettaGridEnv

action_descriptions = {
    "noop": "Do nothing",
    "move": 'Move in a specified direction ("up", "down", "left", or "right")',
    "rotate": 'Rotate to face a specified direction ("up", "down", "left", or "right")',
    "put_items": "",
    "get_items": "",
    "attack": "",
    "swap": "Swap positions with an object directly in front of you",
    "change_color": "",
}


class LLMAgent:
    def __init__(self, env: MettaGridEnv):
        self.env = env
        self.action_names = self.env.action_names
        self.object_type_names = self.env.object_type_names
        self.type_id_feature_index = 0

        # Initialize attributes for simulation compatibility
        self.device = None
        self.action_max_params = None

    def activate_actions(self, action_names: list[str], action_max_params: list[int], device: torch.device) -> None:
        """Run this at the beginning of training/simulation."""
        self.device = device
        self.action_names = action_names
        self.action_max_params = action_max_params

    def __call__(
        self, obs: torch.Tensor, state: PolicyState
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Policy interface expected by the simulation system.

        Args:
            obs: Observation tensor of shape (batch_size, height, width, features)
            state: PolicyState containing LSTM states (unused for LLM agent)

        Returns:
            Tuple of (action, action_log_prob, entropy, value, log_probs)
        """
        batch_size = obs.shape[0]

        # Convert observations to actions
        actions = []
        for i in range(batch_size):
            # The observation for each agent is already in the correct format
            obs_np = obs[i].cpu().numpy()
            # Add agent dimension since get_action expects it
            obs_with_agent_dim = obs_np[np.newaxis, ...]  # shape: (1, height, width, features)
            action_id, action_arg = self.get_action(obs_with_agent_dim, agent_idx=0)
            actions.append([action_id, action_arg])

        actions = torch.tensor(actions, device=self.device, dtype=torch.long)

        # Create dummy values for compatibility (LLM doesn't provide these)
        action_log_prob = torch.zeros(batch_size, device=self.device)
        entropy = torch.zeros(batch_size, device=self.device)
        value = torch.zeros((batch_size, 1), device=self.device)

        # Create dummy log_probs for all actions
        num_actions = (
            sum(max_param + 1 for max_param in self.action_max_params)
            if self.action_max_params
            else len(self.action_names)
        )
        log_probs = torch.full((batch_size, num_actions), -float("inf"), device=self.device)

        return actions, action_log_prob, entropy, value, log_probs

    def get_action(self, obs: np.ndarray, agent_idx: int = 0) -> tuple[int, int]:
        """
        Get an action from the LLM based on the current observation.
        """
        ascii_grid = self._observation_to_ascii(obs, agent_idx)
        prompt = self._format_prompt(ascii_grid)

        print("--- PROMPT ---")
        print(prompt)

        llm_output = self._get_llm_action(prompt)

        print("--- LLM OUTPUT ---")
        print(llm_output)

        action_name, action_arg = self._parse_llm_output(llm_output)

        if action_name not in self.action_names:
            print(f"LLM returned invalid action: {action_name}")
            return self._get_default_action()

        action_id = self.action_names.index(action_name)
        return action_id, action_arg

    def _observation_to_ascii(self, obs: np.ndarray, agent_idx: int = 0) -> str:
        """
        Convert the observation to an ASCII grid.
        """
        agent_obs = obs[agent_idx]
        grid_h, grid_w, num_features = agent_obs.shape

        char_grid = []
        for r in range(grid_h):
            row_chars = []
            for c in range(grid_w):
                feature_vec = agent_obs[r, c]
                if np.sum(feature_vec) == 0:
                    row_chars.append(".")  # empty
                else:
                    if self.type_id_feature_index != -1:
                        obj_type_id = int(feature_vec[self.type_id_feature_index])
                        if 0 <= obj_type_id < len(self.object_type_names):
                            obj_name = self.object_type_names[obj_type_id]
                            if obj_name == "agent":
                                obj_name = "agent.agent"
                            row_chars.append(grid_object_to_char(obj_name))
                        else:
                            row_chars.append("?")  # Unknown object type id
                    else:
                        row_chars.append("?")  # type_id feature not found
            char_grid.append("".join(row_chars))

        # Agent is at the center
        center_r, center_c = grid_h // 2, grid_w // 2
        center_row_list = list(char_grid[center_r])
        center_row_list[center_c] = "@"
        char_grid[center_r] = "".join(center_row_list)

        return "\n".join(char_grid)

    def _format_prompt(self, ascii_grid: str) -> str:
        """
        Format the prompt for the LLM.
        """
        prompt = "You are an agent in a 2D grid world.\n"
        prompt += "Your goal is to survive and thrive.\n\n"
        prompt += "This is what you see:\n"
        prompt += ascii_grid + "\n\n"
        prompt += "Available actions:\n"
        for i, action_name in enumerate(self.action_names):
            prompt += f"- {action_name}: {action_descriptions[action_name]}\n"

        prompt += "\n"
        prompt += "Some actions require a parameter (e.g., direction for 'move').\n"
        prompt += "Respond with a JSON object containing 'action' and 'parameter'.\n"
        prompt += 'Example: {"action": "move", "parameter": 0}\n'
        prompt += "Your action: "
        return prompt

    def _get_llm_action(self, prompt: str) -> str:
        """
        Get an action from the LLM.
        """
        try:
            llm = None
            # response = llm.prompt(
            #     prompt=prompt,
            #     engine="gpt-4",
            #     max_tokens=50,
            #     temperature=0.0,
            # )
            # return response.completion
            return '{"action": "move", "parameter": "up"}'
        except Exception as e:
            print(f"Error getting action from LLM: {e}")
            return json.dumps({"action": "noop", "parameter": 0})

    def _parse_llm_output(self, output: str) -> tuple[Optional[str], int]:
        """
        Parse the JSON output from the LLM.
        """
        try:
            data = json.loads(output)
            action_name = data.get("action")
            parameter = data.get("parameter", 0)

            if action_name in ["move", "rotate"]:
                parameter = ["up", "down", "left", "right"].index(parameter)

            return action_name, int(parameter)
        except (json.JSONDecodeError, TypeError):
            print(f"Could not parse LLM output: {output}")
            return None, 0

    def _get_default_action(self) -> tuple[int, int]:
        """
        Return a default action (e.g., noop) if the LLM fails.
        """
        if "noop" in self.action_names:
            return self.action_names.index("noop"), 0
        return 0, 0
