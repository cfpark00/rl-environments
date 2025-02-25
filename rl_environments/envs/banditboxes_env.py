from rl_environments import utils
from rl_environments.envs import BaseEnv
from pydantic import BaseModel, ValidationError
import copy
import numpy as np
import re

class ResponseFormat(BaseModel):
    choice: int

class BanditBoxesEnv(BaseEnv):
    # Updated format description to require the new format.
    system_prompt = (
        "You will be choosing a box among {n_boxes} boxes and may receive a reward. "
        "This game will be played {n_games} times and each box has a fixed probability of giving you a reward. "
        "Try to maximize your reward over the {n_games} games.\n\n"
        "Respond in the following format:\n"
        "<think>Your thought process here</think><action>{{\"choice\": (int)}}</action>"
    )
    format_error_prompt = (
        "Please stick to the following format:\n"
        "<think>Your thought process here</think><action>{\"choice\": (int)}</action>"
    )
    default_params = {
        "n_boxes": 2,
        "n_games": 16,
    }

    def __init__(self, logger=None, **kwargs):
        super().__init__(logger=logger)
        self.params = copy.deepcopy(BanditBoxesEnv.default_params)
        self.params.update(kwargs)
        self.system_prompt = BanditBoxesEnv.system_prompt.format(
            n_boxes=self.params["n_boxes"], n_games=self.params["n_games"]
        )

    def get_dataset(self, n_rows=256):
        data = []
        for idx in range(n_rows):
            datum = self.get_data_sample()
            data.append(datum)
        return data

    def get_data_sample(self, **kwargs):
        if "p_binoms" in kwargs:
            p_binoms = kwargs["p_binoms"]
            if not isinstance(p_binoms, np.ndarray):
                p_binoms = np.array(p_binoms)
        else:
            p_binoms = np.random.dirichlet(np.ones(self.params["n_boxes"]))
        n_games = self.params["n_games"]
        n_boxes = len(p_binoms)
        assert n_boxes == self.params["n_boxes"]

        env_params = {"p_binoms": p_binoms.tolist(), "n_games": n_games}
        messages = [
            {"role": "system", "content": self.system_prompt,"env_params": env_params},
            {
                "role": "user",
                "content": self.make_task_message(0, n_boxes),
                "done": False,
            },
        ]
        data = {"messages": messages}
        return data

    @staticmethod
    def make_task_message(i_round, n_boxes):
        return (
            f"Round {i_round+1}: "
            + " vs. ".join([f"Box {i}" for i in range(1, n_boxes + 1)])
            + "?"
        )

    @staticmethod
    def get_env_response(messages):
        """
        Processes the assistant's response message:
          - Validates the existence of <think></think> tags and the JSON output in the <action> tags.
          - Uses Pydantic to validate the JSON inside <action> against the ResponseFormat model.
          - If validation fails, returns a message with the format error prompt.
          - Otherwise, computes the reward (if a valid box was chosen) and issues the next task.
        """
        env_params = messages[0]["env_params"]
        p_binoms = env_params["p_binoms"]
        n_games = env_params["n_games"]

        last_assistant_message = utils.get_last_assistant_message(messages)
        last_content = last_assistant_message["content"]
        i_round = utils.get_n_assistant_messages(messages)  # rounds are 0-indexed
        done = i_round >= n_games

        # Enforce the existence of the <think></think> tags.
        if not re.search(r"<think>.*?</think>", last_content, re.DOTALL):
            return {
                "role": "user",
                "content": BanditBoxesEnv.format_error_prompt,
                "done": done,
                "format_error": True,
                "reward": 0.0,
            }

        # Extract the JSON answer from within <action>...</action> tags.
        match = re.search(r"<action>(.*?)</action>", last_content, re.DOTALL)
        if not match:
            return {
                "role": "user",
                "content": BanditBoxesEnv.format_error_prompt,
                "done": done,
                "format_error": True,
                "reward": 0.0,
            }
        json_answer = match.group(1).strip()
        try:
            parsed_response = ResponseFormat.model_validate_json(json_answer)
            last_choice = parsed_response.choice
        except ValidationError:
            return {
                "role": "user",
                "content": BanditBoxesEnv.format_error_prompt,
                "done": done,
                "format_error": True,
                "reward": 0.0,
            }

        # Process the choice and assign reward if valid.
        if last_choice in list(range(1, len(p_binoms) + 1)):
            reward = int(np.random.binomial(1, p_binoms[last_choice - 1]))
            feedback_message = f"Reward: {reward}\n\n"
        else:
            feedback_message = "Please choose a valid box number.\n\n"
            reward = 0.0

        n_boxes = len(p_binoms)
        i_round = utils.get_n_assistant_messages(messages)  # rounds are 0-indexed
        task_message = BanditBoxesEnv.make_task_message(
            i_round=i_round, n_boxes=n_boxes
        )

        response_message = {
            "role": "user",
            "content": feedback_message + task_message,
            "done": done,
            "format_error": False,
            "reward": reward,
        }
        return response_message

    @staticmethod
    def get_reward(messages):
        reward = 0.0
        for message in messages:
            if message["role"] == "user":
                if "reward" in message:
                    reward += message["reward"]
                if "format_error" in message and message["format_error"]:
                    reward -= 1.0
        return reward
