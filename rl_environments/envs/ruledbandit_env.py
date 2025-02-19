from rl_environments import utils
from rl_environments.envs import BaseEnv
from pydantic import BaseModel, ValidationError
import copy
import numpy as np


class ResponseFormat(BaseModel):
    reasoning: str
    final_choice: int


class BanditBoxesEnv(BaseEnv):
    # Update the format description to require JSON output.
    format_description = (
        "{{\n"
        '    "reasoning": (str),    # your thought process\n'
        '    "final_choice": (int)  # the final box choice\n'
        "}}"
    )
    system_prompt = (
        "You will be choosing a box among {n_boxes} boxes and may receive a reward. "
        "This game will be played {n_games} times and each box has a fixed probability of giving you a reward. "
        "Try to maximize your reward over the {n_games} games.\n\n"
        "Respond in the following JSON format:\n" + format_description
    )
    format_error_prompt = (
        "Please stick to the JSON format:\n" + format_description.format()
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

    def get_dataset(self, n_rows=128):
        data = []
        for idx in range(n_rows):
            datum = self.get_data_sample()
            data.append(datum)
        return data

    def get_data_sample(self):
        n_boxes = self.params["n_boxes"]
        n_games = self.params["n_games"]
        p_binoms = np.random.rand(n_boxes)
        # Normalize so the sum is 1
        p_binoms = p_binoms / np.sum(p_binoms)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": self.make_task_message(0, n_boxes),
                "done": False,
            },
        ]
        hidden_params = {"p_binoms": p_binoms.tolist(), "n_games": n_games}
        data = {"messages": messages, "hidden_params": hidden_params}
        return data

    @staticmethod
    def make_task_message(i_round, n_boxes):
        return (
            f"Round {i_round+1}: "
            + " vs. ".join([f"Box {i}" for i in range(1, n_boxes + 1)])
            + "?"
        )

    @staticmethod
    def get_env_response(messages, hidden_params):
        """
        Processes the assistant's response message:
          - Uses Pydantic to validate the output JSON against the ResponseFormat model.
          - If validation fails, returns a message with the format error prompt.
          - Otherwise, computes the reward (if a valid box was chosen) and issues the next task.
        """
        p_binoms = hidden_params["p_binoms"]
        n_games = hidden_params["n_games"]

        last_assistant_message = utils.get_last_assistant_message(messages)
        last_content = last_assistant_message["content"]
        i_round = utils.get_n_assistant_messages(messages)  # rounds are 0-indexed
        done = i_round >= n_games

        # Validate and parse the assistant's output using the Pydantic model.
        try:
            parsed_response = ResponseFormat.model_validate_json(last_content)
            last_choice = parsed_response.final_choice
        except ValidationError:
            return {
                "role": "user",
                "content": BanditBoxesEnv.format_error_prompt,
                "done": done,
                "format_error": True,
            }

        # Process the choice and assign reward if valid.
        if last_choice in list(range(1, len(p_binoms) + 1)):
            reward = int(np.random.binomial(1, p_binoms[last_choice - 1]))
            feedback_message = f"Reward: {reward}\n\n"
        else:
            feedback_message = "Please choose a valid box number.\n\n"

        n_boxes = len(p_binoms)
        i_round = utils.get_n_assistant_messages(messages)  # rounds are 0-indexed
        task_message = BanditBoxesEnv.make_task_message(
            i_round=i_round, n_boxes=n_boxes
        )

        response_message = {
            "role": "user",
            "content": feedback_message + task_message,
            "done": done,
        }
        return response_message

    @staticmethod
    def get_reward(messages, hidden_params):
        reward = 0.0
        for message in messages:
            if message["role"] == "user":
                content = message["content"]
                if "Reward: " in content:
                    split = content.split("Reward: ")
                    # Expecting the reward to be the first number after "Reward: "
                    assert len(split) == 2
                    reward_str = split[1].split("\n")[0].strip()
                    reward += float(reward_str)
                if "format_error" in message and message["format_error"]:
                    reward -= 1.0
                if message["done"]:
                    break
        return reward
