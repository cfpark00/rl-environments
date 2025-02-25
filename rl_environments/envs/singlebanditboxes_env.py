from rl_environments import utils
from rl_environments.envs import BaseEnv
from pydantic import BaseModel, ValidationError
import copy
import numpy as np


class ResponseFormat(BaseModel):
    reasoning: str
    final_choice: int


class SingleBanditBoxesEnv(BaseEnv):
    # Update the format description to require JSON output.
    format_description = (
        "{{\n"
        '    "reasoning": (str),    # your thought process\n'
        '    "final_choice": (int)  # the final box choice\n'
        "}}"
    )
    system_prompt = (
        "You will be choosing a box among {n_boxes} boxes and may receive a reward. "
        "Try to maximize your reward.\n\n"
        "Respond in the following JSON format:\n" + format_description
    )
    format_error_prompt = (
        "Please stick to the JSON format:\n" + format_description.format()
    )
    default_params = {
        "p_binoms": [0.2, 0.9, 0.2],
    }

    def __init__(self, logger=None, **kwargs):
        super().__init__(logger=logger)
        self.params = copy.deepcopy(SingleBanditBoxesEnv.default_params)
        self.params.update(kwargs)
        self.params["n_boxes"] = len(self.params["p_binoms"])
        self.system_prompt = SingleBanditBoxesEnv.system_prompt.format(
            n_boxes=self.params["n_boxes"]
        )

    def get_dataset(self, n_rows=128):
        data = []
        for idx in range(n_rows):
            datum = self.get_data_sample()
            data.append(datum)
        return data

    def get_data_sample(self):
        p_binoms = self.params["p_binoms"]
        n_boxes = len(p_binoms)
        env_params = {"p_binoms": p_binoms.tolist() if isinstance(p_binoms, np.ndarray) else p_binoms}
        messages = [
            {"role": "system", "content": self.system_prompt, "env_params": env_params},
            {"role": "user", "content": self.make_task_message(n_boxes), "done": False},
        ]
        data = {"messages": messages}
        return data

    @staticmethod
    def make_task_message(n_boxes):
        return " vs. ".join([f"Box {i}" for i in range(1, n_boxes + 1)]) + "?"

    @staticmethod
    def get_env_response(messages):
        """
        Processes the assistant's response message:
          - Uses Pydantic to validate the output JSON against the ResponseFormat model.
          - If validation fails, returns a message with the format error prompt.
          - Otherwise, computes the reward (if a valid box was chosen) and issues the next task.
        """
        env_params = messages[0]["env_params"]
        p_binoms = env_params["p_binoms"]

        last_assistant_message = utils.get_last_assistant_message(messages)
        last_content = last_assistant_message["content"]
        done = True  # single turn

        # Validate and parse the assistant's output using the Pydantic model.
        try:
            parsed_response = ResponseFormat.model_validate_json(last_content)
            last_choice = parsed_response.final_choice
        except ValidationError:
            return {
                "role": "user",
                "content": SingleBanditBoxesEnv.format_error_prompt,
                "done": done,
                "format_error": True,
            }

        # Process the choice and assign reward if valid.
        if last_choice in list(range(1, len(p_binoms) + 1)):
            reward = int(np.random.binomial(1, p_binoms[last_choice - 1]))
            feedback_message = f"Reward: {reward}\n\n"
        else:
            feedback_message = "Please choose a valid box number.\n\n"

        response_message = {
            "role": "user",
            "content": feedback_message,
            "done": done,
        }
        return response_message

    @staticmethod
    def get_reward(messages):
        env_params = messages[0]["env_params"]
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
