from rl_environments import utils
from rl_environments.envs import BaseEnv
from pydantic import BaseModel, ValidationError
import numpy as np
import copy


# Define a Pydantic model for the assistant's response.
class GuessResponseFormat(BaseModel):
    reasoning: str
    guess: int


class GuessNumberEnv(BaseEnv):
    # Update the format description to require JSON output.
    format_description = (
        "{{\n"
        '    "reasoning": (str),  # your thought process\n'
        '    "guess": (int)       # your guess\n'
        "}}"
    )
    system_prompt = (
        "The user will think of a number between {min_val} and {max_val}. "
        "You will be given {n_guesses} guesses to find their number, but you should try to guess it with the fewest number of guesses as possible. "
        "On each guess, the user will tell you:\n"
        '- "Higher!" if their number is higher than your guess.\n'
        '- "Lower!" if their number is lower than your guess.\n'
        '- "Correct!" if your guess is correct.\n'
        "Respond in the following JSON format:\n"+ format_description
    )
    format_error_prompt = "Please stick to the format:\n" + format_description.format()
    default_params = {
        "min_val": 1,
        "max_val": 64,
        "n_guesses": 10,
    }

    def __init__(self, logger=None, **kwargs):
        super().__init__(logger=logger)
        self.params = copy.deepcopy(GuessNumberEnv.default_params)
        self.params.update(kwargs)
        self.system_prompt = GuessNumberEnv.system_prompt.format(
            min_val=self.params["min_val"],
            max_val=self.params["max_val"],
            n_guesses=self.params["n_guesses"],
        )

    def get_dataset(self, n_rows=64):
        data = []
        for idx in range(n_rows):
            datum = self.get_data_sample()
            data.append(datum)
        return data

    def get_data_sample(self,**kwargs):
        min_val=kwargs.get("min_val",self.params["min_val"])
        max_val=kwargs.get("max_val",self.params["max_val"])
        n_guesses=kwargs.get("n_guesses",self.params["n_guesses"])
        number = kwargs.get("number",np.random.randint(min_val, max_val + 1))

        ### belows is deterministic
        env_params = {
            "number": number,
            "n_guesses": n_guesses,
            "min_val": min_val,
            "max_val": max_val,
        }
        messages = [
            {
                "role": "system",
                "content": self.system_prompt,
                "env_params": env_params,
            },
            {
                "role": "user",
                "content": f"Ok, I have a number between {min_val} and {max_val} in mind. Try to guess it!",
                "done": False,
            },
        ]
        return {"messages": messages}

    @staticmethod
    def get_env_response(messages):
        env_params = messages[0]["env_params"]
        number = env_params["number"]
        n_guesses = env_params["n_guesses"]#intentionally not setting defaults as these are needed to be passed at inference time

        last_assistant_message = utils.get_last_assistant_message(messages)
        last_content = last_assistant_message["content"]

        # Validate and parse the assistant's output using the Pydantic model.
        try:
            parsed_response = GuessResponseFormat.model_validate_json(last_content)
            last_guess = parsed_response.guess
        except ValidationError:
            return {
                "role": "user",
                "content": GuessNumberEnv.format_error_prompt,
                "done": False,
                "format_error": True,
            }

        # Process the guess.
        if last_guess == number:
            return {"role": "user", "content": "Correct!", "done": True}
        elif last_guess < number:
            feedback_message = "Higher!"
        else:
            feedback_message = "Lower!"

        i_round = utils.get_n_assistant_messages(messages)  # rounds are 0-indexed
        if i_round >= n_guesses:
            return {
                "role": "user",
                "content": feedback_message + "\nMax turns exceeded",
                "done": True,
            }
        return {"role": "user", "content": feedback_message, "done": False}

    @staticmethod
    def get_reward(messages):
        env_params = messages[0]["env_params"]
        number = env_params["number"]
        n_guesses = env_params["n_guesses"]

        reward = 0.0
        for message in messages:
            if message["role"] == "user" and message.get("format_error", False):
                reward -= 1.0

        # Try to get the guess from the assistant's last response.
        last_assistant_message = utils.get_last_assistant_message(messages)
        last_content = last_assistant_message["content"]
        try:
            parsed_response = GuessResponseFormat.model_validate_json(last_content)
            last_guess = parsed_response.guess
        except ValidationError:
            last_guess = None

        n_turns_used = utils.get_n_assistant_messages(messages)
        if last_guess is not None:
            if last_guess == number:
                reward += 10.0
                # Bonus points for using fewer guesses.
                reward += 10.0 * (
                    1.0 - min(1.0, (n_turns_used - 1.0) / (n_guesses - 1.0))
                )
            else:
                # Up to 3 points for closeness.
                reward += 3.0 * (
                    1.0 - min(1.0, (abs(last_guess - number) - 1.0) / (5.0 - 1.0))
                )
        return reward
