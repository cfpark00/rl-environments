from rl_environments import utils
from rl_environments.envs import BaseEnv
import copy


class ExampleEnv(BaseEnv):
    format_description = ""
    system_prompt = "Your researcher is testing out a new environment. This is the system prompt they set."
    format_error_prompt = (
        "You should probably never see this as there is no format for this environment."
    )
    default_params = {
        "n_turns": 5,
    }

    def __init__(self, logger=None, **kwargs):
        super().__init__(logger=logger)
        self.logger = logger
        self.params = copy.deepcopy(ExampleEnv.default_params)
        self.params.update(kwargs)
        self.system_prompt = ExampleEnv.system_prompt

    def get_dataset(self, n_rows=100):
        data = []
        for idx in range(n_rows):
            datum = self.get_data_sample()
            data.append(datum)
        return data

    def get_data_sample(self):
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": "This is a base environment, no data is provided.",
                "done": False,
            },
        ]
        return {"messages": messages, "hidden_params": {}}

    def get_env_response(self, messages, hiddens_params):
        n_assistant_messages = utils.get_n_assistant_messages(messages)
        env_response = {
            "role": "user",
            "content": "You are interacting with a base environment, your researcher is probably just testing things out.",
            "done": n_assistant_messages >= self.params["n_turns"],
        }
        return env_response

    def get_reward(self, messages, hidden_params):
        return 0
