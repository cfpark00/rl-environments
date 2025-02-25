import copy
import json
import re
import genson
import jsonschema
import numpy as np

from rl_environments.envs import BaseEnv
from rl_environments import utils


class ThinkActionJsonEnv(BaseEnv):
    def __init__(self, logger=None, **kwargs):
        super().__init__(logger=logger)
        self.data=json.load(open("/n/home12/cfpark00/ML/llm-meta-rl/rl-environments/assets/think_action_json.json","r"))

    def get_data_sample(self, **kwargs):
        idx=kwargs.pop("idx",np.random.randint(len(self.data)))
        assert len(kwargs)==0
        datum=copy.deepcopy(self.data[np.random.randint(len(self.data))])
        system_prompt=datum["system_prompt"]
        user_prompt=datum["user_prompt"]
        json_str=datum["json_str"]
        messages=[
            {"role":"system","content":system_prompt,"env_params":{"json_str":json_str}},
            {"role":"user","content":user_prompt,"done":False}
        ]
        return {"messages":messages}
        
    @staticmethod
    def get_env_response(messages):
        return {"role":"user","content":"Done.","done": True}

    @staticmethod
    def get_reward(messages):
        env_params = messages[0]["env_params"]
        json_str=env_params["json_str"]

        builder = genson.SchemaBuilder()
        builder.add_object(json.loads(json_str))
        schema = builder.to_schema()

        last_assistant_message = utils.get_last_assistant_message(messages)
        last_content=last_assistant_message["content"]

        reward=0.0
        think_match = re.search(r"<think>(.*?)</think>", last_content, re.DOTALL)
        if think_match and len(think_match.groups())==1:
            reward+=1.0
        action_match = re.search(r"<action>(.*?)</action>", last_content, re.DOTALL)
        if action_match and len(action_match.groups())==1:
            reward+=1.0
            response_json_str = action_match.group(1).strip()
            try:
                response_json = json.loads(response_json_str)
                reward+=1.0
            except json.JSONDecodeError:
                response_json = None
            if response_json is not None:
                try:
                    jsonschema.validate(instance=response_json, schema=schema)
                    reward+=3.0
                except jsonschema.ValidationError:
                    pass
        return reward
