from rl_environments import utils
from rl_environments.envs import BaseEnv
from pydantic import BaseModel, ValidationError
import copy
import numpy as np
import re

rules=[
    "circular",
    "circular_stay",
    "nth",
    "nth_value",
    "odd_smallest_even_biggest",
]
rule_kwargs={
    "circular": ["box_numbers", "seed"],
    "circular_stay": ["box_numbers", "n_stay", "seed"],
    "nth": ["n_boxes", "box_numbers", "nth"],
    "nth_value": ["n_boxes", "box_numbers", "nth_value"],
    "odd_smallest_even_biggest": ["n_boxess", "box_numbers"],
}

class ResponseFormat(BaseModel):
    choice: int

class RuledBanditBoxesEnv(BaseEnv):
    # Updated format description to require the new format.
    system_prompt = (
        "You will be choosing a box between options and may receive a reward. "
        "Try to maximize your reward over {n_games} games.\n\n"
        "Respond in the following format:\n"
        "<think>Your thought process here</think><action>{{\"choice\": (int)}}</action>"
    )
    default_params = {
        "n_games": 16,
        "rule": "circular",
    }

    def __init__(self, logger=None, **kwargs):
        super().__init__(logger=logger)
        self.params = copy.deepcopy(RuledBanditBoxesEnv.default_params)
        self.params.update(kwargs)
        self.system_prompt = RuledBanditBoxesEnv.system_prompt.format(
            n_games=self.params["n_games"]
        )

    def get_dataset(self, n_rows=256):
        data = []
        for idx in range(n_rows):
            datum = self.get_data_sample()
            data.append(datum)
        return data

    def get_data_sample(self, **kwargs):
        kwargs_=copy.deepcopy(self.params)
        kwargs_.update(kwargs)
        kwargs=kwargs_
        assert "rule" in kwargs
        kwargs_set=set(rule_kwargs[kwargs["rule"]])-set(["seed"])
        kwargs_set_got=set(kwargs.keys())-set(["rule", "n_games"])
        assert kwargs_set==kwargs_set_got, f"rule={kwargs['rule']} requires {kwargs_set} but got {kwargs_set_got}"

        env_params, user_message = self.get_env_params_user_message(**kwargs)
        messages = [
            {"role": "system", "content": self.system_prompt,"env_params": env_params},
            user_message,
        ]
        return {"messages": messages}
    
    def get_env_params_user_message(self, **kwargs):
        rule=kwargs["rule"]
        return rule_instances[rule].get_env_params_user_message(**kwargs)

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
        rule=env_params["rule"]
        n_games = int(env_params["n_games"])
        last_assistant_message = utils.get_last_assistant_message(messages)
        last_content = last_assistant_message["content"]
        i_round = utils.get_n_assistant_messages(messages)  # rounds are 0-indexed
        done = i_round >= n_games

        # Validate the format by checking for <think> and <action> tags.
        think_matches = re.search(r"<think>(.*?)</think>", last_content, re.DOTALL)
        action_matches = re.search(r"<action>(.*?)</action>", last_content, re.DOTALL)
        if not (think_matches and action_matches and len(think_matches.groups()) == 1 and len(action_matches.groups()) == 1):
            last_choice=None
        else:
            json_answer = action_matches.group(1).strip()
            try:
                parsed_response = ResponseFormat.model_validate_json(json_answer)
                last_choice = parsed_response.choice
                last_choice = int(last_choice)
            except ValidationError:
                last_choice=None

        response_message=rule_instances[rule].get_rule_response(messages, last_choice)
        if last_choice is None:
            response_message["format_error"]=True
        response_message["done"]=done
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


#####
class CircularRule():
    def __init__(self):
        pass

    def get_env_params_user_message(self, **kwargs):
        box_numbers = kwargs.get("box_numbers",[1,2])
        n_boxes=len(box_numbers)
        seed=kwargs.get("seed", None)
        if seed is None:
            seed=np.random.randint(0, 2**32-1)
        np.random.seed(seed)
        phase=np.random.randint(0, n_boxes)
        env_params=copy.deepcopy(kwargs)
        env_params.update({
            "phase": phase,
            "box_numbers": box_numbers,
            "n_boxes": n_boxes
        })
        i_round=0
        user_message_content=(
                f"Round {i_round+1}: "
                + " vs. ".join([f"Box {n}" for n in box_numbers])
                + "?"
            )
        i_reward=(i_round+phase)%n_boxes
        box_reward=box_numbers[i_reward]
        return env_params, {"role": "user", "content": user_message_content, "done": False, "i_round": i_round, "box_reward": box_reward}
    
    def get_rule_response(self, messages, last_choice):
        env_params=messages[0]["env_params"]
        assert "rule" in env_params
        assert "n_games" in env_params
        assert env_params["rule"]=="circular"
        assert messages[-2]["role"]=="user"
        last_user_message=messages[-2]
        #rule specifics
        assert "phase" in env_params, "rule=circular requires phase"
        assert "n_boxes" in env_params, "rule=circular requires n_boxes"
        assert "box_numbers" in env_params, "rule=circular requires box_numbers"
        assert "i_round" in last_user_message, "rule=circular requires i_round"
        assert "box_reward" in last_user_message, "rule=circular requires box_reward"

        phase=env_params["phase"]
        n_boxes=env_params["n_boxes"]
        box_numbers=env_params["box_numbers"]
        i_round_last=last_user_message["i_round"]
        box_reward_last=last_user_message["box_reward"]

        i_round_new=i_round_last+1
        task_message_new=(
                f"Round {i_round_new+1}: "#+1 for 0-indexed
                +" vs. ".join([f"Box {n}" for n in box_numbers])
                +"?"
        )
        i_reward_new=(i_round_new+phase)%n_boxes
        box_reward_new=box_numbers[i_reward_new]
        if last_choice is not None:
            reward = 1.0 if last_choice==box_reward_last else 0.0
            user_message_content=f"Reward: {reward}\n\n"+task_message_new
        else:
            reward=0.0
            user_message_content=f"Format Error.\n\n"+task_message_new

        response_message = {
            "role": "user",
            "content": user_message_content,
            "reward": reward,
            #specifics
            "i_round": i_round_new,
            "box_reward": box_reward_new,
        }
        return response_message

class CircularStayRule():
    def __init__(self):
        pass

    def get_env_params_user_message(self, **kwargs):
        box_numbers = kwargs.get("box_numbers",[1,2])
        n_boxes = len(box_numbers)
        n_stay = kwargs.get("n_stay", 2)
        effective_period = n_boxes * n_stay

        seed = kwargs.get("seed", None)
        if seed is None:
            seed = np.random.randint(0, 2**32 - 1)
        np.random.seed(seed)
        # Phase is now chosen from the range [0, effective_period)
        phase = np.random.randint(0, effective_period)

        env_params = copy.deepcopy(kwargs)
        env_params.update({
            "phase": phase,
            "box_numbers": box_numbers,
            "n_boxes": n_boxes,
            "n_stay": n_stay,
            "effective_period": effective_period,
        })

        i_round = 0
        # Build the repeated pattern: each box repeated n_stay times.
        pattern = []
        for b in box_numbers:
            pattern.extend([b] * n_stay)
        i_reward = (i_round + phase) % effective_period
        box_reward=pattern[i_reward]

        user_message_content = (
            f"Round {i_round+1}: " +
            " vs. ".join([f"Box {n}" for n in box_numbers]) +
            "?"
        )
        return env_params, {
            "role": "user",
            "content": user_message_content,
            "done": False,
            "i_round": i_round,
            "box_reward": box_reward,
        }
    
    def get_rule_response(self, messages, last_choice):
        env_params = messages[0]["env_params"]
        assert "rule" in env_params
        assert "n_games" in env_params
        assert env_params["rule"] == "circular_stay"
        assert messages[-2]["role"] == "user"
        last_user_message = messages[-2]

        # Ensure required parameters exist
        assert "phase" in env_params, "rule=circular_stay requires phase"
        assert "n_boxes" in env_params, "rule=circular_stay requires n_boxes"
        assert "n_stay" in env_params, "rule=circular_stay requires n_stay"
        assert "box_numbers" in env_params, "rule=circular_stay requires box_numbers"
        assert "i_round" in last_user_message, "rule=circular_stay requires i_round"
        assert "box_reward" in last_user_message, "rule=circular_stay requires box_reward"

        phase = env_params["phase"]
        n_stay = env_params["n_stay"]
        box_numbers = env_params["box_numbers"]
        n_boxes = env_params["n_boxes"]
        effective_period = n_boxes * n_stay

        # Build the pattern as in the initialization step.
        pattern = []
        for b in box_numbers:
            pattern.extend([b] * n_stay)

        i_round_last = last_user_message["i_round"]
        box_reward_last=last_user_message["box_reward"]

        i_round_new = i_round_last + 1
        task_message_new = (
            f"Round {i_round_new+1}: " +
            " vs. ".join([f"Box {n}" for n in box_numbers]) +
            "?"
        )
        i_reward_new = (i_round_new + phase) % effective_period
        box_reward_new = pattern[i_reward_new]
        
        if last_choice is not None:
            reward = 1.0 if last_choice == box_reward_last else 0.0
            user_message_content = f"Reward: {reward}\n\n" + task_message_new
        else:
            reward = 0.0
            user_message_content = f"Format Error.\n\n" + task_message_new

        response_message = {
            "role": "user",
            "content": user_message_content,
            "reward": reward,
            "i_round": i_round_new,
            "box_reward": box_reward_new,
        }
        return response_message

class NthRule():
    def __init__(self):
        pass

    def get_env_params_user_message(self, **kwargs):
        # Defaults:
        n_boxes = kwargs.get("n_boxes", 3)
        box_numbers = kwargs.get("box_numbers", [1, 2, 3, 4, 5, 6, 7, 8, 9])
        # nth parameter: default is 0 (first box). To replicate current behavior (reward in second box), pass nth=1.
        nth = kwargs.get("nth", 0)
        kwargs["rule"] = "nth"
        kwargs["n_boxes"] = n_boxes
        kwargs["box_numbers"] = box_numbers
        kwargs["nth"] = nth

        chosen_boxes = np.random.choice(box_numbers, size=n_boxes, replace=False).tolist()
        reward_box = chosen_boxes[nth]
        env_params = copy.deepcopy(kwargs)
        env_params.update({"current_boxes": chosen_boxes})
        i_round = 0
        task_message = (
            f"Round {i_round+1}: " +
            " vs. ".join([f"Box {n}" for n in chosen_boxes]) +
            "?"
        )
        user_message = {
            "role": "user",
            "content": task_message,
            "done": False,
            "i_round": i_round,
            "box_reward": reward_box,
        }
        return env_params, user_message

    def get_rule_response(self, messages, last_choice):
        env_params = messages[0]["env_params"]
        assert env_params.get("rule") == "nth"
        last_user_message = messages[-2]
        if last_choice is not None:
            reward = 1.0 if last_choice == last_user_message["box_reward"] else 0.0
            prefix = f"Reward: {reward}\n\n"
        else:
            reward = 0.0
            prefix = "Format Error.\n\n"
        i_round_new = last_user_message["i_round"] + 1

        n_boxes = env_params.get("n_boxes", 3)
        box_numbers = env_params.get("box_numbers", [1, 2, 3, 4, 5, 6, 7, 8, 9])
        nth = env_params.get("nth", 0)
        chosen_boxes = np.random.choice(box_numbers, size=n_boxes, replace=False).tolist()
        reward_box = chosen_boxes[nth]
        task_message = (
            f"Round {i_round_new+1}: " +
            " vs. ".join([f"Box {n}" for n in chosen_boxes]) +
            "?"
        )
        response_message = {
            "role": "user",
            "content": prefix + task_message,
            "reward": reward,
            "i_round": i_round_new,
            "box_reward": reward_box,
        }
        return response_message

class NthValueRule():
    def __init__(self):
        pass

    def get_env_params_user_message(self, **kwargs):
        # Defaults:
        n_boxes = kwargs.get("n_boxes", 3)
        box_numbers = kwargs.get("box_numbers", [1, 2, 3, 4, 5, 6, 7, 8, 9])
        # nth_value: index in the sorted order. Default to the median index if not provided.
        nth_value = kwargs.get("nth_value", None)
        if nth_value is None:
            nth_value = (n_boxes - 1) // 2

        kwargs["rule"] = "nth_value"
        kwargs["n_boxes"] = n_boxes
        kwargs["box_numbers"] = box_numbers
        kwargs["nth_value"] = nth_value

        # Sample n_boxes without replacement:
        chosen_boxes = np.random.choice(box_numbers, size=n_boxes, replace=False).tolist()
        sorted_boxes = sorted(chosen_boxes)
        reward_box = sorted_boxes[nth_value]
        
        env_params = copy.deepcopy(kwargs)
        env_params.update({"current_boxes": chosen_boxes})
        i_round = 0
        task_message = (
            f"Round {i_round+1}: " +
            " vs. ".join([f"Box {n}" for n in chosen_boxes]) +
            "?"
        )
        user_message = {
            "role": "user",
            "content": task_message,
            "done": False,
            "i_round": i_round,
            "box_reward": reward_box,
        }
        return env_params, user_message

    def get_rule_response(self, messages, last_choice):
        env_params = messages[0]["env_params"]
        assert env_params.get("rule") == "nth_value"
        last_user_message = messages[-2]
        if last_choice is not None:
            reward = 1.0 if last_choice == last_user_message["box_reward"] else 0.0
            prefix = f"Reward: {reward}\n\n"
        else:
            reward = 0.0
            prefix = "Format Error.\n\n"
        
        i_round_new = last_user_message["i_round"] + 1
        n_boxes = env_params.get("n_boxes", 3)
        box_numbers = env_params.get("box_numbers", [1, 2, 3, 4, 5, 6, 7, 8, 9])
        nth_value = env_params.get("nth_value", (n_boxes - 1) // 2)
        
        chosen_boxes = np.random.choice(box_numbers, size=n_boxes, replace=False).tolist()
        sorted_boxes = sorted(chosen_boxes)
        reward_box = sorted_boxes[nth_value]
        task_message = (
            f"Round {i_round_new+1}: " +
            " vs. ".join([f"Box {n}" for n in chosen_boxes]) +
            "?"
        )
        response_message = {
            "role": "user",
            "content": prefix + task_message,
            "reward": reward,
            "i_round": i_round_new,
            "box_reward": reward_box,
        }
        return response_message

class OddSmallestEvenBiggestRule():
    def __init__(self):
        pass

    def get_env_params_user_message(self, **kwargs):
        # Defaults for this rule:
        n_boxess = kwargs.get("n_boxess", [2, 3, 4, 5])
        box_numbers = kwargs.get("box_numbers", [1, 2, 3, 4, 5, 6, 7, 8, 9])
        kwargs["rule"] = "odd_smallest_even_biggest"
        kwargs["n_boxess"] = n_boxess
        kwargs["box_numbers"] = box_numbers

        # Randomly choose the number of boxes from n_boxess:
        n_boxes = int(np.random.choice(n_boxess))
        # Randomly select n_boxes without replacement:
        chosen_boxes = np.random.choice(box_numbers, size=n_boxes, replace=False).tolist()
        # Reward depends on parity of the number of boxes:
        if n_boxes % 2 == 0:
            reward_box = max(chosen_boxes)
        else:
            reward_box = min(chosen_boxes)

        env_params = copy.deepcopy(kwargs)
        env_params.update({"current_boxes": chosen_boxes})
        i_round = 0
        task_message = (
            f"Round {i_round+1}: " +
            " vs. ".join([f"Box {n}" for n in chosen_boxes]) +
            "?"
        )
        user_message = {
            "role": "user",
            "content": task_message,
            "done": False,
            "i_round": i_round,
            "box_reward": reward_box,
        }
        return env_params, user_message

    def get_rule_response(self, messages, last_choice):
        env_params = messages[0]["env_params"]
        assert env_params.get("rule") == "odd_smallest_even_biggest"
        last_user_message = messages[-2]
        if last_choice is not None:
            reward = 1.0 if last_choice == last_user_message["box_reward"] else 0.0
            prefix = f"Reward: {reward}\n\n"
        else:
            reward = 0.0
            prefix = "Format Error.\n\n"
        i_round_new = last_user_message["i_round"] + 1

        n_boxess = env_params.get("n_boxess", [2, 3, 4, 5])
        box_numbers = env_params.get("box_numbers", [1, 2, 3, 4, 5, 6, 7, 8, 9])
        n_boxes = int(np.random.choice(n_boxess))
        chosen_boxes = np.random.choice(box_numbers, size=n_boxes, replace=False).tolist()
        if n_boxes % 2 == 0:
            reward_box = max(chosen_boxes)
        else:
            reward_box = min(chosen_boxes)
        task_message = (
            f"Round {i_round_new+1}: " +
            " vs. ".join([f"Box {n}" for n in chosen_boxes]) +
            "?"
        )
        response_message = {
            "role": "user",
            "content": prefix + task_message,
            "reward": reward,
            "i_round": i_round_new,
            "box_reward": reward_box,
        }
        return response_message


rule_instances={
    "circular": CircularRule(),
    "circular_stay": CircularStayRule(),
    "nth": NthRule(),
    "nth_value":NthValueRule(),
    "odd_smallest_even_biggest": OddSmallestEvenBiggestRule(),
}