###### HELPER FUNCTIONS ######
def get_n_assistant_messages(messages):
    n_assistant_messages = 0
    for message in messages:
        if message["role"] == "assistant":
            n_assistant_messages += 1
    return n_assistant_messages

def get_last_assistant_message(messages):
    if messages[-1]["role"]=="user":
        messages=messages[:-1]
    last_message=messages[-1]
    assert last_message["role"]=="assistant"
    return last_message

###### IMPLEMENT ######
ENV_NAME = "bandit_boxes_env"

format_description=(
    "Reasoning: {reasoning (string)}\n"
    "Final Choice: {choice (integer)}"
)

import numpy as np

def make_task_message(i_round, n_boxes):
    return f"Round {i_round+1}: "+" vs. ".join([f"Box {i}" for i in range(1,n_boxes+1)])+"?"

def make_dataset(n_rows=128,n_boxes=2,max_games=16):
    system_prompt=(
        f"You will be choosing a box among {n_boxes} boxes and potentially receive a reward. "
        "There might be some hidden rules that you can exploit to maximize your reward. "
        "Try to maximize the reward you get over multiple interactions. "
        "Response Format:\n"+format_description
    )
    data=[]
    for idx in range(n_rows):
        p_binoms=np.random.rand(n_boxes)
        #normalize so the sum is 1
        p_binoms=p_binoms/np.sum(p_binoms)
        messages=[
            {"role":"system","content":system_prompt},
            {"role":"user","content":make_task_message(0,n_boxes)}
        ]
        hidden_params={
            "p_binoms":p_binoms.tolist(),
            "max_games":max_games
        }
        datum={
            "prompt":messages,
            "hidden_params":hidden_params,
        }
        data.append(datum)
    return data

def get_choice(content):
    try:
        assert "Reasoning: " in content
        assert "Final Choice: " in content
        choice_str=content.split("Final Choice: ")[1].strip()
        choice_str=choice_str.replace(".","").replace("Box","")
        choice=int(choice_str)
    except:
        choice=None
    return choice

def get_response_message(messages,hidden_params):
    """
    Takes in a list of messages and hidden parameters and returns the response message.

    Args:
    Returns:
    response_message: {"role":"user","content":"<message_content>", "done":True/False}
    """
    p_binoms=hidden_params["p_binoms"]
    max_games=hidden_params["max_games"]

    last_assistant_message=get_last_assistant_message(messages)
    last_content=last_assistant_message["content"]
    last_choice=get_choice(last_content)
    #give reward
    if last_choice is not None:
        if last_choice in list(range(1,len(p_binoms)+1)):
            reward=int(np.random.binomial(1,p_binoms[last_choice-1]))
            feedback_message=f"Reward: {reward}\n\n"
        else:
            reward=0
            feedback_message="Please choose a valid box number. Reward: 0\n\n"
    else:
        reward=0
        feedback_message="Please respect the response format:\n"+format_description+"\nReward: 0\n\n"

    n_boxes=len(p_binoms)
    n_assistant_messages=get_n_assistant_messages(messages)
    i_round=n_assistant_messages#i_round is 0-indexed
    task_message=make_task_message(i_round=i_round,n_boxes=n_boxes)
    done=i_round>=max_games

    response_message={
        "role":"user",
        "content":feedback_message+task_message,
        "done":done
    }
    return response_message

def compute_reward(messages,hidden_params):
    reward=0.0
    for message in messages:
        if message["role"]=="user":
            content=message["content"]
            if "Reward: " in content:#the first user message will not have a reward
                split=content.split("Reward: ")
                assert len(split)==2
                reward_str=split[1].split("\n")[0].strip()
                reward+=float(reward_str)
    return reward


######### DO NOT MODIFY ANYTHING BELOW THIS LINE (Unless you know what you are doing) #########

import argparse
import re
import uvicorn
import logging
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import socket

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_messages(text):
    pattern = r"<\|im_start\|>(\w+)\n(.*?)<\|im_end\|>"
    matches = re.findall(pattern, text, re.DOTALL)
    
    messages = []
    for role, content in matches:
        messages.append({
            "role": role,
            "content": content.strip()  # Remove any extra whitespace or newlines
        })
    return messages

class Environment:#template
    def __init__(self,wrong_template_reward=0.0):
        self.wrong_template_reward=wrong_template_reward

    def get_env_response(self, text, hidden_params):
        env_response=[]
        for i in range(len(text)):
            try:
                messages = get_messages(text[i])
                logger.info("\tGot messages [%d]: %s", i, messages)
                env_response_message=get_response_message(messages,hidden_params[i])
                env_response.append({"env_response_message":env_response_message})
                logger.info("\tProduced response [%d]: %s", i, env_response_message)
            except Exception as e:
                logger.error("\tError processing text[%d]: %s", i, e)
                env_response.append({"env_response_message":None})
        return env_response
    
    def get_reward(self, text, hidden_params):
        rewards=[]
        for i in range(len(text)):
            try:
                messages = get_messages(text[i])
                #save text ./text_example.txt when i==0
                if i==0:
                    with open("./text_example.txt","w") as f:
                        f.write(text[i])
                logger.info("\tGot messages [%d]: %s", i, messages)
                reward=compute_reward(messages,hidden_params[i])
                rewards.append(reward)
                logger.info("\tProduced reward [%d]: %s", i, reward)
            except Exception as e:
                logger.error("\tError processing text[%d]: %s", i, e)
                rewards.append(self.wrong_template_reward)
        return rewards

#################

app = FastAPI()
environment = Environment(wrong_template_reward=0.0)

@app.post("/get_env_response")
async def get_env_response(request: Request):
    data = await request.json()
    env_response=environment.get_env_response(text=data.get("text"), hidden_params=data.get("hidden_params"))
    return JSONResponse(env_response)

@app.post("/get_reward")
async def get_reward(request: Request):
    data = await request.json()
    reward=environment.get_reward(text=data.get("text"), hidden_params=data.get("hidden_params"))
    return JSONResponse({"reward": reward})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5000, help="Port number for the server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="IP for the server")
    parser.add_argument("--workers", type=int, default=32, help="Number of workers for the server")
    args = parser.parse_args()

    print(socket.gethostname())

    uvicorn.run(ENV_NAME+":app", host=args.host, port=args.port, log_level="info", workers=args.workers)
