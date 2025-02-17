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
ENV_NAME = "your_env_name"

def get_response_message(messages,hidden_params):
    """
    Takes in a list of messages and hidden parameters and returns the response message.

    Args:
    Returns:
    response_message: {"role":"user","content":"<message_content>", "done":True/False}
    """
    return {"role":"user","content":"You are interacting with a template environment. Maybe mess up the chat template to tell the user.","done":False}

def compute_reward(messages,hidden_params):
    reward=0.0
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
    text = data.get("text")
    hidden_params = data.get("hidden_params")
    env_response=environment.get_env_response(text, hidden_params)
    return JSONResponse(env_response)

@app.post("/get_reward")
async def get_reward(request: Request):
    data = await request.json()
    text=data.get("text")
    hidden_params=data.get("hidden_params")
    reward=environment.get_reward(text, hidden_params)
    return JSONResponse({"reward": reward})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5000, help="Port number for the server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="IP for the server")
    parser.add_argument("--workers", type=int, default=32, help="Number of workers for the server")
    args = parser.parse_args()

    print(socket.gethostname())

    uvicorn.run(ENV_NAME+":app", host=args.host, port=args.port, log_level="info", workers=args.workers)
