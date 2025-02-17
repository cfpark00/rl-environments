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

def get_proposition(content):
    try:
        proposition_str=content.split("Proposition: ")[1].strip()
        proposition_str=proposition_str.replace(".","")
        proposition=int(proposition_str)
    except:
        proposition=None
    return proposition

def get_response_message(messages,hidden_params):
    number=hidden_params["number"]
    max_turns=hidden_params["max_turns"]
    last_message=messages[-1]
    assert last_message["role"]=="assistant"

    n_turns=(len(messages)-1)//2#-1 for system prompt, //2 for user parts
    if n_turns>max_turns:
        return {"role":"user","content":"Max turns exceeded","done":True}

    content=last_message["content"]
    if "Reasoning:" not in content or "Proposition: " not in content:
        return {"role":"user","content":"Please stick to the format:\nReasoning: {reasoning (string)}\nProposition: {guess (integer)}","done":False}
    proposition=get_proposition(content)
    if proposition is None:
        return {"role":"user","content":"Please provide a valid integer proposition.","done":False}
    if proposition==number:
        return {"role":"user","content":"Correct!","done":True}
    elif proposition<number:
        return {"role":"user","content":"Higher!","done":False}
    else:
        return {"role":"user","content":"Lower!","done":False}

def compute_reward(messages,hidden_params):
    #if last message is user, remove it
    if messages[-1]["role"]=="user":
        messages=messages[:-1]
    number=hidden_params["number"]
    max_turns=hidden_params["max_turns"]
    last_message=messages[-1]
    assert last_message["role"]=="assistant"

    reward=0.0
    content=last_message["content"]
    proposition=get_proposition(content)
    #guess rewards
    if proposition is not None:
        reward+=max(0.0,10.0-abs(proposition-number))
    #length
    n_turns=(len(messages)-1)//2#-1 for system prompt, //2 for user parts
    reward+=max(0.0,max_turns-n_turns)#reward for finishing early
    return reward

class Environment:
    def __init__(self):
        pass

    def get_env_response(self, text, hidden_params):
        assert all(["number" in hp for hp in hidden_params])
        assert all(["max_turns" in hp for hp in hidden_params])

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
        assert all(["number" in hp for hp in hidden_params])

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
                rewards.append(0.0)#-10 for wrong message format
        return rewards

#################

app = FastAPI()
environment = Environment()

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

    uvicorn.run("guessnumber_env:app", host=args.host, port=args.port, log_level="info", workers=args.workers)
