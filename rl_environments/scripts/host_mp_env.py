import argparse

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import socket
import logging

from rl_environments import utils

from rl_environments.envs.examples_envs import ExampleEnv
from rl_environments.envs.banditboxes_env import BanditBoxesEnv
from rl_environments.envs.singlebanditboxes_env import SingleBanditBoxesEnv
from rl_environments.envs.guessnumber_env import GuessNumberEnv
from rl_environments.envs.guessnumberhotcold_env import GuessNumberHotColdEnv
from rl_environments.envs.tictactoerandom_env import TicTacToeRandomEnv
from rl_environments.envs.thinkactionjson_env import ThinkActionJsonEnv
from rl_environments.envs.maze_env import MazeEnv
from rl_environments.envs.ruledbanditboxes_env import RuledBanditBoxesEnv


env_classes={
    "example": ExampleEnv,
    #bandits
    "banditboxes": BanditBoxesEnv,
    "singlebanditboxes": SingleBanditBoxesEnv,
    #guess number
    "guessnumber": GuessNumberEnv,
    "guessnumberhotcold": GuessNumberHotColdEnv,
    #tictactoe
    "tictactoerandom": TicTacToeRandomEnv,
    #thinkactionjson
    "thinkactionjson": ThinkActionJsonEnv,
    #maze
    "maze": MazeEnv,
    #ruledbanditboxes
    "ruledbanditboxes": RuledBanditBoxesEnv
}

parser = argparse.ArgumentParser(description="Host an environment")
parser.add_argument("env_name", type=str, help="Name of the environment to host")
parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
parser.add_argument("--port", type=int, default=7000, help="Port")
parser.add_argument("--workers", type=int, default=32, help="Number of workers")
args = parser.parse_args()


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
environment = env_classes[args.env_name](logger=logger)

app = utils.create_app(environment)

if __name__ == "__main__":
    print("Hostname:", socket.gethostname())
    uvicorn.run(
        "rl_environments.scripts.host_mp_env:app",
        host=args.host,
        port=args.port,
        log_level="info",
        workers=args.workers,
    )
