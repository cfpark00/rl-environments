import argparse
import os
import datasets

from rl_environments.envs.examples_envs import ExampleEnv
from rl_environments.envs.banditboxes_env import BanditBoxesEnv
from rl_environments.envs.singlebanditboxes_env import SingleBanditBoxesEnv
from rl_environments.envs.guessnumber_env import GuessNumberEnv

env_classes={
    "example": ExampleEnv,
    #bandits
    "banditboxes": BanditBoxesEnv,
    "singlebanditboxes": SingleBanditBoxesEnv,
    #guess number
    "guessnumber": GuessNumberEnv,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Host an environment")
    parser.add_argument("env_name", type=str, help="Name of the environment to host")
    parser.add_argument("save_path", type=str, help="Path to save the environment")
    parser.add_argument("split", type=str, help="Split to save")
    parser.add_argument("--n_rows", type=int, default=128, help="Number of rows")
    args = parser.parse_args()

    assert not os.path.exists(args.save_path), f"Path {args.save_path} already exists"

    environment = env_classes[args.env_name]()
    dataset=environment.get_dataset(n_rows=args.n_rows)
    ds=datasets.Dataset.from_list(dataset)
    print(ds)
    #ds.to_parquet(os.path.join(local_dir,f"{split}.parquet"))
    #ds.to_json(os.path.join(local_dir,f"{split}.json"))