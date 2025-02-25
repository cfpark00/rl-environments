import argparse
import os
import datasets
import shutil

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
    parser.add_argument("--n_rows", type=int, default=None, help="Number of rows")
    args = parser.parse_args()

    file_path_parquet=os.path.join(args.save_path,args.split+".parquet")
    file_path_json=os.path.join(args.save_path,args.split+".json")
    assert not os.path.exists(file_path_parquet), f"File already exists: {file_path_parquet}"
    assert not os.path.exists(file_path_json), f"File already exists: {file_path_json}"
    

    environment = env_classes[args.env_name]()
    if args.n_rows is None:
        dataset=environment.get_dataset()
    else:
        dataset=environment.get_dataset(n_rows=args.n_rows)
    ds=datasets.Dataset.from_list(dataset)
    print(ds)
    print(ds[0])

    os.makedirs(args.save_path,exist_ok=True)
    ds.to_parquet(file_path_parquet)
    ds.to_json(file_path_json)