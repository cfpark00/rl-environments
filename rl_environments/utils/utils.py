import omegaconf
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


def load_config(config_path):
    return omegaconf.OmegaConf.load(config_path)


def create_app(env):
    # Create the FastAPI app
    app = FastAPI()

    @app.post("/get_env_response_batched")
    async def get_env_response(request: Request):
        data = await request.json()
        response = env.get_env_response_batched(
            messages_b=data.get("messages_b"),
            hidden_params_b=data.get("hidden_params_b"),
        )
        return JSONResponse(response)

    @app.post("/get_reward_batched")
    async def get_reward(request: Request):
        data = await request.json()
        reward = env.get_reward_batched(
            messages_b=data.get("messages_b"),
            hidden_params_b=data.get("hidden_params_b"),
        )
        return JSONResponse(reward)

    @app.post("/get_data_sample")
    async def get_data_sample(request: Request):
        data = await request.json()
        response = env.get_data_sample(**data)
        return JSONResponse(response)

    return app
