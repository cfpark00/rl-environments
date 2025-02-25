import abc


class BaseEnv(abc.ABC):

    def __init__(self, logger=None):
        self.logger = logger

    def get_dataset(self, n_rows, **kwargs):
        raise NotImplementedError

    def get_env_response_batched(self, messages_batched):
        env_response_b = []
        batch_size = len(messages_batched)
        for i in range(batch_size):
            if self.logger is not None:
                self.logger.info(
                    "\tProcessing messages [%d/%d]: %s",
                    i + 1,
                    batch_size,
                    messages_batched[i],
                )
            env_response_message = self.get_env_response(
                messages_batched[i]
            )
            if self.logger is not None:
                self.logger.info(
                    "\tProduced response [%d/%d]: %s",
                    i + 1,
                    batch_size,
                    env_response_message,
                )
            env_response_b.append(env_response_message)
        return env_response_b

    def get_reward_batched(self, messages_batched):
        reward_b = []
        batch_size = len(messages_batched)
        for i in range(batch_size):
            if self.logger is not None:
                self.logger.info(
                    "\tProcessing messages [%d/%d]: %s",
                    i + 1,
                    batch_size,
                    messages_batched[i],
                )
            reward = self.get_reward(messages_batched[i])
            if self.logger is not None:
                self.logger.info(
                    "\tProduced reward [%d/%d]: %s", i + 1, batch_size, reward
                )
            reward_b.append(reward)
        return reward_b

    @abc.abstractmethod
    def get_env_response(self, messages):
        raise NotImplementedError

    @abc.abstractmethod
    def get_reward(self, messages):
        raise NotImplementedError

    def host(self, host, port, workers=1):
        import uvicorn
        from fastapi import FastAPI, Request
        from fastapi.responses import JSONResponse
        import logging

        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

        self.logger = logger

        """Host the environment as a FastAPI service."""
        app = FastAPI()
        assert workers == 1, "To host multiple processes, use the host.py script."

        @app.post("/get_env_response_batched")
        async def get_env_response(request: Request):
            data = await request.json()
            response = self.get_env_response_batched(
                messages_batched=data.get("messages_batched"),
            )
            return JSONResponse(response)

        @app.post("/get_reward_batched")
        async def get_reward(request: Request):
            data = await request.json()
            reward = self.get_reward_batched(
                messages_batched=data.get("messages_batched"),
            )
            return JSONResponse(reward)

        @app.post("/get_data_sample")
        async def get_data_sample(request: Request):
            data = await request.json()
            response = self.get_data_sample(**data)
            return JSONResponse(response)

        # Start the Uvicorn server with the given host, port, and number of workers.
        uvicorn.run(app, host=host, port=port, log_level="info", workers=workers)
