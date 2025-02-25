from rl_environments import utils
from rl_environments.envs import BaseEnv
from pydantic import BaseModel, ValidationError
import copy
import json
import numpy as np
import re  # for regex parsing

def board_to_string(board):
    return "\n".join("".join(row) for row in board)

def string_to_board(board_str):
    rows = board_str.strip().split("\n")
    return [list(row) for row in rows]

def is_valid_board(board):
    if len(board) != 3:
        return False
    for row in board:
        if len(row) != 3:
            return False
        for cell in row:
            if cell not in {"-", "X", "O"}:
                return False
    return True

def check_win(board, marker):
    # Check rows, columns, and diagonals for a win.
    for i in range(3):
        if all(cell == marker for cell in board[i]):
            return True
        if all(board[j][i] == marker for j in range(3)):
            return True
    if board[0][0] == marker and board[1][1] == marker and board[2][2] == marker:
        return True
    if board[0][2] == marker and board[1][1] == marker and board[2][0] == marker:
        return True
    return False

def board_full(board):
    return all(cell != "-" for row in board for cell in row)

# We no longer validate the thought process with pydantic;
# only the JSON answer within <action>...</action> is validated.
class ResponseFormat(BaseModel):
    board_state: str

class TicTacToeRandomEnv(BaseEnv):
    system_prompt = (
        "You are playing Tic-Tac-Toe. The board is represented by '-' for empty spaces and 'O' or 'X' for markers.\n"
        "The user will tell you what marker you are playing and who starts.\n"
        "Respond in the following format:\n"
        "<think>Your thought process here (can be any reasoning you want)</think>"
        "<action>{{\"board_state\": \"current board layout with rows separated by newline, e.g. 'X--\\n-O-\\n--X'\"}}</action>"
    )
    format_error_prompt = (
        "Format Error. Please respond in the following format:\n"
        "<think>Your thought process here (can be any reasoning you want)</think>"
        "<action>{\"board_state\": \"current board layout with rows separated by newline, e.g. 'X--\\n-O-\\n--X'\"}</action>"
    )

    default_params = {
    }

    def __init__(self, logger=None, **kwargs):
        super().__init__(logger=logger)
        self.params = copy.deepcopy(TicTacToeRandomEnv.default_params)
        self.params.update(kwargs)
        self.system_prompt = TicTacToeRandomEnv.system_prompt.format()

    def get_dataset(self, n_rows=1):
        # For Tic-Tac-Toe, a single starting state is sufficient.
        data = [self.get_data_sample()]
        return data

    def get_data_sample(self, **kwargs):
        symbol = kwargs.get("symbol", ["X", "O"][np.random.choice(2)])
        user_start = kwargs.get("user_start", np.random.random() < 0.5)
        user_symbol = "O" if symbol == "X" else "X"
        board = [["-" for _ in range(3)] for _ in range(3)]
        if user_start:
            row, col = np.random.choice(3), np.random.choice(3)
            board[row][col] = user_symbol
            user_message_content = (
                f"You are playing as '{symbol}' and I will play first as '{user_symbol}'. "
                f"Your move should be reflected by placing {symbol} in one empty cell. "
                f"Here is the board state:\n"
                f"{{\"board_state\": \"{board_to_string(board)}\"}}"
            )
        else:
            user_message_content = (
                f"You are playing as '{symbol}' and it's your turn. "
                f"Your move should be reflected by placing {symbol} in one empty cell. "
                f"Here is the board state:\n"
                f"{{\"board_state\": \"{board_to_string(board)}\"}}"
            )
        env_params = {"symbol": symbol, "user_symbol": user_symbol, "user_start": user_start, "initial_board": board}
        messages = [
            {"role": "system", "content": self.system_prompt, "env_params": env_params},
            {"role": "user",
             "content": user_message_content,
             "board": board,
             "done": False}
        ]
        return {"messages": messages}

    @staticmethod
    def get_env_response(messages):
        """
        Process the assistant's move:
          - Validates the output format and board state.
          - Checks that the assistant made a valid move (exactly one assistant marker added to an empty cell).
          - If valid, applies the move, then the environment makes a random move as 'O' (if game not over).
          - Checks for win/draw conditions.
          - Returns the updated board state in the required format.
        """
        # Retrieve the last assistant message.
        env_params = messages[0]["env_params"]
        last_assistant_message = utils.get_last_assistant_message(messages)
        last_content = last_assistant_message["content"]

        # Enforce the existence of the <think></think> tags.
        if not re.search(r"<think>.*?</think>", last_content, re.DOTALL):
            return {
                "role": "user",
                "content": TicTacToeRandomEnv.format_error_prompt,
                "done": True,
                "format_error": True,
            }

        # Extract the JSON answer from within <action>...</action> tags.
        match = re.search(r"<action>(.*?)</action>", last_content, re.DOTALL)
        if not match:
            return {
                "role": "user",
                "content": TicTacToeRandomEnv.format_error_prompt,
                "done": True,
                "format_error": True,
            }
        json_answer = match.group(1).strip()
        try:
            parsed_response = ResponseFormat.model_validate_json(json_answer)
            board_str = parsed_response.board_state
        except ValidationError:
            return {
                "role": "user",
                "content": TicTacToeRandomEnv.format_error_prompt,
                "done": True,
                "format_error": True,
            }
        try:
            current_board = string_to_board(board_str)
        except Exception:
            return {
                "role": "user",
                "content": TicTacToeRandomEnv.format_error_prompt,
                "done": True,
                "format_error": True,
            }
        # Validate board structure.
        if not is_valid_board(current_board):
            return {
                "role": "user",
                "content": TicTacToeRandomEnv.format_error_prompt,
                "done": True,
                "format_error": True,
            }

        # Get previous board state from env_params.
        assert messages[-1]["role"] == "assistant"
        assert messages[-2]["role"] == "user", messages
        prev_board = messages[-2]["board"]
        assistant_symbol = env_params["symbol"]
        user_symbol = env_params["user_symbol"]
        # Determine the difference: assistant should have placed one assistant_symbol in an empty cell.
        diff = []
        for i in range(3):
            for j in range(3):
                if prev_board[i][j] != current_board[i][j]:
                    diff.append((i, j, prev_board[i][j], current_board[i][j]))
        if len(diff) != 1 or diff[0][2] != "-" or diff[0][3] != assistant_symbol:
            return {
                "role": "user",
                "content": "Invalid move.",
                "done": True,
                "format_error": True,
            }

        # Check if the assistant won.
        if check_win(current_board, assistant_symbol):
            return {
                "role": "user",
                "content": "Assistant Wins!",
                "done": True,
                "result": "assistant_win",
                "board": current_board,
            }
        # Check if board is full (draw) after assistant move.
        if board_full(current_board):
            return {
                "role": "user",
                "content": "Draw!",
                "done": True,
                "result": "draw",
                "board": current_board,
            }
        # Environment (playing as 'O') makes a random move.
        empty_cells = [(i, j) for i in range(3) for j in range(3) if current_board[i][j] == "-"]
        if empty_cells:
            i, j = empty_cells[np.random.choice(len(empty_cells))]
            current_board[i][j] = user_symbol
        # Check if environment wins.
        if check_win(current_board, user_symbol):
            return {
                "role": "user",
                "content": "User Wins!",
                "done": True,
                "result": "user_win",
                "board": current_board,
            }
        # Check for draw after environment move.
        if board_full(current_board):
            return {
                "role": "user",
                "content": "Draw!",
                "done": True,
                "result": "draw",
                "board": current_board,
            }
        return {
            "role": "user",
            "content": json.dumps({"board_state": board_to_string(current_board)}),
            "done": False,
            "board": current_board,
        }

    @staticmethod
    def get_reward(messages):
        """
        Reward scheme:
          - Format errors: -1.0 penalty each.
          - Completing the game without errors: +5.0 reward.
          - Additional bonus for a win: extra +10.0.
        """
        env_params = messages[0]["env_params"]
        reward = 0.0
        for message in messages:
            if message["role"] == "user":
                if "format_error" in message and message["format_error"]:
                    reward -= 1.0
            if "result" in message:
                if message["result"] == "assistant_win":
                    reward += 5.0
                elif message["result"] == "draw":
                    reward += 3.0
                elif message["result"] == "user_win":
                    reward += 1.0  # still rewarded for playing until the end
        return reward
