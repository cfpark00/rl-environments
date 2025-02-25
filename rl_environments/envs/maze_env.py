from rl_environments.envs import BaseEnv
from pydantic import BaseModel, ValidationError
import copy
import json
import numpy as np
import re

# --- Helper functions for maze processing ---
def raw_coord_to_cell_coord(raw_coord):
    assert raw_coord[0] % 2 == 1 and raw_coord[1] % 2 == 1, "raw_coord must be odd to be a cell"
    return (raw_coord[0] // 2, raw_coord[1] // 2)

def cell_coord_to_raw_coord(cell_coord):
    return (2 * cell_coord[0] + 1, 2 * cell_coord[1] + 1)

def ascii_maze_to_2dlist(maze):
    n_rows = maze.count("\n")+1
    n_colss = [len(line) for line in maze.split("\n")]
    assert all(n_cols == n_colss[0] for n_cols in n_colss), "All rows must have the same number of columns"
    assert n_colss[0] == n_rows, "Maze must be square"
    return [list(line) for line in maze.split("\n")]

def maze_2dlist_to_cell_2dlist(maze2dlist):
    side=len(maze2dlist)
    assert side%2==1, "Maze must have odd side length"
    n_cells_side=(side-1)//2
    cells=[]
    for i in range(n_cells_side):
        row=[]
        for j in range(n_cells_side):
            row.append(maze2dlist[2*i+1][2*j+1])
        cells.append(row)
    return cells

def do_move(maze2dlist, loc, move, target=None):
    raw_loc = np.array(cell_coord_to_raw_coord(loc))
    proposal_direction = np.array({
        "left": [0, -1],
        "right": [0, 1],
        "up": [1, 0],
        "down": [-1, 0]
    }[move])
    proposed_loc = raw_loc + proposal_direction
    check = maze2dlist[proposed_loc[0]][proposed_loc[1]]
    if check == "W":  # move hit a wall
        return 0, loc
    elif check == "P":
        # Advance one more cell in the same direction.
        new_loc = proposed_loc + proposal_direction
        new_cell_coord = raw_coord_to_cell_coord(new_loc.tolist())
        if target is not None and all(a == b for a, b in zip(new_cell_coord, target)):
            return 2, new_cell_coord
        return 1, new_cell_coord
    else:
        raise ValueError("Unexpected cell value encountered in maze.")

def get_start_end(maze):
    cells=maze_2dlist_to_cell_2dlist(ascii_maze_to_2dlist(maze))
    side=len(cells)
    for i in range(side):
        for j in range(side):
            if cells[i][j]=="S":
                start=[i,j]
            if cells[i][j]=="E":
                end=[i,j]
    return start,end

# --- Pydantic response format for maze moves ---
class MazeResponseFormat(BaseModel):
    moves: list[str]

# --- Maze Environment Implementation ---
class MazeEnv(BaseEnv):
    """
    system_prompt = (
        'The user will give you a maze to solve where the walls are marked as "W" and paths are marked as "P". '
        'The start position is "S" and the goal position is "E".\n'
        'After you propose moves, you will be given a per-move result: 0 if the move hit a wall (and further moves are ignored), '
        '1 if it succeeded, and 2 if you reached the goal.\n'
        'Output the sequence of moves (left, right, up, down) to perform like this example:\n'
        '<think>Your internal reasoning here</think><action> {"moves": ["left", "left", "up"]} </action>\n'
    )
    system_prompt = (
        'The user will give you a maze presented as coordinates and in which direction they have walls/paths. '
        'They will also give \"loc\", your current location. '
        'After you propose moves, you will be given a per-move result: 0 if the move hit a wall (and further moves are ignored), '
        '1 if it succeeded, and 2 if you reached the goal.\n'
        'Output the sequence of moves (left, right, up, down) to perform like this example:\n'
        '<think>Your internal reasoning here</think><action> {"moves": ["left", "left", "up"]} </action>\n'
    )
    """
    system_prompt = (
        'The user will give you a maze presented as coordinates of cells and in whether they have walls (W) or paths (P) in each direction. '
        'They will also give \"loc\", your current location. '
        'You should propose moves and you will be given a per-move result: 0 if the move hit a wall (and further moves are ignored), '
        '1 if it succeeded, and 2 if you reached the goal.\n'
        'The available moves are:\n'
        ' - "left": (0,-1)\n'
        ' - "right": (0,+1)\n'
        ' - "up": (+1,0)\n'
        ' - "down": (-1,0)\n\n'
        'Output in a format like this example:\n'
        '<think>Your internal reasoning here</think>\n<action> {"moves": ["left", "left", "up"]} </action>\n'
    )
    format_error_prompt = (
        'Format Error.'
    )

    default_params = {
        "max_turns": 16,
        "reward_version": "v1",
    }

    def __init__(self, logger=None, **kwargs):
        super().__init__(logger=logger)
        self.params = copy.deepcopy(MazeEnv.default_params)
        self.params.update(kwargs)
        self.system_prompt = MazeEnv.system_prompt

    def get_dataset(self, n_rows=1):
        data = [self.get_data_sample()]
        return data

    def translate_maze(self, maze_raw):
        ascii_maze=maze_raw.replace("#","W")
        ascii_maze=ascii_maze.replace("X"," ")
        n_side=len(ascii_maze.split("\n")[0])
        #replace path(" ") with "P"
        rp_locs=[i*n_side+j for i in range(0,n_side,2) for j in range(1,n_side,2)]
        rp_locs+=[i*n_side+j for i in range(1,n_side,2) for j in range(0,n_side,2)]
        ascii_maze_l=list(ascii_maze.replace("\n",""))
        for loc in rp_locs:
            if ascii_maze_l[loc]!="W":
                ascii_maze_l[loc]="P"
        #insert \n at the end of each row
        for i in range(n_side-1,0,-1):
            ascii_maze_l.insert(i*n_side,"\n")
        ascii_maze="".join(ascii_maze_l)
        return ascii_maze

    def get_maze_text(self, maze):
        """
        maze2dlist=ascii_maze_to_2dlist(maze)
        cells=[]
        side=len(maze2dlist)
        grid_n=(side-1)//2
        wallvecs={
            "left": [0,-1],
            "right": [0,1],
            "up": [1,0],
            "down": [-1,0]
        }
        for i in range(grid_n):
            for j in range(grid_n):
                maze_coord=(2*i+1,2*j+1)
                centerchar=maze2dlist[maze_coord[0]][maze_coord[1]]
                assert centerchar!="W", "Walls should not be included in the maze text"
                walls=[]
                paths=[]
                for name,vec in wallvecs.items():
                    new_coord=(maze_coord[0]+vec[0],maze_coord[1]+vec[1])
                    sidechar=maze2dlist[new_coord[0]][new_coord[1]]
                    if sidechar=="W":
                        walls.append(name)
                    elif sidechar=="P":
                        paths.append(name)
                    else:
                        raise ValueError(f"Unexpected character {sidechar} encountered in maze")
                cell=f"[Coord: ({i}, {j}), Walls: {walls}, Paths: {paths}]"
                cells.append(cell)
        cells_text=json.dumps(cells,indent=2)
        return cells_text
        """
        maze2dlist=ascii_maze_to_2dlist(maze)
        cells=[]
        side=len(maze2dlist)
        grid_n=(side-1)//2
        movevecs={
            "left": [0,-1],
            "right": [0,1],
            "up": [1,0],
            "down": [-1,0]
        }
        for i in range(grid_n):
            for j in range(grid_n):
                maze_coord=(2*i+1,2*j+1)
                centerchar=maze2dlist[maze_coord[0]][maze_coord[1]]
                assert centerchar!="W", "Walls should not be included in the maze text"
                moveschars={}
                for name,vec in movevecs.items():
                    new_coord=(maze_coord[0]+vec[0],maze_coord[1]+vec[1])
                    sidechar=maze2dlist[new_coord[0]][new_coord[1]]
                    moveschars[name]=sidechar
                cell=[(i,j), moveschars["left"], moveschars["right"], moveschars["up"], moveschars["down"]]
                cells.append(cell)
        header=["Coordinates", "left (0, -1)", "right (0, 1)", "up (1, 0)", "down (-1, 0)"]
        #format into csv
        cells_text="; ".join(header)+"\n"
        for cell in cells:
            cells_text+="; ".join([str(cell[0])]+[str(cell[i]) for i in range(1,5)])+"\n"
        return cells_text


    def get_data_sample(self, **kwargs):
        maze_raw=kwargs.get('maze_raw', '#####\n# #S#\n# #X#\n#EXX#\n#####')
        maze=self.translate_maze(maze_raw)
        maze_text=self.get_maze_text(maze)
        start,end=get_start_end(maze)
        env_params = {'maze': maze, 'maze_text': maze_text, 'start': start, 'end': end, 'max_turns': self.params['max_turns'], "reward_version": self.params["reward_version"]}
        # The initial user message provides the maze and starting location.
        user_message_content = json.dumps({"maze": maze_text, "loc": str(tuple(start)), "goal": str(tuple(end))},indent=2)
        messages = [
            {"role": "system", "content": self.system_prompt, "env_params": env_params},
            {"role": "user", "content": user_message_content, "loc": start, "done": False},
        ]
        return {"messages": messages}

    @staticmethod
    def get_env_response(messages):
        """
        Process the assistant's move in the maze:
          - Validates the output format (requires <think> and <action> tags).
          - Extracts the sequence of moves from the JSON inside <action>...</action>.
          - Applies each move (using do_move) until a move fails or the goal is reached.
          - Returns updated move results and the new location.
        """
        env_params = messages[0]["env_params"]
        maze = env_params["maze"]
        end = env_params["end"]
        max_turns = env_params["max_turns"]
        maze2dlist = ascii_maze_to_2dlist(maze)
        assert messages[0]["role"] == "system"
        assert messages[-1]["role"] == "assistant"
        assert messages[-2]["role"] == "user"

        # The previous user message should contain the current location.
        loc = copy.deepcopy(messages[-2]["loc"])
        done = False
        # Optionally, limit the number of moves (for example, after 32 cycles).
        if (len(messages) - 1) // 2 >= max_turns:
            done = True

        # Retrieve the last assistant message.
        last_assistant_message = messages[-1]
        content = last_assistant_message["content"]

        # Validate the format by checking for <think> and <action> tags.
        think_matches = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
        action_matches = re.search(r"<action>(.*?)</action>", content, re.DOTALL)
        if not (think_matches and action_matches and len(think_matches.groups()) == 1 and len(action_matches.groups()) == 1):
            return {
                "role": "user",
                "content": MazeEnv.format_error_prompt,
                "format_error": True,
                "done": done,
                "loc": loc,
                "locs": [loc]
            }

        action_str = action_matches.group(1).strip()
        try:
            parsed_response = MazeResponseFormat.model_validate_json(action_str)
            moves = parsed_response.moves
        except ValidationError:
            return {
                "role": "user",
                "content": MazeEnv.format_error_prompt,
                "format_error": True,
                "done": done,
                "loc": loc,
                "locs": [loc]
            }
        if not all([move in ["left", "right", "up", "down"] for move in moves]):
            return {
                "role": "user",
                "content": MazeEnv.format_error_prompt,
                "format_error": True,
                "done": done,
                "loc": loc,
                "locs": [loc]
            }

        response_message = {"role": "user"}
        json_response = {}
        move_results = []
        locs = []

        # Process each move in sequence.
        for move in moves:
            success, loc = do_move(maze2dlist, loc, move, target=end)
            move_results.append(success)
            locs.append(loc)
            if success == 0:  # hit a wall; stop further moves
                break
            if success == 2:  # reached the goal
                response_message["success"] = True
                done = True
                break

        json_response["move_results"] = move_results
        json_response["loc"] = tuple(loc)
        json_response_str = json.dumps(json_response)

        response_message["content"] = json_response_str
        response_message["done"] = done
        response_message["loc"] = loc
        response_message["last_moves"] = moves
        response_message["last_locs"] = locs
        response_message["move_results"] = move_results
        return response_message

    @staticmethod
    def get_reward(messages):
        """
        Example reward scheme:
          - A penalty of -1.0 for any format error.
          - A bonus of +10.0 for reaching the goal.
        """
        env_params = messages[0]["env_params"]
        reward_version = env_params.get("reward_version", "v1")
        start, end = env_params["start"], env_params["end"]
        max_turns = env_params["max_turns"]

        if reward_version=="v1":
            success_coeff=10.0
            turns_coeff=5.0
            moves_coeff=10.0
            distance_coeff=None
        elif reward_version=="v1.1":
            success_coeff=3.0
            turns_coeff=2.0
            moves_coeff=2.0
            distance_coeff=3.0


        reward = 0.0
        success=False
        n_tot_moves_proposed=0
        format_errors=[]
        n_assistant_messages=0
        for message in messages:
            format_error=message.get("format_error", False)
            format_errors.append(format_error)
            n_tot_moves_proposed+=len(message.get("last_moves", []))
            if message.get("success", False):
                success=True
            if message["role"]=="assistant":
                n_assistant_messages+=1
        
        reward+=-1.0*sum(format_errors)
        reward+=success_coeff if success else 0.0
        if success:
            #up to turns_coeff points for reaching the goal with the least number of turns
            reward+=turns_coeff*max(0, max_turns-n_assistant_messages)/(max_turns-1)
            #up to moves_coeff points for reaching the goal with the least number of moves
            reward+=moves_coeff*max(0,10.0-n_tot_moves_proposed)/(10.0-1)
        if distance_coeff is not None:
            final_loc=messages[-1]["loc"]
            distance=np.sqrt((end[0]-final_loc[0])**2+(end[1]-final_loc[1])**2)
            reward+=distance_coeff*max(0,3.0-distance)/3.0
        return reward

#<think> </think><action> {"moves": ["down","left"]} </action>
