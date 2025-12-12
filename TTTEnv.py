import numpy as np
import gymnasium
from gymnasium import spaces

# --- Game Constants ---
ROWS = 3
COLS = 3
EMPTY_CHAR = '.'
PLAYER_X = 'X'
PLAYER_O = 'O'

# --- State Class ---
class State:
    def __init__(self, board):
        self.board = np.array(board)

    def __eq__(self, other):
        return np.array_equal(self.board, other.board)

    def __hash__(self):
        return hash(self.board.tobytes())

# --- Game Logic Functions ---

def check_win(board, player):
    for i in range(ROWS):
        if all([board[i][j] == player for j in range(COLS)]):
            return True
        if all([board[j][i] == player for j in range(ROWS)]):
            return True
    
    if all([board[i][i] == player for i in range(ROWS)]):
        return True
    if all([board[i][ROWS - 1 - i] == player for i in range(ROWS)]):
        return True
        
    return False

def is_board_full(board):
    return not np.any(board == EMPTY_CHAR)

def is_end(board):
    return check_win(board, PLAYER_X) or check_win(board, PLAYER_O) or is_board_full(board)

def calculate_reward(board):
    if check_win(board, PLAYER_X):
        return 100
    if check_win(board, PLAYER_O):
        return -100
    if is_board_full(board):
        return 0
    return -1

# --- The Environment Class (Gymnasium API) ---

class TicTacToeEnv(gymnasium.Env):
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 1}

    def __init__(self):
        super(TicTacToeEnv, self).__init__()
        
        self.action_space = spaces.Discrete(ROWS * COLS) 
        
        self.observation_space = spaces.Box(
            low=0, high=2, shape=(ROWS * COLS,), dtype=np.int8
        )
        
        self.board = self._get_initial_board()
        self.current_player = PLAYER_X

    def _get_initial_board(self):
        return np.full((ROWS, COLS), EMPTY_CHAR, dtype=str)

    def _encode_board(self, board):
        encoded = np.zeros(ROWS * COLS, dtype=np.int8)
        mapping = {EMPTY_CHAR: 0, PLAYER_X: 1, PLAYER_O: 2}
        
        for i, char in enumerate(board.flatten()):
            encoded[i] = mapping[char]
        return encoded

    def _random_opponent_move(self):
        empty_spots = np.argwhere(self.board == EMPTY_CHAR)
        
        if len(empty_spots) > 0:
            move_coords = empty_spots[np.random.choice(len(empty_spots))]
            r, c = move_coords
            self.board[r, c] = PLAYER_O
            return True
        return False

    def step(self, action):
        r, c = action // COLS, action % COLS
        reward = 0
        terminated = False
        truncated = False
        
        # 1. Agent's Move (Player X)
        if self.board[r, c] != EMPTY_CHAR:
            reward = -1000
            terminated = True
        else:
            self.board[r, c] = PLAYER_X
            
            if check_win(self.board, PLAYER_X) or is_board_full(self.board):
                reward = calculate_reward(self.board)
                terminated = True
            
            # 2. Opponent's Move (Player O)
            if not terminated:
                if self._random_opponent_move():
                    reward = calculate_reward(self.board)
                    if is_end(self.board):
                        terminated = True
                else:
                    terminated = True

        return self._encode_board(self.board), reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = self._get_initial_board()
        return self._encode_board(self.board), {}

    def render(self):
        print("-------")
        for row in range(ROWS):
            print("|", end=" ")
            for col in range(COLS):
                print(self.board[row, col], end=" ")
            print("|")
        print("-------")

    def close(self):
        pass