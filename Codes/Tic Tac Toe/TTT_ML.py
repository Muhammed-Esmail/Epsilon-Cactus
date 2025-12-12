import os
import numpy as np
import random
import time
from copy import deepcopy

ROWS = 3
COLS = 3
EPISODE_LENGTH = 100
PRINT_FREQUENCY = 0.2


class state:

    def __init__(self, board):
        self.board = board

    def __eq__(self, other):
        for i in range(ROWS):
            for j in range(COLS):
                if(self.board[i][j] != other.board[i][j]):
                    return False
        return True
    
    


start_board = np.array([
    ['.', '.', '.'],
    ['.', '.', '.'],
    ['.', '.', '.']
])

starting_State = state(start_board)

# Define Unicode Characters
WALL_CHAR = '█'
PATH_CHAR = ' '
START_CHAR = 'S'
END_CHAR = 'E'
VISITED_PATH_CHAR = '.'
CURRENT_POS_CHAR = '@' 
VISITED_VISUAL = '• '

def print_board_viz(current_board):
    """
    Prints the 3x3 Tic-Tac-Toe board using a clean format.
    
    current_board is expected to be a 3x3 NumPy array of strings ('.', 'X', 'O').
    """
    print("-------")
    for row in range(ROWS):
        print("|", end=" ")
        for col in range(COLS):
            # Print the cell content ('.', 'X', or 'O')
            print(current_board[row][col], end=" ")
        print("|")
    print("-------")


# dx = [-1, 1, 0, 0]
# dy = [0, 0, -1, 1]


def inBounds(x, y):
    return x <= ROWS and y <= COLS and x >= 1 and y >= 1

def badMove(state, x, y):
    return not inBounds(x,y) or state.board[x][y] != '.'

def check_win(board, player):
    # Check rows and columns
    for i in range(ROWS):
        # Check rows
        if all([board[i][j] == player for j in range(COLS)]):
            return True
        # Check columns
        if all([board[j][i] == player for j in range(ROWS)]):
            return True
    
    # Check diagonals
    # Main diagonal (top-left to bottom-right)
    if all([board[i][i] == player for i in range(ROWS)]):
        return True
    # Anti-diagonal (top-right to bottom-left)
    if all([board[i][ROWS - 1 - i] == player for i in range(ROWS)]):
        return True
        
    return False

def is_board_full(board):
    return not np.any(board == '.')

def isEnd(current_state):
    board = current_state.board
    
    if check_win(board, 'X') or check_win(board, 'O') or is_board_full(board):
        return True
    
    return False

def immediate_reward(S):
    # good
    if(check_win(S.board, 'X')):
        return 100
    # bad
    if(check_win(S.board, 'O')):
        return -100
    # penalty for extra steps
    return -1


EPISODE_LENGTH = 20

def choose_action(qf, S, epsilon, train = True):
    
    if( train and random.random() <= epsilon ):
        #return random action
        return random.choice([0,1,2,3])
    
    else:
        #return best action
        x,y = S.x-1, S.y-1
        return np.argmax(qf.data[x][y])

def color_move(cur_maze, x, y, current=False):
    """Marks the cell (x, y) as visited or current position."""
    char = CURRENT_POS_CHAR if current else VISITED_PATH_CHAR
    
    # Only mark if it's a regular path cell ('P')
    if cur_maze[x][y] == 'P' or cur_maze[x][y] == VISITED_PATH_CHAR:
        cur_maze[x][y] = char

def episode(qf, epsilon, train = True):
    
    S = state(starting_State.x, starting_State.y)
    
    cur_maze = deepcopy(maze)

    color_move(cur_maze, S.x-1, S.y-1)

    for i in range(EPISODE_LENGTH):

        time.sleep(PRINT_FREQUENCY)

        # print(S.x, S.y)

        action = choose_action(qf, S, epsilon, train)

        if train:
            qf.prop(S, action)
        
        S_prime = state(S.x+dx[action], S.y + dy[action])
        
        if(not inBounds(S_prime) or isWall(S_prime)):
            # print("WALL")
            continue

        S = S_prime
    
        color_move(cur_maze, S.x-1, S.y-1)
        print_board_viz(cur_maze)

        if S == ending_State:
            print("\n" + "*"*50)
            print(f"!!! GOAL REACHED !!!")
            print(f"!!! EPISODE TERMINATED SUCCESSFULLY IN {i + 1} STEPS !!!")
            print("*"*50 + "\n")
            # You can also add a brief pause to ensure the user sees it
            # time.sleep(1.0) 
            return i+1

    return EPISODE_LENGTH

def learn(episodes = 100):

    dqn = DQN(alpha=0.1, gamma=0.9, np.array([9, 10, 9]))

    epsilon = 1

    epsilon_min = 0.01
    epsilon_decay = 0.99

    # qf.load("q_table1.npy")

    for i in range(episodes):
        print(f"Episode #{i+1}: ")
        steps = episode(qf, epsilon, True)
        epsilon = max(epsilon_min, epsilon*epsilon_decay)
    
    # qf.save()

if __name__ == "__main__":
    print("Initial Board Layout:")
    print_board_viz(start_board)
        
    learn(episodes=EPISODE_LENGTH)
