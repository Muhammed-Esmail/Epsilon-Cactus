import os
import numpy as np
import random
import time
from copy import deepcopy

class state:

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

ROWS = 6
COLS = 6
EPISODE_LENGTH = 100
PRINT_FREQUENCY = 0.2
PATH = "Models/"


starting_State = state(2,1)
ending_State = state(5,6)

# Example Maze Data (for demonstration)
# W = Wall, P = Path, S = Start, E = End
# The code below will replace these with Unicode characters.
maze = [
    ['W', 'W', 'W', 'W', 'W', 'W'],
    ['W', 'P', 'P', 'W', 'P', 'W'],
    ['W', 'W', 'P', 'W', 'P', 'W'],
    ['W', 'P', 'P', 'P', 'P', 'W'],
    ['W', 'P', 'W', 'W', 'P', 'W'],
    ['W', 'W', 'W', 'W', 'W', 'W']
]

# Define Unicode Characters
WALL_CHAR = '█'
PATH_CHAR = ' '
START_CHAR = 'S'
END_CHAR = 'E'
VISITED_PATH_CHAR = '.'
CURRENT_POS_CHAR = '@' 
VISITED_VISUAL = '• '

def print_maze_viz(cur_maze):
    """Prints the maze using stylish Unicode characters."""
    for row in range(ROWS):
        for col in range(COLS):
            cell = cur_maze[row][col]
            
            if cell == VISITED_PATH_CHAR:
                print(VISITED_VISUAL, end='')
            elif row == starting_State.x - 1 and col == starting_State.y - 1:
                print(START_CHAR, end=' ')
            elif row == ending_State.x - 1 and col == ending_State.y - 1:
                maze[row][col] = END_CHAR
                print(END_CHAR, end=' ')
            elif cell == 'W':
                print(WALL_CHAR, end=WALL_CHAR)
            elif cell == 'P':
                print(PATH_CHAR, end=PATH_CHAR)
            elif cell == CURRENT_POS_CHAR:
                print('X ', end='') 
            else:
                print(cell, end=' ')
        print()

'''
Q-Learning:

Q(S, A) := Q(S,A) + alpha * [rt + gamma * max_{a}(Q(S, a)) - Q(S,A)]

S = (x,y)
A = 0, 1, 2, 3
    u  d  l  r
r = 0 if not end
    1 if end
'''

dx = [-1, 1, 0, 0]
dy = [0, 0, -1, 1]


def inBounds(state):
    x = state.x
    y = state.y
    return x <= ROWS and y <= COLS and x >= 1 and y >= 1

def isWall(state):
    x = state.x - 1
    y = state.y - 1
    return maze[x][y] == 'W'

def isEnd(state):
    return state == ending_State

def immediate_reward(S):
    # bad
    if(not inBounds(S) or isWall(S)):
        return -100
    # good
    if(isEnd(S)):
        return 100
    # penalty for extra steps
    return -1

class QF:
    
    def __init__(self, alpha, gamma):
        self.lr = alpha
        self.gamma = gamma
        self.data = np.zeros(shape=(ROWS,COLS,4))

    def prop(self, S, A):

        x = S.x - 1
        y = S.y - 1

        # calc next state
        nextX = x + dx[A]
        nextY = y + dy[A]
        nextState = state(nextX + 1,nextY + 1)

        # max Q for next state
        maxQF = 0
        if not inBounds(nextState):
            maxQF = -100
        else:
            maxQF = np.max(self.data[nextX][nextY])

        old_Q_Val = self.data[x][y][A]
        self.data[x][y][A] = old_Q_Val + self.lr * (immediate_reward(nextState) + self.gamma * maxQF - old_Q_Val)

    def save(self, filename = 'q_table'):
        # ensure directory exists
        if not os.path.exists(PATH):
            os.makedirs(PATH)

        try:
            num = 1
            while(os.path.exists(PATH + filename + str(num))):
                num += 1
            np.save(PATH + filename + str(num), self.data)
            print("Model Saved Correctly <3")
        except Exception as e:
            print("Error Saving File!!!")
            return

    def load(self, filename : str):
        if(os.path.exists(PATH + filename)):
            try:
                file = PATH + filename
                self.data = np.load(file)
                print("Model Loaded Correctly <3")
            except Exception as e:
                print("Error loading file!!!")
        else:
            print("File Doesn't Exist!!!")


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
        print_maze_viz(cur_maze)

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
    qf = QF(alpha=0.1, gamma=0.9)

    epsilon = 1

    epsilon_min = 0.01
    epsilon_decay = 0.99

    qf.load("q_table1.npy")

    for i in range(episodes):
        print(f"Episode #{i+1}: ")
        steps = episode(qf, epsilon, False)
        epsilon = max(epsilon_min, epsilon*epsilon_decay)
    
    # qf.save()

if __name__ == "__main__":
    print("Initial Maze Layout:")
    print_maze_viz(maze)
        
    learn(episodes=100)
