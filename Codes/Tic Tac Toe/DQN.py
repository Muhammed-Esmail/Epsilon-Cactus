import numpy as np
import os

PATH = "Models/"

'''

Must implement the (encode) function:
    def encode(board) -> board encoding into numbers

'''

class DQN:
    
    def __init__(self, alpha, gamma, topology):
        
        self.alpha = alpha
        self.gamma = gamma
        self.Net = []
        for i in range(len(topology) - 1):
            # Weights: (input_size x output_size) - random initialization is crucial!
            weights = np.random.randn(topology[i], topology[i+1]) * 0.01 
            # Biases: (1 x output_size)
            biases = np.zeros((1, topology[i+1]))
            self.Net.append({'W': weights, 'B': biases})
    

    def relu(self, x):
        return np.maximum(0, x)
    

    def feedforward(self, state_input):
        # State input should be a flattened 9-element array (0, 1, or 2)
        A = self.encode(state_input)
        for layer in self.Net:
            # Z = A_prev * W + B
            Z = np.dot(A, layer['W']) + layer['B']
            A = self.relu(Z) # Use ReLU activation
        
        # The final output is the Q-values for all 9 actions
        return A.flatten()
    
    def backProp(self, ):
        pass

    def save(self, filename = 'q_net'):

        # ensure directory exists
        if not os.path.exists(PATH):
            os.makedirs(PATH)

        try:
            num = 1
            while(os.path.exists(PATH + filename + str(num))):
                num += 1
            np.save(PATH + filename + str(num), self.Net)
            print("Model Saved Correctly <3")
        except Exception as e:
            print("Error Saving File!!!")
            return

    def load(self, filename : str):
        if(os.path.exists(PATH + filename)):
            try:
                file = PATH + filename
                self.Net = np.load(file)
                print("Model Loaded Correctly <3")
            except Exception as e:
                print("Error loading file!!!")
        else:
            print("File Doesn't Exist!!!")
