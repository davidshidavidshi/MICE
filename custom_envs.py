import numpy as np
import copy

class ExplorationGameDual(object):
    def __init__(self, size=40):
        self.size = size
        self.action_space = 4

    def reset(self):
        self.state = np.zeros((self.size, self.size))
        self.shape_grid()
        self.steps = 0
        self.position = [0,0]
        self.path = [copy.copy(self.position)]
        return self.state.astype(np.float32)

    def step(self, action):
        self.steps += 1
        current_position = copy.copy(self.position)

        if action == 0: # up
            self.position[0] = max(self.position[0]-1,0)
        elif action == 1: # down
            self.position[0] = min(self.position[0]+1,self.size-1)
        elif action == 2: # left
            self.position[1] = max(self.position[1]-1,0)
        elif action == 3: # right
            self.position[1] = min(self.position[1]+1,self.size-1)

        if self.state[self.position[0],self.position[1]] == 2:  # hit the wall
             self.position = current_position

        self.state[self.position[0],self.position[1]] = 1
        self.path.append(copy.copy(self.position))

        done = False
        reward = 0
        info = self.path
        if self.steps == 40:
            done = True

        return self.state.astype(np.float32), reward, done, info

    def shape_grid(self):
        self.state[1:,1:] = 2
        self.state[0,0] = 1

class ExplorationGameNoisy(object):
    def __init__(self, size=40):
        self.size = size
        self.action_space = 4

    def reset(self):
        self.state = np.zeros((self.size, self.size))
        self.shape_grid()
        self.steps = 0
        self.position = [0,0]
        self.path = [copy.copy(self.position)]
        return self.state.astype(np.float32)

    def step(self, action):
        self.steps += 1
        current_position = copy.copy(self.position)

        if action == 0: # up
            self.position[0] = max(self.position[0]-1,0)
        elif action == 1: # down
            self.position[0] = min(self.position[0]+1,self.size-1)
        elif action == 2: # left
            self.position[1] = max(self.position[1]-1,0)
        elif action == 3: # right
            self.position[1] = min(self.position[1]+1,self.size-1)

        if self.state[self.position[0],self.position[1]] == 2:  # hit the wall
             self.position = current_position

        if self.position[0] > 0:
            self.state[-3:,-3:] = np.random.randint(3,6,size=(3, 3)) # noisy tv
        else:
            self.state[-3:,-3:] = np.random.randint(3,4,size=(3, 3)) # noisy tv
        self.state[self.position[0],self.position[1]] = 1
        self.path.append(copy.copy(self.position))

        done = False
        reward = 0
        info = self.path
        if self.steps == 40:
            done = True

        return self.state.astype(np.float32), reward, done, info

    def shape_grid(self):
        self.state[1:,1:] = 2
        self.state[-3:,-3:] = 3
        self.state[0,0] = 1