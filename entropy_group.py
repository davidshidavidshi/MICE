import numpy as np
from scipy.stats import entropy

class EntropyGroup(object):
    def __init__(self, args):
        self.args = args
        self.num_agents = args.num_agents

    def reset(self):
        self.total_counts = {idx: 0 for idx in range(self.num_agents)}
        self.count_matrixs = {idx: np.zeros((40,40,1)) for idx in range(self.num_agents)}
        self.search_idx = {0:0}

        self.current_entropys = {idx: 0 for idx in range(self.num_agents)}
        self.current_mutual_informations = {idx: 0 for idx in range(self.num_agents)}
        self.last_entropys = {idx: 0 for idx in range(self.num_agents)}
        self.last_mutual_informations = {idx: 0 for idx in range(self.num_agents)}
    
    def update_count_matrix(self, idx, state):
        grid = np.round(state*255).astype(int)
        rows, cols = grid.shape
        for row in range(rows):
            for col in range(cols):
                try:
                    place_idx = self.search_idx[grid[row,col]]
                except:
                    self.search_idx[grid[row,col]] = len(self.search_idx)
                    self.expand_count_matrixs()
                    place_idx = self.search_idx[grid[row,col]]
                self.count_matrixs[idx][row,col,place_idx] += 1
        self.total_counts[idx] += 1

    def expand_count_matrixs(self):
        for idx in range(self.num_agents):
            self.count_matrixs[idx] = np.concatenate((self.count_matrixs[idx], np.zeros((40,40,1))), axis=2)

    def calculate_step_entropy(self, idx):
        prob_matrix = self.count_matrixs[idx]/self.total_counts[idx]
        entropy_matrix = entropy(prob_matrix,axis=2,base=2)
        self.current_entropys[idx] = np.sum(entropy_matrix)
        step_entropy = self.current_entropys[idx] - self.last_entropys[idx]
        self.last_entropys[idx] = self.current_entropys[idx]
        return step_entropy

    def calculate_step_mutual_information(self, idx):
        mutual_informations = 0
        for agent_idx in range(idx):
            prob_matrix_xy = (self.count_matrixs[agent_idx] + self.count_matrixs[idx])/(self.total_counts[agent_idx]+self.total_counts[idx])
            entropy_matrix_xy = entropy(prob_matrix_xy,axis=2,base=2)
            current_entropy_xy = np.sum(entropy_matrix_xy)
            mutual_information = self.current_entropys[agent_idx] + self.current_entropys[idx] - current_entropy_xy
            mutual_informations += mutual_information
        self.current_mutual_informations[idx] = mutual_informations
        step_mutual_information = self.current_mutual_informations[idx] - self.last_mutual_informations[idx]
        self.last_mutual_informations[idx] = self.current_mutual_informations[idx]
        return step_mutual_information

    def get_trajectory_entropy(self):
        return self.current_entropys
    
    def get_mutual_information(self):
        return self.current_mutual_informations

            


