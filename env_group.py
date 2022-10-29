import cv2
import gym
import numpy as np
import torch
import copy

from entropy_group import EntropyGroup
from custom_envs import ExplorationGameDual, ExplorationGameNoisy

class EnvGroup(object):
    def __init__(self, args):
        self.traj_coef = args.traj_coef
        self.mutual_coef = args.mutual_coef
        self.num_agents = args.num_agents
        self.env_name = args.env_name
        self.max_episode_length = args.max_episode_length

        self.envs = {idx: self.create_env() for idx in range(self.num_agents)}
        self.entropy_group = EntropyGroup(args)

    def create_env(self):
        if self.env_name == 'gridworlddual':
            env = ExplorationGameDual(40)
            self.num_actions = env.action_space
        elif self.env_name == 'gridworldnoisy':
            env = ExplorationGameNoisy(40)
            self.num_actions = env.action_space
        else:
            env = gym.make(self.env_name)
            self.num_actions = env.action_space.n
        return env

    def reset(self):
        self.dones = {}
        self.infos = {}
        self.game_lens = {idx: 0 for idx in range(self.num_agents)}
        self.entropy_group.reset()
        self.canvas = torch.zeros(1,self.num_agents,40,40)
        for idx in range(self.num_agents):
            state = self.envs[idx].reset()
            state = self._process_frame(state)
            self.entropy_group.update_count_matrix(idx, state)
            self.canvas[0,idx,:,:] = torch.from_numpy(state).unsqueeze(0).unsqueeze(0)
            self.dones[idx] = False
            self.infos[idx] = None
        
        for idx in range(self.num_agents):
            _ = self.get_traj_reward(idx)
            _ = self.get_mutual_reward(idx)

    def step(self, idx, action):
        self.game_lens[idx] += 1
        state, extrinsic_reward, done, self.infos[idx] = self.envs[idx].step(action)
        state = self._process_frame(state)
        done = done or self.game_lens[idx] >= self.max_episode_length
        self.infos[idx] = done

        self.entropy_group.update_count_matrix(idx, state)
        self.canvas[0,idx,:,:] = torch.from_numpy(state).unsqueeze(0).unsqueeze(0)
        traj_reward = self.get_traj_reward(idx)
        mutual_reward = self.get_mutual_reward(idx)
        reward = extrinsic_reward + traj_reward + mutual_reward
        return reward, extrinsic_reward, traj_reward, mutual_reward

    def _process_frame(self, frame):
        if 'gridworld' in self.env_name:
            return frame

        elif 'Pong' in self.env_name:
            frame = frame[35:195]
            frame = frame[::4,::4,0]
            frame[frame == 144] = 0
            frame[frame == 109] = 0
            frame[frame != 0] = 1
            return frame.astype(np.float32)

        elif 'Montezuma' in self.env_name:
            frame = frame[35:195]
            frame = frame[::4,::4,0]
            frame = frame/255
            return frame.astype(np.float32)        

    def get_traj_reward(self, idx):
        step_entropy = self.entropy_group.calculate_step_entropy(idx)
        traj_reward = self.traj_coef*step_entropy
        return traj_reward

    def get_mutual_reward(self, idx):
        step_mutual_information = self.entropy_group.calculate_step_mutual_information(idx)
        mutual_reward = -self.mutual_coef*step_mutual_information
        return mutual_reward

    def get_trajectory_entropy(self):
        return self.entropy_group.get_trajectory_entropy()

    def get_mutual_information(self):
        return self.entropy_group.get_mutual_information()

    def get_info(self):
        return self.infos

    def get_game_lens(self):
        return self.game_lens

    def is_all_done(self):
        num_dones = 0
        for idx in range(self.num_agents):
            num_dones += self.dones[idx]
        if num_dones == self.num_agents:
            return True
        else:
            return False

    

