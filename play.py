import argparse

import copy
import numpy as np
import torch
import torch.nn.functional as F

from env_group import EnvGroup
from model_group import ModelGroup

def play(args):
    env_group = EnvGroup(args)
    local_model_group = ModelGroup(args, env_group.num_actions)
    local_model_group.load_models()

    env_group.reset()
    local_model_group.init_hxcx()

    canvas_paths = {idx: [env_group.canvas[0,idx,:,:]] for idx in range(args.num_agents)}

    while not env_group.is_all_done():
        for idx in range(args.num_agents):
            if env_group.dones[idx] == True:
                continue
            state = copy.deepcopy(env_group.canvas)
            state[0,idx+1:,:,:] = 0
            value, logit = local_model_group.inference(idx, env_group.canvas, eval = True)
            prob = F.softmax(logit, dim=-1)
            action = prob.multinomial(num_samples=1).detach().item()
            _ = env_group.step(idx, action)

            print(f'agent{idx} took action {action}', prob)
            canvas_paths[idx].append(env_group.canvas[0,idx,:,:])
        
        for idx in range(args.num_agents):
            canvas_path = np.asarray(canvas_paths[idx])
            np.save(f'history/path{idx}', canvas_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='A3C')
    parser.add_argument('--num-agents', type=int, default=2)
    parser.add_argument('--env-name', default='gridworldwall')
    parser.add_argument('--traj-coef', type=float, default=1)
    parser.add_argument('--mutual-coef', type=float, default=1)
    args = parser.parse_args()

    play(args)

