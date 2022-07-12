from __future__ import print_function

import argparse
import os
os.environ['OMP_NUM_THREADS'] = '1'
import torch
import torch.multiprocessing as mp

from env_group import EnvGroup
from model_group import ModelGroup
from test import test
from train import train

# Training settings
parser = argparse.ArgumentParser(description='A3C')
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--gae-lambda', type=float, default=1.00)
parser.add_argument('--entropy-coef', type=float, default=0.01)
parser.add_argument('--value-loss-coef', type=float, default=0.5)
parser.add_argument('--max-grad-norm', type=float, default=50)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--num-processes', type=int, default=4)
parser.add_argument('--num-agents', type=int, default=2)
parser.add_argument('--num-steps', type=int, default=20)
parser.add_argument('--max-episode-length', type=int, default=4000)
parser.add_argument('--env-name', default='gridworldwall')
parser.add_argument('--traj-coef', type=float, default=0.05)
parser.add_argument('--mutual-coef', type=float, default=0.02)
parser.add_argument('--load-model', default=False)


if __name__ == '__main__':
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    env_group = EnvGroup(args)
    shared_model_group = ModelGroup(args, env_group.num_actions)
    shared_model_group.share_memory()
    shared_model_group.create_share_optimizer()
    
    processes = []

    p = mp.Process(target=test, args=(args.num_processes, args, shared_model_group))
    p.start()
    processes.append(p)
    
    for rank in range(0, args.num_processes):
        p = mp.Process(target=train, args=(rank, args, shared_model_group))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
