import time
import numpy as np
import copy

import torch
import torch.nn.functional as F
import logging

from env_group import EnvGroup
from model_group import ModelGroup
from utils import process_info, process_traj, process_mutual, process_reward

def test(rank, args, shared_model_group):
    log_file_name = f'{args.env_name.split("/")[-1]}-{args.traj_coef}-{args.mutual_coef}.log'
    open(f'logs/{log_file_name}', 'w').close()
    logging.basicConfig(filename=f'logs/{log_file_name}', level=logging.DEBUG)

    start_time = time.time()
    # torch.manual_seed(args.seed + rank)

    env_group = EnvGroup(args)
    local_model_group = ModelGroup(args, env_group.num_actions)
    
    all_rewards = {idx:[] for idx in range(args.num_agents)}
    env_group.reset()
    local_model_group.init_hxcx()
    for idx in range(args.num_agents):
        local_model_group.reload_models(idx,shared_model_group)

    while True:
        for idx in range(args.num_agents):
            if env_group.dones[idx] == True:
                continue
            state = copy.deepcopy(env_group.canvas)
            state[0,idx+1:,:,:] = 0
            _, logit = local_model_group.inference(idx, env_group.canvas, eval = True) # can replace env_group.canvas with state
            prob = F.softmax(logit, dim=-1)
            if 'gridworld' in args.env_name:
                action = prob.multinomial(num_samples=1).detach()
            else:
                action = prob.max(1, keepdim=True)[1].numpy()
            _, extrinsic_reward, _, _ = env_group.step(idx, action.item())
            all_rewards[idx].append(extrinsic_reward)
        
        if env_group.is_all_done():
            trajectory_entropys = env_group.get_trajectory_entropy()
            mutual_entropys = env_group.get_mutual_information()
            infos = env_group.get_info()
            game_lens = env_group.get_game_lens()

            time_past = time.strftime("%Hh %Mm %Ss", time.gmtime(time.time()-start_time))
            if 'grid' in args.env_name:
                info_msg = process_info(infos)
            traj_msg = process_traj(trajectory_entropys)
            mutual_msg = process_mutual(mutual_entropys)
            reward_msg = process_reward(all_rewards)
            if 'grid' in args.env_name:
                print(time_past, traj_msg, mutual_msg, info_msg, reward_msg)
            else:
                print(time_past, reward_msg, game_lens)

            if args.env_name == 'gridworlddual':
                logging.info(f'{time_past}|{traj_msg[0]}|{traj_msg[1]}|{mutual_msg[1]}|{info_msg["all1"]}')
            elif args.env_name == 'gridworldnoisy':
                logging.info(f'{time_past}|{info_msg["all0"]}|{info_msg["all1"]}|{info_msg["all2"]}|{info_msg["all3"]}')
            elif args.env_name == 'PongDeterministic-v4':
                logging.info(f'{time_past}|{reward_msg[0]}|{reward_msg[1]}|{reward_msg[2]}')

            all_rewards = {idx:[] for idx in range(args.num_agents)}
            env_group.reset()
            local_model_group.init_hxcx()
            for idx in range(args.num_agents):
                local_model_group.reload_models(idx,shared_model_group)
            local_model_group.save_models()


