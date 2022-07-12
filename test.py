import time
import numpy as np
import copy

import torch
import torch.nn.functional as F

from env_group import EnvGroup
from model_group import ModelGroup
from utils import process_info, process_traj, process_mutual

def test(rank, args, shared_model_group):
    start_time = time.time()
    # torch.manual_seed(args.seed + rank)

    env_group = EnvGroup(args)
    local_model_group = ModelGroup(args, env_group.num_actions)
    for idx in range(args.num_agents):
        local_model_group.reload_models(idx,shared_model_group)

    env_group.reset()
    local_model_group.init_hxcx()

    while True:
        for idx in range(args.num_agents):
            if env_group.dones[idx] == True:
                continue
            state = copy.deepcopy(env_group.canvas)
            state[0,idx+1:,:,:] = 0
            _, logit = local_model_group.inference(idx, env_group.canvas, eval = True)
            prob = F.softmax(logit, dim=-1)
            action = prob.multinomial(num_samples=1).detach()
            _ = env_group.step(idx, action.item())
        
        if env_group.is_all_done():
            trajectory_entropys = env_group.get_trajectory_entropy()
            mutual_entropys = env_group.get_mutual_information()
            infos = env_group.get_info()

            time_past = time.strftime("%Hh %Mm %Ss", time.gmtime(time.time()-start_time))
            info_msg = process_info(infos)
            traj_msg = process_traj(trajectory_entropys)
            mutual_msg = process_mutual(mutual_entropys)
            print(time_past, traj_msg, mutual_msg, info_msg)

            env_group.reset()
            local_model_group.init_hxcx()
            for idx in range(args.num_agents):
                local_model_group.reload_models(idx,shared_model_group)
            local_model_group.save_models()

