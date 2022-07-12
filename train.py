import torch
import torch.nn.functional as F
import numpy as np
import copy
import time

from env_group import EnvGroup
from model_group import ModelGroup
from utils import ensure_shared_grads

def train(rank, args, shared_model_group):
    # torch.manual_seed(args.seed + rank)

    env_group = EnvGroup(args)
    local_model_group = ModelGroup(args, env_group.num_actions)
    for idx in range(args.num_agents):
        local_model_group.reload_models(idx, shared_model_group)

    env_group.reset()
    local_model_group.init_hxcx()

    values = {idx: [] for idx in range(args.num_agents)}
    log_probs = {idx: [] for idx in range(args.num_agents)}
    rewards = {idx: [] for idx in range(args.num_agents)}
    entropies = {idx: [] for idx in range(args.num_agents)}

    while True:
        for step in range(args.num_steps):
            for idx in range(args.num_agents):
                if env_group.dones[idx] != True:
                    state = copy.deepcopy(env_group.canvas)
                    state[0,idx+1:,:,:] = 0
                    value, logit = local_model_group.inference(idx, copy.deepcopy(env_group.canvas))
                    prob = F.softmax(logit, dim=-1)
                    log_prob = F.log_softmax(logit, dim=-1)
                    entropy = -(log_prob * prob).sum(1, keepdim=True)
                    action = prob.multinomial(num_samples=1).detach()
                    log_prob = log_prob.gather(1, action)
                    reward = env_group.step(idx, action.item())

                    values[idx].append(value)
                    log_probs[idx].append(log_prob)
                    rewards[idx].append(reward)
                    entropies[idx].append(entropy)

                next_idx = (idx+1)%args.num_agents
                require_train = (next_idx == 0 and step == args.num_steps-2) or (next_idx != 0 and step == args.num_steps-1) or (env_group.dones[next_idx] == True)
                if require_train:
                    if env_group.dones[next_idx] == True:
                        values[next_idx].append(torch.zeros(1,1))
                    else:
                        value, _ = local_model_group.inference(next_idx, copy.deepcopy(env_group.canvas), dummy=True)
                        values[next_idx].append(value)
                    update_model(args, next_idx, values[next_idx], log_probs[next_idx], rewards[next_idx], entropies[next_idx], local_model_group, shared_model_group)
                    
                    local_model_group.detach_hxcx(next_idx)
                    values[next_idx] = []
                    log_probs[next_idx] = []
                    rewards[next_idx] = []
                    entropies[next_idx] = []
                    local_model_group.reload_models(next_idx, shared_model_group)
        
        if env_group.is_all_done():
            env_group.reset()
            local_model_group.init_hxcx()

            values = {idx: [] for idx in range(args.num_agents)}
            log_probs = {idx: [] for idx in range(args.num_agents)}
            rewards = {idx: [] for idx in range(args.num_agents)}
            entropies = {idx: [] for idx in range(args.num_agents)}

def update_model(args, idx, values, log_probs, rewards, entropies, local_model_group, shared_model_group):
    if len(values) <= 1:
        return

    R = values[-1].detach()

    value_loss = 0
    policy_loss = 0
    entropy_loss = 0
    gae = torch.zeros(1, 1)
    for i in reversed(range(len(rewards))):
        R = args.gamma * R + rewards[i]
        advantage = - values[i]
        value_loss = value_loss + 0.5 * advantage.pow(2)

        delta_t = rewards[i] + args.gamma * values[i + 1] - values[i]
        gae = gae * args.gamma * args.gae_lambda + delta_t

        policy_loss = policy_loss - log_probs[i] * gae.detach()
        entropy_loss = entropy_loss - entropies[i]

    shared_model_group.optimizers[idx].zero_grad()
    (policy_loss + args.value_loss_coef * value_loss + args.entropy_coef*entropy_loss).backward()

    ensure_shared_grads(local_model_group.models[idx], shared_model_group.models[idx])
    shared_model_group.optimizers[idx].step()


