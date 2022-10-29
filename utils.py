import numpy as np

def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad

def process_info(infos):
    results = {}
    paths = []
    for idx in infos:
        path = infos[idx]
        paths.append(path)
        results[idx] = len(np.unique(path,axis=0))
        paths_so_far = np.concatenate(paths,axis=0)
        results[f'all{idx}'] = len(np.unique(paths_so_far,axis=0))
    return results

def process_traj(traj_entropys):
    results = {}
    for idx in traj_entropys:
        results[idx] = round(traj_entropys[idx],2)
    return results

def process_mutual(mutual_entropys):
    results = {}
    for idx in mutual_entropys:
        results[idx] = round(mutual_entropys[idx],2)
    return results

def process_reward(all_rewards):
    results = {idx: round(sum(all_rewards[idx]),2) for idx in all_rewards}
    return results
