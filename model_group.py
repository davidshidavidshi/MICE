import torch
from model import ActorCritic
import my_optim

class ModelGroup(object):
    def __init__(self, args, num_actions):
        self.args = args
        self.num_agents = args.num_agents
        self.models = {idx: ActorCritic(args.num_agents, num_actions) for idx in range(self.num_agents)}

    def share_memory(self):
        for idx in range(self.num_agents):
            self.models[idx].share_memory()
    
    def create_share_optimizer(self):
        self.optimizers = {}
        for idx in range(self.num_agents):
            self.optimizers[idx] = my_optim.SharedAdam(self.models[idx].parameters(), lr=self.args.lr)
            self.optimizers[idx].share_memory()

    def reload_models(self, idx, shared_model_group):
        self.models[idx].load_state_dict(shared_model_group.models[idx].state_dict())

    def inference(self, idx, state, dummy = False, eval = False):
        if eval:
            with torch.no_grad():
                value, logit, (self.hxs[idx], self.cxs[idx]) = self.models[idx](state, self.hxs[idx], self.cxs[idx])
        else:
            if dummy:
                value, logit, (_, _) = self.models[idx](state, self.hxs[idx], self.cxs[idx])
            else:
                value, logit, (self.hxs[idx], self.cxs[idx]) = self.models[idx](state, self.hxs[idx], self.cxs[idx])
        return value, logit

    def init_hxcx(self):
        self.hxs = {idx: torch.zeros(1, 256) for idx in range(self.num_agents)}
        self.cxs = {idx: torch.zeros(1, 256) for idx in range(self.num_agents)}

    def detach_hxcx(self, idx):
        self.hxs[idx] = self.hxs[idx].detach()
        self.cxs[idx] = self.cxs[idx].detach()

    def save_models(self):
        for idx in range(self.num_agents):
            torch.save(self.models[idx].state_dict(), f'weights/model{idx}.pth')

    def load_models(self):
        for idx in range(self.num_agents):
            self.models[idx].load_state_dict(torch.load(f'weights/model{idx}.pth'))
            
            

        

