import torch
import numpy as np


class ASGD(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, factor=1.5,**kwargs):
        defaults = dict(factor=factor, **kwargs)
        super(ASGD, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            step_size = group['lr']
            momentum = group['momentum']
            for p in group["params"]:
                if p.grad is None: continue
                d_p = p.grad.data
                
                half_threshold = torch.median(d_p)
                d_p = torch.where(d_p >= half_threshold, d_p * group['factor'], d_p / group['factor'])
                
                param_state = self.state[p]
                
                if weight_decay != 0:
                    d_p.add_(p.data, alpha=weight_decay)
                    
                if 'exp_avg' not in param_state:
                    param_state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                param_state['exp_avg'].mul_(momentum).add_(d_p)
                
                p.add_(param_state['exp_avg'], alpha=-step_size)
                
        if zero_grad: self.zero_grad()

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
