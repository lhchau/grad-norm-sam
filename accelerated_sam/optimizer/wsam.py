import torch
import numpy as np

# Worst Sharpness-Aware Minimization
class WSAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, inner_rho=0.01, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, inner_rho=inner_rho, adaptive=adaptive, **kwargs)
        super(WSAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)
        self.state["step"] = 0
        self.beta = 0.9
        self.exp_avg_old_grad_norm_sq, self.var_old_grad = 0, 0

    @torch.no_grad()
    def first_step(self, zero_grad=False):   
        self.state["step"] += 1
        self.old_grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (self.old_grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None: continue
                param_state = self.state[p]
                
                if 'exp_avg_old_g' not in param_state:
                    param_state['exp_avg_old_g'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                # param_state['exp_avg_old_g'].mul_(self.beta).add_(p.grad)
                param_state['exp_avg_old_g'].lerp_(p.grad, 1 - self.beta)
                # param_state['old_g'] = p.grad.clone().detach()
                
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                
                param_state["e_w"] = e_w.clone().detach()
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        self.exp_avg_old_grad_norm = self._grad_norm(by='exp_avg_old_g')
        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            step_size = group['lr']
            momentum = group['momentum']
            inner_rho = group["inner_rho"]
            for p in group["params"]:
                if p.grad is None: continue
                param_state = self.state[p]
                
                p.sub_(param_state["e_w"])          # get back to "w-worst" from "w-worst + e(w-worst)"
                p.sub_(param_state["step_length"])  # get back to "w" from "w-worst"
                
                d_p = p.grad.data
                
                # param_state['step_length'] = (param_state['old_g'].mul(-inner_rho/self.old_grad_norm)).clone().detach()
                param_state['step_length'] = (param_state['exp_avg_old_g'].mul(-inner_rho/self.exp_avg_old_grad_norm)).clone().detach()
                
                if weight_decay != 0:
                    d_p.add_(p.data, alpha=weight_decay)
                    
                if 'exp_avg' not in param_state:
                    param_state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                param_state['exp_avg'].mul_(momentum).add_(d_p)
                
                p.add_(param_state['exp_avg'], alpha=-step_size)
                
        if zero_grad: self.zero_grad()
    
    @torch.no_grad()
    def step_forward(self):
        for group in self.param_groups:
            for p in group["params"]:
                param_state = self.state[p]
                if 'step_length' not in param_state:
                    param_state['step_length'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                p.add_(param_state['step_length'])
                
    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    @torch.no_grad()
    def _grad_norm(self, by=None):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        if by is None:
            norm = torch.norm(
                        torch.stack([
                            ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                            for group in self.param_groups for p in group["params"]
                            if p.grad is not None
                        ]),
                        p=2
                )
            return norm
        else:
            norm = torch.norm(
                        torch.stack([
                            ((torch.abs(p) if group["adaptive"] else 1.0) * self.state[p][by]).norm(p=2).to(shared_device)
                            for group in self.param_groups for p in group["params"]
                            if p.grad is not None
                        ]),
                        p=2
                )
            return norm
    
    @torch.no_grad()
    def _weight_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        p.data.norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm
    
    @torch.no_grad()
    def cosine_similarity(self, grad1, grad2):
        dot_product = torch.sum(grad1 * grad2)
        norm_grad1 = torch.norm(grad1)
        norm_grad2 = torch.norm(grad2)
        similarity = dot_product / (norm_grad1 * norm_grad2 + 1e-18)
        return similarity.item()
    
    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups