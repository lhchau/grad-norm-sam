import torch
import numpy as np


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)
        self.state["step"] = 0
        self.exp_avg_old_grad_norm_sq, self.var_old_grad = 0, 0


    @torch.no_grad()
    def first_step(self, zero_grad=False):   
        self.state["step"] += 1
        step = self.state["step"]
     
        self.weight_norm = self._weight_norm()
        self.old_grad_norm = self._grad_norm()
        self.exp_avg_old_grad_norm_sq = self.exp_avg_old_grad_norm_sq * self.beta + (1 - self.beta) * (self.old_grad_norm ** 2)
        self.var_old_grad = self.var_old_grad * self.beta + (1 - self.beta) * ((self.old_grad_norm ** 2 - self.exp_avg_old_grad_norm_sq) ** 2)
        sim2_list = []
        for group in self.param_groups:
            scale = group["rho"] / (self.old_grad_norm + 1e-12)
            
            for p in group["params"]:
                if p.grad is None: continue
                
                if step % 100 == 0:
                    sim2_list.append(self.cosine_similarity(self.state[p]["old_g"], p.grad))
                
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                
                self.state[p]["old_g"] = p.grad.clone()
        if step % 100 == 0:
            self.sim2 = np.mean(sim2_list)
    
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        step = self.state["step"]
        sim1_list = []
        sim3_list = []
        sim4_list = []
        sim5_list = []
        if step % 352 == 0:
            self.new_grad_norm = self._grad_norm()
            self.cnt_diff_sign = 0
        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            step_size = group['lr']
            momentum = group['momentum']
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"
                d_p = p.grad.data
                
                if step % 352 == 0:
                    sim3_list.append(self.cosine_similarity(self.state[p]["new_g"], d_p))
                    sim1_list.append(self.cosine_similarity(self.state[p]["old_g"], d_p))
                    self.cnt_diff_sign += (self.state[p]["old_g"].mul(d_p) < 0).sum().item()
                self.state[p]["new_g"] = p.grad.clone()
                
                param_state = self.state[p]
                
                if weight_decay != 0:
                    d_p.add_(p.data, alpha=weight_decay)
                if step % 352 == 0:
                    sim4_list.append(self.cosine_similarity(self.state[p]["old_g"], d_p))
                    sim5_list.append(self.cosine_similarity(self.state[p]["new_g"], d_p))
                    
                if 'exp_avg' not in param_state:
                    param_state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                param_state['exp_avg'].mul_(momentum).add_(d_p)
                
                p.add_(param_state['exp_avg'], alpha=-step_size)
                
        if step % 352 == 0:
            self.sim1 = np.mean(sim1_list)
            self.sim3 = np.mean(sim3_list)
            self.sim4 = np.mean(sim4_list)
            self.sim5 = np.mean(sim5_list)
        
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    @torch.no_grad()
    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
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
