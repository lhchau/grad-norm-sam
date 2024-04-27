import torch
import numpy as np


class VARSAM2(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, alpha1=0.1, alpha2=0.1, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(VARSAM2, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)
        self.state["step"] = 0
        self.beta = 0.9
        self.alpha1, self.alpha2 = alpha1, alpha2
        self.exp_avg_old_grad_norm_sq, self.var_old_grad_norm_sq = 0, 0
        self.beta1, self.beta2, self.beta3 = 0.9, 0.9, 0.9

    @torch.no_grad()
    def first_step(self, zero_grad=False):   
        self.state["step"] += 1
     
        self.weight_norm = self._weight_norm()
        self.old_grad_norm = self._grad_norm()
        self.exp_avg_old_grad_norm_sq = self.exp_avg_old_grad_norm_sq * self.beta + (1 - self.beta) * (self.old_grad_norm ** 2)
        self.var_old_grad_norm_sq = self.var_old_grad_norm_sq * self.beta + (1 - self.beta) * ((self.old_grad_norm ** 2 - self.exp_avg_old_grad_norm_sq) ** 2)
        
        for group in self.param_groups:
            scale = group["rho"] / (self.old_grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None: continue
                param_state = self.state[p]
                
                if 'exp_avg_old_g' not in param_state:
                    param_state['exp_avg_old_g'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                param_state['exp_avg_old_g'].lerp_(p.grad, 1-self.beta1)
                
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                
                self.state[p]["old_g"] = p.grad.clone()
    
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        step = self.state["step"]
        sim1_list = []
        if (step + 1) % 352 == 0:
            self.new_grad_norm = self._grad_norm()
        
        self.third_grad_norm = self._grad_norm(by='exp_avg_old_g')
        self.var_old_grad = self.exp_avg_old_grad_norm_sq - self.third_grad_norm ** 2
        for group in self.param_groups:
            scale = group["rho"] / (self.third_grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None: continue
                param_state = self.state[p]
                
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"
                
                if (step + 1) % 352 == 0:
                    sim1_list.append(self.cosine_similarity(self.state[p]["old_g"], p.grad))
                param_state["new_g"] = p.grad.clone()
                
                param_state['d_norm_d_p'] = (p.grad.sub(param_state['old_g'])).mul(self.old_grad_norm)
                if 'exp_avg_d_norm_d_p' not in param_state:
                    param_state['exp_avg_d_norm_d_p'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                param_state['exp_avg_d_norm_d_p'].lerp_(param_state['d_norm_d_p'], self.beta2)
                
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * param_state['exp_avg_old_g'] * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                    
        if (step + 1) % 352 == 0:
            self.sim1 = np.mean(sim1_list)
            self.norm_d_norm_d_p = self._grad_norm(by='d_norm_d_p')
        
        if zero_grad: self.zero_grad()
        
    @torch.no_grad()
    def third_step(self, zero_grad=False):   
        for group in self.param_groups:
            step_size = group['lr']
            momentum = group['momentum']
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None: continue
                param_state = self.state[p]

                p.data = param_state["old_p"]  # get back to "w" from "w + e(w)"
                
                if 'exp_avg_third_g' not in param_state:
                    param_state['exp_avg_third_g'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                param_state['exp_avg_third_g'].lerp_(p.grad, self.beta3)
                
                param_state['full_d_norm_d_p'] = (param_state['exp_avg_third_g'].sub(param_state['exp_avg_old_g'])).mul(self.third_grad_norm)
                
                d_p = p.grad
                
                if weight_decay != 0:
                    d_p.add_(p.data, alpha=weight_decay)
                
                regularized_term = param_state['exp_avg_d_norm_d_p'].mul(self.alpha1).sub(param_state['full_d_norm_d_p'], alpha=self.alpha2)
                d_p.add_(regularized_term)
                
                if 'exp_avg' not in param_state:
                    param_state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                param_state['exp_avg'].mul_(momentum).add_(d_p)
                
                p.add_(param_state['exp_avg'], alpha=-step_size)
                
        if zero_grad: self.zero_grad()

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
    def _weight_norm(self, by=None):
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
