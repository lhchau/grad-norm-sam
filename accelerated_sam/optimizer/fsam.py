import torch
import numpy as np
import math


class FSAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(FSAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)
        self.state["step"] = 0
        self.beta1 = 0.9
        self.sigma = 1
        self.exp_avg_noise, self.var_noise = 0, 0
        self.bc = -1
        self.uc = 1
        self.total_step_num = 352 * 200

    def get_ct(self, global_step_cnt):
        return self.bc * global_step_cnt/self.total_step_num + self.uc*(1-global_step_cnt/self.total_step_num)

    @torch.no_grad()
    def first_step(self, zero_grad=False):   
        self.state["step"] += 1
        global_step_cnt = self.state["step"]
                
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                param_state = self.state[p]
                if "old_exp_avg" not in param_state:
                    param_state["old_exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                param_state["old_exp_avg"].lerp_(p.grad, 1 - self.beta1)
                
                param_state["d_t"] = p.grad - self.sigma * param_state["old_exp_avg"]

        self.old_grad_norm = self._grad_norm(by="d_t")
        
        factor = 1
        if self.old_grad_norm >= self.exp_avg_noise + self.get_ct(global_step_cnt)*math.sqrt(self.var_noise): # go SAM
            factor = 2
        for group in self.param_groups:
            scale = group["rho"] * factor / (self.old_grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None: continue
                param_state = self.state[p]
                param_state["old_p"] = p.data.clone()
                param_state["old_g"] = p.grad.clone()

                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * param_state["d_t"] * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                
        self.exp_avg_noise = self.beta1 * self.exp_avg_noise + (1 - self.beta1) * self.old_grad_norm
        self.var_noise = self.beta1 * self.var_noise + (1 - self.beta1) * (self.exp_avg_noise - self.old_grad_norm)**2
        
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        step = self.state["step"]
        sim1_list = []
        if step % 352 == 0:
            self.new_grad_norm = self._grad_norm()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"
                
                if step % 100 == 0:
                    sim1_list.append(self.cosine_similarity(self.state[p]["old_g"], p.grad))
                self.state[p]["new_g"] = p.grad.clone()
        if step % 352 == 0:
            self.sim1 = np.mean(sim1_list)
        
        self.base_optimizer.step()  # do the actual "sharpness-aware" update

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
        if not by:
            norm = torch.norm(
                        torch.stack([
                            ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                            for group in self.param_groups for p in group["params"]
                            if p.grad is not None
                        ]),
                        p=2
                )
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
    def cosine_similarity(self, grad1, grad2):
        dot_product = torch.sum(grad1 * grad2)
        norm_grad1 = torch.norm(grad1)
        norm_grad2 = torch.norm(grad2)
        similarity = dot_product / (norm_grad1 * norm_grad2 + 1e-18)
        return similarity.item()
    
    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
