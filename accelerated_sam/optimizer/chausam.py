import torch
import numpy as np

# CHAU Sharpness-Aware Minimization
class CHAUSAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, inner_rho=0.01, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, inner_rho=inner_rho, adaptive=adaptive, **kwargs)
        super(CHAUSAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)
        self.state["step"] = 0
        self.beta = 0.9

    @torch.no_grad()
    def first_step(self, zero_grad=False):   
        self.state["step"] += 1
        self.first_grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["inner_rho"] / (self.first_grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None: continue
                param_state = self.state[p]
                
                # p.sub_(param_state["e_w"])  # get back to "w" from "w + old_e_w"

                param_state['first_g'] = p.grad.clone().detach()
                
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.sub_(e_w)  # move to the local mimnimum "w - e'(w)"
                
                param_state["e_w_prime"] = e_w.clone().detach()
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):   
        self.second_grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (self.second_grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None: continue
                param_state = self.state[p]
                
                param_state['second_g'] = p.grad.clone().detach()
                                
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                
                param_state["e_w"] = e_w.clone().detach()
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def third_step(self, zero_grad=False):
        step = self.state["step"]
        sim_first_second_list = []
        sim_first_third_list = []
        sim_second_third_list = []
        
        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            step_size = group['lr']
            momentum = group['momentum']
            for p in group["params"]:
                if p.grad is None: continue
                param_state = self.state[p]
                
                param_state['third_g'] = p.grad.clone().detach()
                
                if (step + 1) % 352 == 0:
                    sim_first_second_list.append(self.cosine_similarity(param_state['first_g'], param_state['second_g']))
                    sim_first_third_list.append(self.cosine_similarity(param_state['first_g'], param_state['third_g']))
                    sim_second_third_list.append(self.cosine_similarity(param_state['second_g'], param_state['third_g']))
                
                p.add_(param_state["e_w_prime"])          # get back to "w + e_w" from "w + e_w - e_w'"
                p.sub_(param_state["e_w"])        # get back to "w" from "w + e_w"
                
                d_p = p.grad.data
                
                if weight_decay != 0:
                    d_p.add_(p.data, alpha=weight_decay)
                    
                if 'exp_avg' not in param_state:
                    param_state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                param_state['exp_avg'].mul_(momentum).add_(d_p)
                
                p.add_(param_state['exp_avg'], alpha=-step_size)
                
        if (step + 1) % 352 == 0:
            self.sim_first_second_list = np.mean(sim_first_second_list)
            self.sim_first_third_list = np.mean(sim_first_third_list)
            self.sim_second_third_list = np.mean(sim_second_third_list)
            
        if zero_grad: self.zero_grad()
    
    @torch.no_grad()
    def step_backward(self):
        for group in self.param_groups:
            for p in group["params"]:
                param_state = self.state[p]
                if 'e_w' not in param_state:
                    param_state['e_w'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                p.add_(param_state['e_w'])
                
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