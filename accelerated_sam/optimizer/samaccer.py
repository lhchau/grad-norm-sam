import torch
import math
import copy


class SAMACCER(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, betas=(0.9, 0.95), **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAMACCER, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)
        self.state['step'] = torch.tensor(0.)
        self.beta1, self.beta2 = betas
        self.perturb_eps = 1e-8
        self.threshold = 0.001
        self.total_value, self.total_para = 0, 0

    @torch.no_grad()
    def first_step(self, zero_grad=False):        
        self.old_grad_norm = self._grad_norm()
        
        for group in self.param_groups:
            scale = group["rho"] / (self.old_grad_norm + self.perturb_eps)
            
            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                self.state[p]["old_g"] = p.grad.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]["e_w"] = e_w

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        self.state['step'] += 1
        step = self.state['step']
        if step % 100 == 0:
            num_clamped_elements, total_elements = 0, 0
        for group in self.param_groups:
            bias_correction1 = 1 - self.beta1 ** step
            bias_correction2 = 1 - self.beta2 ** step
            weight_decay = group['weight_decay']
            step_size = group['lr'] / bias_correction1
            
            for p in group["params"]:
                if p.grad is None: continue
                d_p = p.grad.data
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"
                param_state = self.state[p]
                
                if weight_decay != 0:
                    d_p.add_(p.data, alpha=weight_decay)
                    
                if (step-1) % 10 == 0:
                    if 'exp_avg_sq' not in self.state[p].keys():
                        param_state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    delta = d_p.sub(param_state["old_g"]).div(group['rho'])
                    param_state['exp_avg_sq'].mul_(self.beta2).addcmul_(delta, delta.conj(), value=1 - self.beta2)
                    self.total_value += torch.sum(param_state['exp_avg_sq'])
                    self.total_para += param_state['exp_avg_sq'].numel()
                
                    
                if 'exp_avg' not in param_state:
                    param_state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                
                param_state['exp_avg'].mul_(self.beta1).add_(d_p)
                
                numer = param_state['exp_avg']
                _denom = self.state[p]['exp_avg_sq'].sqrt()
                denom = _denom.sub(self.total_value/self.total_para).clamp(0,).add(self.total_value/self.total_para)
                denom =  (denom / math.sqrt(bias_correction2)).add_(self.perturb_eps)
                
                ratio = (numer.div(denom)).clamp(-1, 1)
                p.add_(ratio, alpha=-step_size)
                
                if step % 100 == 0:
                    # num_clamped_elements += (ratio != (numer.div(denom))).sum().item()
                    num_clamped_elements += (_denom != denom).sum().item()
                    total_elements += ratio.numel()
        if step % 100 == 0:
            self.clamped_proportion = num_clamped_elements / total_elements
            self.hessian_norm = self.get_hessian_norm()
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
    def get_hessian_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * self.state[p]['exp_avg_sq']).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def set_beta(self, betas):
        self.beta1, self.beta2 = betas
    
    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
