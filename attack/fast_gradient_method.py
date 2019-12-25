import torch
from torch import nn
from attack.attacker import AdditiveAttacker
import numpy as np


class FastGradientMethod(nn.Module, AdditiveAttacker):
    def __init__(self, eps, norm_type:str, victim_model:nn.Module, loss_func, targeted=False):
        nn.Module.__init__(self)
        AdditiveAttacker.__init__(self, eps=eps)
        self.norm_type = norm_type
        self.victim_model = victim_model
        self.targeted = targeted
        self.loss_func = loss_func
        self.victim_model.train()

    def _get_perturbation(self, x, y=None):
        tol = 1e-8
        assert y is not None
        x = x.requires_grad_(True)
        pred = self.victim_model.forward(x)
        loss = self.loss_func(pred, y)
        self.victim_model.zero_grad()
        loss.backward()
        x_grad = x.grad.data.detach()

        # Apply norm bound
        if self.norm_type == 'inf':
            x_grad = torch.sign(x_grad)
        elif self.norm_type == '1':
            ind = tuple(range(1, len(x.shape)))
            x_grad = x_grad / (torch.sum(torch.abs(x_grad), axis=ind, keepdims=True) + tol)
        elif self.norm_type == '2':
            ind = tuple(range(1, len(x.shape)))
            x_grad = x_grad / (torch.sqrt(torch.sum(x_grad*x_grad, axis=ind, keepdims=True)) + tol)

        return x_grad
        


        