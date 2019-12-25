import torch
from torch import nn
from abc import ABC, abstractmethod

class AdditiveAttacker(ABC):
    def __init__(self, eps, data_min=-1, data_max=1):
        self.eps = eps
        self.data_min = data_min
        self.data_max = data_max

    @abstractmethod
    def _get_perturbation(self, x, y=None):
        pass

    def generate(self, x, y=None):
        delta = self._get_perturbation(x, y)
        x = x + delta * self.eps
        x[x<self.data_min] = self.data_min
        x[x>self.data_max] = self.data_max
        return x

