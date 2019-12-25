import torch
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from attack.attacker import AdditiveAttacker


class AttackerEvaluator(object):
    def __init__(self, dataloader:DataLoader, attacker:[AdditiveAttacker], victim:nn.Module, cuda=True):
        self.dataloader = dataloader
        self.attacker = attacker
        self.victim = victim
        self.device = torch.device('cuda') if cuda else torch.device('cpu')
        self.victim.eval()


    def run(self):
        loader = tqdm(self.dataloader)
        correct_num = 0
        norm_add = 0
        for data in loader:
            x, y = data
            x = x.float().to(self.device)
            y = y.long().to(self.device)
            x_delta = self.attacker.generate(x, y)
            pred = self.victim.forward(x_delta)
            pred_idx = pred.max(1)[1]
            correct = (pred_idx==y).sum()
            correct_num += correct
            delta_norm = abs(x-x_delta).mean()
            norm_add += delta_norm
            loader.set_description("correct:{}, delta norm:{}".format(correct, delta_norm))
        loader.close()
        print("accuracy:{:5.3f}, delta norm:{:5.3f}".format(
            float(correct_num)/len(self.dataloader.dataset), norm_add/len(self.dataloader)
            ))





