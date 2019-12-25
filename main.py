import argparse
import os, logging
from copy import deepcopy
import torch
from torch import nn 
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, CIFAR10
from torchvision import transforms
from attack.fast_gradient_method import FastGradientMethod
from common.evaluate_attacker import AttackerEvaluator
from victim.classifier import ConvNet
from common.trainer import load_checkpoint, save_checkpoint

def get_parser():
    parser = argparse.ArgumentParser("eval the attacker")
    parser.add_argument("--data_root", type=str, default="./data/mnist", help="")
    parser.add_argument("--dataset", choices=['mnist', 'cifar10'], type=str, default='mnist', help="")
    parser.add_argument("--attacker", choices=['FGM'], type=str, default='FGM', help="")
    parser.add_argument("--victim", choices=['convnet'], type=str, default='convnet', help="")
    parser.add_argument("--victim_path", type=str, default="./victim/models/convnet/epoch_99.pth", help="")
    parser.add_argument("--eps", type=float, default=0.01, help="eps for the additive attacker")
    parser.add_argument("--norm_type", choices=['inf', '1', '2'], type=str, help="")
    args = parser.parse_args()
    return args

dataset_all = {'mnist':MNIST, 'cifar10':CIFAR10}
attacker_all = {'FGM':FastGradientMethod}
victim_all = {'convnet':ConvNet}
def main(args):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])
    dataset_train = MNIST(args.data_root, train=True, transform=transform, download=True)
    dataset_eval = MNIST(args.data_root, train=False, transform=transform)

    dataloader_train = DataLoader(dataset_train, num_workers=8, batch_size=64, pin_memory=True, shuffle=True)
    dataloader_eval = DataLoader(dataset_eval, num_workers=8, batch_size=64, pin_memory=True, shuffle=False)
    victim = ConvNet()
    # victim = victim_all[args.victim]()
    load_checkpoint(victim, args.victim_path)
    victim = victim.cuda()
    attacker = FastGradientMethod(eps=args.eps, norm_type=args.norm_type, victim_model=deepcopy(victim), loss_func=torch.nn.CrossEntropyLoss())
    # attacker = attacker_all[args.attacker]
    attacker = attacker.cuda()
    attacker_evaluator = AttackerEvaluator(dataloader_eval, attacker, victim)
    attacker_evaluator.run()


if __name__=="__main__":
    args = get_parser()
    main(args)
