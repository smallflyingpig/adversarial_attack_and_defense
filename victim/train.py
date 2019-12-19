import argparse
from torch.utils.data import Dataset, DataLoader
from victim.classifier import ConvNet
from common.trainer import ClassifierTrainer, save_checkpoint, load_checkpoint
from torchvision.datasets import MNIST, CIFAR10
from torchvision import transforms 
from torch import nn, optim
import logging
import os
from tensorboardX import SummaryWriter


def batch_process(model:nn.Module, data, train_mode=True, **kwargs):
    x, y = data
    x, y = x.float().cuda(), y.long().cuda()
    B = x.shape[0]
    if train_mode:
        model.train()
        optimizer, loss_func = kwargs.get("optimizer"), kwargs.get("loss_func")
        pred = model.forward(x)
        loss = loss_func(pred, y)
        model.zero_grad()
        loss.backward()
        optimizer.step()
        pred_idx = pred.data.max(dim=1, keepdim=True)[1]
        accu = pred_idx.eq(y.data.view_as(pred_idx)).long().cpu().sum().float()/float(B)
        
        rtn = {
            "output":"loss:{:6.3f}, accu:{:6.3f}".format(
                 loss.cpu().detach().data, accu),
            "vars":{'loss':loss.cpu().detach().data, 'accu':accu},
            "count":{'loss':B, 'accu':B}
        }
    else: #eval
        model.eval()
        loss_func = kwargs.get("loss_func")
        pred = model.forward(x)
        loss = loss_func(pred, y)
        pred_idx = pred.data.max(dim=1, keepdim=True)[1]
        accu = pred_idx.eq(y.data.view_as(pred_idx)).long().cpu().sum().float()/float(B)
        rtn = {
            "output":"loss:{:6.3f}, accu:{:6.3f}".format(
                 loss.cpu().detach().data, accu),
            "vars":{'loss':loss.cpu().detach().data, 'accu':accu},
            "count":{'loss':B, 'accu':B}
        }

    return rtn



class EvalHook(object):
    def __init__(self):
        self.best_accu = 0
    
    def __call__(self, model:nn.Module, epoch_idx, output_dir, 
        eval_rtn:dict, test_rtn:dict, logger:logging.Logger, writer:SummaryWriter):
        # save model
        acc = eval_rtn.get('accu', 0)
        is_best = acc > self.best_accu
        self.best_accu = acc if is_best else self.best_accu
        model_filename = "epoch_{}.pth".format(epoch_idx)
        save_checkpoint(model, os.path.join(output_dir, model_filename), 
            meta={'epoch':epoch_idx})
        os.system(
            "ln -sf {} {}".format(os.path.abspath(os.path.join(output_dir, model_filename)), 
            os.path.join(output_dir, "latest.pth"))
            )
        if is_best:
            os.system(
            "ln -sf {} {}".format(os.path.abspath(os.path.join(output_dir, model_filename)), 
            os.path.join(output_dir, "best.pth"))
            )

        if logger is not None:
            logger.info("EvalHook: best accu: {:.3f}, is_best: {}".format(self.best_accu, is_best))


def get_parser():
    parser = argparse.ArgumentParser("train the classifier")
    parser.add_argument("--local_rank", type=int, default=0, help="")
    parser.add_argument("--dataset", type=str, choices=['mnist', 'cifar10'], help="")
    parser.add_argument("--model", type=str, choices=['convnet'], help="")
    parser.add_argument("--output_dir", type=str, default="./output/mnist_convnet", help="")
    parser.add_argument("--data_root", type=str, default="./data/mnist", help="")
    parser.add_argument("--epoch", type=int, default=100, help="")
    args = parser.parse_args()
    return args


dataset_all = {'mnist':MNIST, 'cifar10':CIFAR10}
model_all = {'convnet':ConvNet}
def main(args):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])
    dataset_train = MNIST(args.data_root, train=True, transform=transform, download=True)
    dataset_eval = MNIST(args.data_root, train=False, transform=transform)
    dataloader_train = DataLoader(dataset_train, num_workers=8, batch_size=64, pin_memory=True, shuffle=True)
    dataloader_eval = DataLoader(dataset_eval, num_workers=8, batch_size=64, pin_memory=True, shuffle=False)
    model = ConvNet(in_channel=1, n_class=10)
    model = model.cuda()
    loss_func = nn.NLLLoss()
    eval_hook = EvalHook()
    optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-6)
    trainer = ClassifierTrainer(model, dataloader_train, optimizer, loss_func, batch_process, 
        args.output_dir, args.local_rank, dataloader_eval, no_dist=True, print_every=300, eval_hook=eval_hook)
    trainer.logger.info(args)
    trainer.run(args.epoch)

    
if __name__=="__main__":
    args = get_parser()
    main(args)
