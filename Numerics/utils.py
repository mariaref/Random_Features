# some useful functions
import os
import shutil
import math
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np


class FastMNIST(datasets.MNIST):
    def __init__(self, *args, **kwargs): 
        super().__init__(*args, **kwargs)

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.data = self.data.unsqueeze(1).float().div(255)
        self.data = self.data.sub_(self.data.mean()).div_(self.data.std())
        self.data, self.targets = self.data.to(self.device), self.targets.to(self.device)
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        return img, target, index

class FastFashionMNIST(datasets.FashionMNIST):
    def __init__(self, *args, **kwargs): 
        super().__init__(*args, **kwargs)

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.data = self.data.unsqueeze(1).float().div(255)
        self.data = self.data.sub_(self.data.mean()).div_(self.data.std())
        self.data, self.targets = self.data.to(self.device), self.targets.to(self.device)
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        return img, target, index


class FastCIFAR10(datasets.CIFAR10):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.data = torch.from_numpy(self.data).float().div(255)
        self.data = self.data.sub_(self.data.mean()).div_(self.data.std())
        self.data, self.targets = self.data.to(self.device), torch.LongTensor(self.targets).to(self.device)
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        return img, target, index

class FastCIFAR100(datasets.CIFAR100):
    def __init__(self, *args, **kwargs): 
        super().__init__(*args, **kwargs)

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.data = torch.from_numpy(self.data).float().div(255)
        self.data = self.data.sub_(self.data.mean()).div_(self.data.std())
        self.data, self.targets = self.data.to(self.device), torch.LongTensor(self.targets).to(self.device)
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        return img, target, index

def get_pca(tr_data, te_data, input_size):
    device = tr_data.device
    
    tr_data.data = tr_data.data.view(tr_data.data.size(0),-1)
    te_data.data = te_data.data.view(te_data.data.size(0),-1)
    x = tr_data.data.cpu()
    # DATA IS ALREADY NORMALIZED
    # m = x.mean(0).expand_as(x)
    # u,s,v = torch.svd(torch.t(x-m))
    u,s,v = torch.svd(torch.t(x))
    tr_data.data = (tr_data.data) @ u[:, :input_size].to(device) / s[:input_size].to(device) ** 0.5
    te_data.data = (te_data.data) @ u[:, :input_size].to(device) / s[:input_size].to(device) ** 0.5
    return tr_data, te_data

def linear_hinge_loss(output, target):
    binary_target = output.new_empty(*output.size()).fill_(-1)
    binary_target.scatter_(1, target.view(-1, 1), 1)
    delta = 1 - binary_target * output # still batch_size * num_classes
    delta[delta <= 0] = 0
    return delta.mean()

def quadratic_hinge_loss(output,target,epsilon=0.5):
    output_size=output.size(1)
    if output_size==1:
        target = 2*target.double()-1
        print(target,output)
        return 0.5*(epsilon-output*target).mean()
    delta = torch.zeros(output.size(0))
    for i,(out,tar) in enumerate(zip(output,target)):
        tar = int(tar)
        delta[i] = epsilon + torch.cat((out[:tar],out[tar+1:])).max() - out[tar]
    loss = 0.5 * torch.nn.functional.relu(delta).pow(2).mean()
    return loss

def swap_loss(output, target):
    binary_target = output.new_empty(*output.size()).fill_(-1)
    binary_target.scatter_(1, target.view(-1, 1), 1)
    delta = 1 - binary_target * output
    delta[delta <= 0] = 0
    return delta.mean()

def copy_py(dst_folder):
    if not os.path.exists(dst_folder):
        print("Folder doesn't exist!")
        return
    for f in os.listdir():
        if f.endswith('.py'):
            shutil.copy2(f, dst_folder)

def get_layerWise_norms(net):
    w = []
    g = []
    for p in net.parameters():    
        if p.requires_grad:
            w.append(p.view(-1).norm())
            g.append(p.grad.view(-1).norm())
    return w, g


def get_grads(model): 
    res = []
    for p in model.parameters():
        if p.requires_grad:
            res.append(p.grad.view(-1))
    grad_flat = torch.cat(res)
    return grad_flat


def who_am_i():
    import subprocess
    whoami = subprocess.run(['whoami'], stdout=subprocess.PIPE)
    whoami = whoami.stdout.decode('utf-8')
    whoami = whoami.strip('\n')
    return whoami

def accuracy(out, y):
    _, pred = out.max(1)
    correct = pred.eq(y)
    return 100 * correct.sum().float().item() / y.size(0)

def corrects(out, y):
    _, pred = out.max(1)
    correct = pred.eq(y)
    return correct.sum().float().item()

def get_data(args):

    if args.preprocess:
        raise NotImplementedError("We don't use that!")
    PATH = '~/data'
    default_input_dims = {'MNIST':28*28, 'FashionMNIST':28*28, 'CIFAR10':32*32*3, 'CIFAR100':32*32*3}
    default_num_classes = {'MNIST':10, 'FashionMNIST':10, 'CIFAR10':10, 'CIFAR100':100}
    tr_data = eval('Fast'+args.dataset)(PATH, train=True, download=True)
    te_data = eval('Fast'+args.dataset)(PATH, train=False, download=True)
    if not args.input_dim:
        input_dim = default_input_dims[args.dataset]
    else:
        input_dim = args.input_dim
        tr_data, te_data = get_pca(tr_data, te_data, input_dim)
    if args.num_classes is None:
        num_classes = default_num_classes(args.dataset)
    else:
        num_classes = args.num_classes
        tr_data.targets = tr_data.targets%num_classes
        te_data.targets = te_data.targets%num_classes
    if args.data_size:
        tr_data.data = tr_data.data[:args.data_size]
        tr_data.targets = tr_data.targets[:args.data_size]
    
    train_loader = DataLoader(
        dataset=tr_data,
        batch_size=args.bs, 
        shuffle=True,
        )

    train_loader_eval = DataLoader(
        dataset=tr_data,
        batch_size=args.bs_eval, 
        shuffle=False,
        )

    # test data is fixed
    test_loader_eval = DataLoader(
        dataset=te_data,
        batch_size=args.bs_eval, 
        shuffle=False,
        )

    return train_loader, test_loader_eval, train_loader_eval, input_dim, num_classes

