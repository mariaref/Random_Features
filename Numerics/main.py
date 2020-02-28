import os
import time
import math
import argparse
import copy

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from models import fc
from utils import get_id, get_data, get_grads
from utils import alpha_estimator
from utils import accuracy, corrects, linear_hinge_loss, quadratic_hinge_loss 

def evaluation(eval_loader, net, crit, opt, args, test=True):

    net.eval()

    # run over both test and train set
    total_size = 0
    total_loss = 0
    total_corr = 0

    outputs = []
    # grads = []
    
    if args.zero_init_method == 'centered':
        if test == True:
            def closure(x, y, idx):
                out = net(x).sub(out0_te[idx]).mul(args.alpha)
                loss = crit(out, y).div(args.alpha ** 2)
                return loss, out
        else:
            def closure(x, y, idx):
                out = net(x).sub(out0[idx]).mul(args.alpha)
                loss = crit(out, y).div(args.alpha ** 2)
                return loss, out
    else: 
        def closure(x, y, idx):
            out = net(x)
            loss = crit(out, y)
            return loss, out 
    
    with torch.no_grad(): 
        for x, y, idx in eval_loader:
            # loop over dataset
        
            loss, out = closure(x, y, idx)
            corr = corrects(out, y)
            bs = int(x.size(0))
        
            # record data
            outputs.append(out)
            total_size += bs
            total_loss += loss.item() * bs
            total_corr += corr
    
    hist = [
        total_loss / total_size, 
        total_corr / total_size
    ]

    net.train()
    
    return hist, outputs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', default=25, type=int,
        help='number of samples for train/test eval to be taken on the way')
    parser.add_argument('--bs', default=16, type=int)
    parser.add_argument('--bs_eval', default=2048, type=int,
        help='must be equal to tr bs for alpha-index!!')

    parser.add_argument('--optimizer', default='sgd', type=str)
    parser.add_argument('--lr', default=1, type=float)
    parser.add_argument('--mom', default=0, type=float)
    parser.add_argument('--wd', default=0, type=float)

    parser.add_argument('--alpha', default=100, type=float)  
    parser.add_argument('--zero_init_method', default='none', type=str, choices=['symmetrized', 'centered', 'none'])  
    
    parser.add_argument('--path', default='./data', type=str)
    parser.add_argument('--seed', default=0, type=int)
    
    parser.add_argument('--depth', default=4, type=int)
    parser.add_argument('--model', default='fc', type=str)
    parser.add_argument('--crit', default='NLL', type=str,
        help='NLL | linear_hinge')
    parser.add_argument('--dataset', default='MNIST', type=str,
        help='MNIST |  CIFAR10 | CIFAR100')
    
    parser.add_argument('--input_dim', default=None, type=int,
        help='if None no reduction')
    parser.add_argument('--num_classes', default=None, type=int,
        help='if None no reduction')
    parser.add_argument('--data_size', default=None, type=int,
        help='if 0 the whole dataset')
    parser.add_argument('--scale', default=64, type=int,
        help='scale of the number of convolutional filters')
    parser.add_argument('--width', default=512, type=int, 
        help='width of fully connected layers')

    parser.add_argument('--save_at', default='test', type=str)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--double', action='store_true', default=False)
    parser.add_argument('--no_bias', action='store_true', default=False)
    parser.add_argument('--dropout', action='store_true', default=False)
    parser.add_argument('--schedule', action='store_true', default=False)
    parser.add_argument('--preprocess', action='store_true', default=False)
    parser.add_argument('--lr_schedule', action='store_true', default=False)
    args = parser.parse_args()

    ####
    args.lr = min(1, args.alpha) * args.lr
    ####

    if args.double: 
        torch.set_default_tensor_type('torch.DoubleTensor')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    print(args)

    # training setup
    train_loader, test_loader_eval, train_loader_eval, input_dim, num_classes = get_data(args)

    if args.model == 'fc':
        bias = not args.no_bias # if no_bias is True then bias must be False
        net = fc(bias=bias, width=args.width, depth=args.depth, input_dim=input_dim, num_classes=num_classes).to(device)
    else:
        raise NotImplementedError

    print(net)

    torch.manual_seed(args.seed)

    if args.zero_init_method == 'centered':
        with torch.no_grad():
            out0_list = []
            for x, y, idx in train_loader_eval: 
                out0_list.append(net(x))
            out0 = torch.cat(out0_list)
            out0_list = []
            for x, y, idx in test_loader_eval: 
                out0_list.append(net(x))
            out0_te = torch.cat(out0_list)

    if args.optimizer=='sgd':
        opt = optim.SGD(
            net.parameters(), 
            lr=args.lr, 
            momentum=args.mom,
            weight_decay=args.wd
        )
    elif args.optimizer=='adam':
        opt = optim.Adam(
            net.parameters(), 
            lr=args.lr, 
        )
    else:
        print('optimizer not implemented')
        exit(1)

    if args.lr_schedule:
        milestone = int(args.max_iter / 3)
        scheduler = optim.lr_scheduler.MultiStepLR(opt,
            milestones=[milestone, 2*milestone],
            gamma=0.5)
    
    if args.crit == 'NLL':
        crit = nn.CrossEntropyLoss().to(device)
    elif args.crit == 'linear_hinge':
        crit = linear_hinge_loss
    elif args.crit == 'quadratic_hinge':
        crit = quadratic_hinge_loss
    
    def cycle_loader(dataloader):
        while 1:
            for data in dataloader:
                yield data

    circ_train_loader = cycle_loader(train_loader)

    res = {
        'args': args,
        'test': [],
        'train': [],
        'te_outs': [],
        'tr_outs': [],
        'time': [time.time()], # start by logging the initial time
    }

    # avoid conditional statements in the main training loop
    if args.zero_init_method == 'centered':
        def closure(x, y, idx):
            opt.zero_grad()
            out = net(x).sub(out0[idx]).mul(args.alpha)
            loss = crit(out, y).div(args.alpha ** 2)
            loss.backward()
            opt.step()
    else:
        def closure(x, y, idx):
            opt.zero_grad()
            out = net(x)    
            loss = crit(out, y)
            loss.backward()
            opt.step()
            
    iter_bound = 1000000

    args.checkpts = [0] + [int(x) for x in torch.logspace(1, math.log(iter_bound, 2) , args.num_samples, 2)]

    iter_bound = args.checkpts[-1]    

    best_test = 0
    for i, (x, y, idx) in enumerate(circ_train_loader):
        
        if i in args.checkpts:

            te_hist, te_outs = evaluation(test_loader_eval, net, crit, opt, args)
            tr_hist, tr_outs = evaluation(train_loader_eval, net, crit, opt, args, test=False)
            res['test'].append([i, *te_hist])
            res['train'].append([i, *tr_hist])
            res['time'].append(time.time()) # includes initial point as well
            
            torch.save(res, '{}.res'.format(args.save_at)) # save (override the cumulative results) results and outputs
            
            # save init and final weights and outs // override for last timestep
            if i == 0:
                torch.save(net.state_dict(), '{}.init.weights.pyT'.format(args.save_at))
                torch.save(te_outs, '{}.init.te_outs.pyT'.format(args.save_at)) 
                # torch.save(tr_outs, '{}.init.tr_outs.pyT'.format(args.save_at))
                
            if te_hist[1] > best_test:
                best_test = te_hist[1]
                # torch.save(net.state_dict(), '{}.best.weights.pyT'.format(args.save_at))
                torch.save(te_outs, '{}.best.te_outs.pyT'.format(args.save_at)) 
                # torch.save(tr_outs, '{}.best.tr_outs.pyT'.format(args.save_at))
                
            torch.save(net.state_dict(), '{}.final.weights.pyT'.format(args.save_at))
            torch.save(te_outs, '{}.final.te_outs.pyT'.format(args.save_at))
            # torch.save(tr_outs, '{}.final.tr_outs.pyT'.format(args.save_at))
    
            print('at iter {}, {:2f}, {:1f}, {:2f}, {:1f}, {}'.format(i, *te_hist[:2], *tr_hist[:2], int(time.time())), flush='True')

            # check stopping conditions
            if i == iter_bound: break
            if tr_hist[0] == 0: break
            if np.isnan(tr_hist[0]): break 

        closure(x, y, idx)

