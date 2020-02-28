import torch
import torch.nn as nn
import copy 

class MulConstant(nn.Module):
    def __init__(self, constant=10):
        super(MulConstant, self).__init__()
        self.constant = constant

    def forward(self, x):
        return x * self.constant

    def backward(self, g):
        return g * self.constant, None


class FullyConnected(nn.Module):
    def __init__(self, sigma_0=1, input_dim=28*28, width=512, depth=4, num_classes=10, bias=True):
        super(FullyConnected, self).__init__()
        self.input_dim = input_dim 
        self.width = width
        self.depth = depth
        self.bias = bias 
        self.num_classes = num_classes    
        
        layers = []
        layers.append(nn.Linear(self.input_dim, self.width, bias=self.bias))
        layers.append(MulConstant( 1 / (self.input_dim ** 0.5)))
        layers.append(nn.ReLU(inplace=True))
        for i in range(self.depth - 2):
            layers.append(nn.Linear(self.width, self.width, bias=self.bias))
            layers.append(MulConstant( 1 / (self.width ** 0.5))) 
            layers.append(nn.ReLU())
        layers.append(nn.Linear(self.width, self.num_classes, bias=self.bias),)
        layers.append(MulConstant( 1 / (self.width ** 0.5)))
        self.net = nn.Sequential(*layers)
        
        # NTK initialization
        with torch.no_grad():
            for m in self.net.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0, std=sigma_0)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
 
    def forward(self, x):
        x = x.view(x.size(0), self.input_dim)
        x = self.net(x)
        return x


def fc(**kwargs):
    return FullyConnected(**kwargs)


if __name__ == '__main__':

    # time and memory test 
    import time 
    t = time.time()
    
    x = torch.randn(3, 1, 28, 28)
    net = FullyConnected(width=123, bias=False)
    print(net)

