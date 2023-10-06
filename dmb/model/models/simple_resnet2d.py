import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops.misc import SqueezeExcitation

def conv2d(in_channels, out_channels, kernel_size, bias=False):

    return nn.Conv2d(in_channels=in_channels,
                     out_channels=out_channels,
                     kernel_size=kernel_size, padding=kernel_size//2, padding_mode="circular", bias=bias)


class BasicBlock(nn.Module):

    """
    ResNet Basic Block
    """

    def __init__(self, in_channels, out_channels, kernel_size, is_first_block=False,dropout=0):
        super(BasicBlock, self).__init__()

        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.out_channels = out_channels

        self.is_first_block = is_first_block

        # the first conv
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU()
        self.dropout_1 = nn.Dropout(dropout)
        self.conv1 = conv2d(in_channels=in_channels,
                            out_channels=out_channels, kernel_size=kernel_size,bias=False)
        # the second conv
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()
        self.dropout_2 = nn.Dropout(dropout)
        self.conv2 = conv2d(in_channels=out_channels,
                            out_channels=out_channels, kernel_size=kernel_size,bias=False)

    def forward(self, x):

        identity = x

        # the first conv
        out = x
        if not self.is_first_block:
            out = self.bn1(out)
            out = self.relu1(out)
            out = self.dropout_1(out)

        out = self.conv1(out)

        # the second conv
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout_2(out)
        out = self.conv2(out)

        # if expand channel, also pad zeros to identity
        if self.out_channels != self.in_channels:
            identity = identity.permute(0,3,2,1)
            ch1 = (self.out_channels-self.in_channels)//2
            ch2 = self.out_channels-self.in_channels-ch1
            identity = F.pad(identity, (ch1, ch2), "constant", 0)
            identity = identity.permute(0,3,2,1)

        # shortcut
        out += identity

        return out

from typing import List

class ResNet2d(nn.Module):

    def __init__(self,in_channels:int,out_channels:int,kernel_sizes:List[int],n_channels:List[int], dropout:float=0.0):

        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_sizes = kernel_sizes
        self.n_channels = n_channels
        self.dropout = dropout

        self.first_block = nn.Sequential(*[conv2d(
            in_channels=in_channels, out_channels=n_channels[0], kernel_size=kernel_sizes[0],bias=False),
            nn.BatchNorm2d(n_channels[0]), nn.ReLU(),nn.Dropout(dropout)])
        
        self.basicblock_list = []
        for i_block in range(len(n_channels)-1):
            # is_first_block
            if i_block == 0:
                is_first_block = True
            else:
                is_first_block = False

            in_channels = n_channels[i_block]
            out_channels = n_channels[i_block+1]

            tmp_block = BasicBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_sizes[i_block],
                is_first_block=is_first_block,dropout=dropout)

            self.basicblock_list.append(tmp_block)

        self.last_block = nn.Sequential(*[nn.BatchNorm2d(self.n_channels[-1]), nn.ReLU(),nn.Dropout(dropout),conv2d(
            in_channels=self.n_channels[-1], out_channels=self.out_channels, kernel_size=3,bias=False)])
        
        self.resnet = nn.Sequential(
            self.first_block, *self.basicblock_list, self.last_block)

    def forward(self,x):

        if isinstance(x,(tuple,list)):
            out = tuple(self.resnet(_x) for _x in x)
        else:
            out = self.resnet(x)

        return out


class SeBasicBlock(nn.Module):

    """
    ResNet Basic Block
    """

    def __init__(self, in_channels, out_channels, kernel_size, is_first_block=False,dropout=0,squeeze_factor:int=4):
        super(SeBasicBlock, self).__init__()

        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.out_channels = out_channels

        self.is_first_block = is_first_block

        # the first conv
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU()
        self.dropout_1 = nn.Dropout(dropout)
        self.conv1 = conv2d(in_channels=in_channels,
                            out_channels=out_channels, kernel_size=kernel_size,bias=False)
        # the second conv
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()
        self.dropout_2 = nn.Dropout(dropout)
        self.conv2 = conv2d(in_channels=out_channels,
                            out_channels=out_channels, kernel_size=kernel_size,bias=False)

        self.se = SqueezeExcitation(in_channels, in_channels//squeeze_factor)

    def forward(self, x):

        identity = x

        # the first conv
        out = x
        if not self.is_first_block:
            out = self.bn1(out)
            out = self.se(out)
            out = self.relu1(out)
            out = self.dropout_1(out)

        out = self.conv1(out)

        # the second conv
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout_2(out)
        out = self.conv2(out)

        # if expand channel, also pad zeros to identity
        if self.out_channels != self.in_channels:
            identity = identity.permute(0,3,2,1)
            ch1 = (self.out_channels-self.in_channels)//2
            ch2 = self.out_channels-self.in_channels-ch1
            identity = F.pad(identity, (ch1, ch2), "constant", 0)
            identity = identity.permute(0,3,2,1)

        # shortcut
        out += identity

        return out


class SeResNet2d(nn.Module):

    def __init__(self,in_channels:int,out_channels:int,kernel_sizes:List[int],n_channels:List[int], dropout:float=0.0, squeeze_factor:int=4):

        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_sizes = kernel_sizes
        self.n_channels = n_channels
        self.dropout = dropout

        self.first_block = nn.Sequential(*[conv2d(
            in_channels=in_channels, out_channels=n_channels[0], kernel_size=kernel_sizes[0],bias=False),
            nn.BatchNorm2d(n_channels[0]), SqueezeExcitation(n_channels[0],n_channels[0]//squeeze_factor) , nn.ReLU(),nn.Dropout(dropout)])
        
        self.basicblock_list = []
        for i_block in range(len(n_channels)-1):
            # is_first_block
            if i_block == 0:
                is_first_block = True
            else:
                is_first_block = False

            in_channels = n_channels[i_block]
            out_channels = n_channels[i_block+1]

            tmp_block = SeBasicBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_sizes[i_block],
                is_first_block=is_first_block,dropout=dropout)

            self.basicblock_list.append(tmp_block)

        self.last_block = nn.Sequential(*[nn.BatchNorm2d(self.n_channels[-1]), SqueezeExcitation(self.n_channels[-1],self.n_channels[-1]//4), nn.ReLU(),nn.Dropout(dropout),conv2d(
            in_channels=self.n_channels[-1], out_channels=self.out_channels, kernel_size=3,bias=False)])
        
        self.resnet = nn.Sequential(
            self.first_block, *self.basicblock_list, self.last_block)

    def forward(self,x):

        if isinstance(x,(tuple,list)):
            out = tuple(self.resnet(_x) for _x in x)
        else:
            out = self.resnet(x)

        return out

