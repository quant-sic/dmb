from torch import nn
from typing import List, Dict, Any
import hydra
import torch


class Exponential(torch.nn.Module):
    def __init__(self, eps=1e-10):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return torch.exp(x) + self.eps
    

class DMBModel(nn.Module):

    def __init__(self, in_channels:int, out_channels:int,observables: List[str],module_list: List[Dict[str, Any]], output_modification: List[Dict[str, Any]] = []):
        super().__init__()

        self.modules_list = torch.nn.ModuleList()
        self.output_modification = torch.nn.ModuleList()

        for module in module_list:
            self.modules_list.append(
                hydra.utils.instantiate(module, _recursive_=False, _convert_="all")
            )

        # output modifications do not change the shape of the output
        for module in output_modification:
            self.output_modification.append(
                hydra.utils.instantiate(module, _recursive_=False, _convert_="all")
            )

        self._observables = observables
        self._in_channels = in_channels
        self._out_channels = out_channels

        self.sanity_check()


    def sanity_check(self):

        if not self._in_channels == self.modules_list[0].in_channels:
            raise ValueError("in_channels of first module must match in_channels of model")
        
        if not self._out_channels == self.modules_list[-1].out_channels:
            raise ValueError("out_channels of last module must match out_channels of model")

    @property
    def observables(self):
        return self._observables
    
    @property
    def in_channels(self):
        return self._in_channels
    
    @property
    def out_channels(self):
        return self._out_channels

    def forward_single_size(self, x):

        for module in self.modules_list:
            x = module(x)

        for module in self.output_modification:
            x = module(x)

        return x
    
    def forward(self, x):

        if isinstance(x,(tuple,list)):
            out = tuple(self.forward_single_size(_x) for _x in x)
        else:
            out = self.forward_single_size(x)
        
        return out