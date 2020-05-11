# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import math

import torch
from resnest.torch import *
from torch import nn

_MODEL_NAMES_2_MODEL = {
    'resnest50': resnest50,
    'resnest101': resnest101,
    'resnest200': resnest200,
    'resnest269': resnest269,
    'resnest50_fast_1s1x64d': resnest50_fast_1s1x64d,
    'resnest50_fast_1s2x40d': resnest50_fast_1s2x40d,
    'resnest50_fast_1s4x24d': resnest50_fast_1s4x24d,
    'resnest50_fast_2s1x64d': resnest50_fast_2s1x64d,
    'resnest50_fast_2s2x40d': resnest50_fast_2s2x40d,
    'resnest50_fast_4s1x64d': resnest50_fast_4s1x64d,
    'resnest50_fast_4s2x40d': resnest50_fast_4s2x40d,
}


class ResNeSt(nn.Module):
    def __init__(self, model_name='resnest50', **kwargs):
        super().__init__()

        assert model_name in _MODEL_NAMES_2_MODEL
        self.model = _MODEL_NAMES_2_MODEL[model_name](**kwargs)
        self.model.avgpool = nn.Identity()
        self.model.fc = nn.Identity()

    def forward(self, x):
        x = self.model(x)
        return x

    def load_param(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            if 'fc' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])

    def random_init(self):
        pass
