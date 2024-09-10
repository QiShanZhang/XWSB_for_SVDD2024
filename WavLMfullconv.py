import random
import sys
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from WavLM.WavLM import WavLM, WavLMConfig


############################
## FOR fine-tuned SSL MODEL
############################


class SSLModel(nn.Module):
    def __init__(self,device):
        super(SSLModel, self).__init__()

        checkpoint = torch.load('./WavLM/WavLM-Large.pt')
        self.cfg = WavLMConfig(checkpoint['cfg'])
        self.model = WavLM(self.cfg)
        self.model.load_state_dict(checkpoint['model'])
        self.model.to(device)
        self.model.train()
        self.device = device
        return

    def extract_feat(self, input_data):

        input_data = input_data.to(self.device)

        # input should be in shape (batch, length)
        if input_data.ndim == 3:
            input_tmp = input_data[:, :, 0]
        else:
            input_tmp = input_data

        # wav_input_16khz = torch.randn(1, 10000).to(self.device)
        # if self.cfg.normalize:
        #     wav_input_16khz = torch.nn.functional.layer_norm(wav_input_16khz, wav_input_16khz.shape)
        rep, layer_results = self.model.extract_features(input_data, output_layer=self.model.cfg.encoder_layers, ret_layer_results=True)[0]
        layer_reps = [x.transpose(0, 1).unsqueeze(dim=1) for x, _ in layer_results]
        return rep, layer_results, layer_reps
#
# def getAttenF(layerResult):
#     poollayerResult = []
#     fullf = []
#     for layer in layerResult:
#
#         #for caculate attention scores
#         layery = layer[0].transpose(0, 1).transpose(1, 2) #(x,z)  x(201,b,1024) (b,201,1024) (b,1024,201)
#         layery = F.adaptive_avg_pool1d(layery, 1) #(b,1024,1)
#         layery = layery.transpose(1, 2) # (b,1,1024)
#         poollayerResult.append(layery)
#
#         #concat 24 layer feature
#         x = layer[0].transpose(0, 1)
#         x = x.view(x.size(0), -1,x.size(1), x.size(2))
#         fullf.append(x)
#
#     layery = torch.cat(poollayerResult, dim=1)
#     fullfeature = torch.cat(fullf, dim=1)
#     return layery, fullfeature
#

class Model(nn.Module):
    def __init__(self,device):
        super().__init__()
        self.device = device
        self.ssl_model = SSLModel(self.device)
        # self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU(inplace=True)

        # wav2vec2 attention Feature
        # self.fc0 = nn.Linear(1024, 1)
        # self.sig = nn.Sigmoid()

        self.fc1 = nn.Linear(92736, 32)
        self.fc3 = nn.Linear(32,1)
        # self.logsoftmax = nn.LogSoftmax(dim=1)
        self.dropout = nn.AlphaDropout(p=0.3)

        self.conv1 = nn.Conv2d(in_channels=25, out_channels=32, kernel_size=3, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=0)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=0)
        self.pool = nn.MaxPool2d(2, 2)


    def forward(self, x):
        x = x.to(self.device)
        x_ssl_feat, layerResult, layer_reps = self.ssl_model.extract_feat(x.squeeze(-1))  # layerresult = [(x,z),24个] x(201,1,1024) z(1,201,201)
        #print()
        full_feat = torch.cat(layer_reps, dim=1)
        #print(full_feat.shape) # [2,25,249.1024]
        #x = x_ssl_feat.mean(dim=1)
        x = self.conv1(full_feat)
        x = self.pool(x)
        x = self.selu(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.selu(x)
        x = self.conv3(x)
        x = self.pool(x)
        x = self.selu(x)
        #print(x.shape)

        x = torch.flatten(x,1)
        #print(x.shape)
        x = self.fc1(x)
        x = self.selu(x)
        x = self.dropout(x)
        x = self.fc3(x)



        return x
