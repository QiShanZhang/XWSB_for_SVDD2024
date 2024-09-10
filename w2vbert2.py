import random
import sys
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from transformers import (
    Wav2Vec2FeatureExtractor,
    HubertModel,
)
from transformers import AutoProcessor, AutoModelForCTC
from transformers import AutoFeatureExtractor, Wav2Vec2BertForSequenceClassification,Wav2Vec2BertConfig,Wav2Vec2BertForAudioFrameClassification,Wav2Vec2BertModel


class Model(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

        #self.selu = nn.SELU(inplace=True)
        #self.dropout = nn.AlphaDropout(p=0.2)
        #self.fc3 = nn.Linear(32,2)

        #model_path = "huggingface/facebook/w2v-bert-2.0"
        self.config = Wav2Vec2BertConfig.from_pretrained("huggingface/facebook/w2v-bert-2.0")

        # 启用输出隐藏状态和注意力权重
        self.config.output_hidden_states = True
        #self.config.output_attentions = True

        self.processor = AutoFeatureExtractor.from_pretrained("huggingface/facebook/w2v-bert-2.0")
        self.model = Wav2Vec2BertModel.from_pretrained("huggingface/facebook/w2v-bert-2.0", config=self.config)

        # feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
        # model = Wav2Vec2BertForSequenceClassification.from_pretrained("facebook/w2v-bert-2.0")
        # self.feature_extractor = AutoProcessor.from_pretrained(model_path)
        # self.model = AutoModelForCTC.from_pretrained(model_path)

        self.model = self.model.to(device)
        #self.model = self.model.half()
        self.fc1 = nn.Linear(1024, 2)
        # # self.selu = nn.SELU()
        # # self.dropout = nn.AlphaDropout(p=0.2)
        # # self.fc2 = nn.Linear(16, 2)
        self.model.train()


    def forward(self, x):
        print('in model x',x.shape)

        x = x.cpu()

        x  = self.processor(x)#.input_values.to(self.device) , return_tensors='pt'
        print(x)
        x = x.input_features
        print(x)
        print(x.shape)

        #print('x feature extra befor squeeze', x.shape)
        #x = x.squeeze(0)
        x = x.to(self.device)  # or input_tensor.cuda()
        #print('x feature extra',x.shape)
        x = self.model(x).hidden_states
        # print(len(x))
        # print(x[0])
        # print(x[0].shape)
        x = x.mean(dim=1)
        x = self.fc1(x)
        # x = self.selu(x)
        # x = self.dropout(x)
        # x = self.fc2(x)

        return x
