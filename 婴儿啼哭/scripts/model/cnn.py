# coding=utf-8


import torch.nn as nn
from conf import config
import torchvision.models as models


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = models.resnet18(pretrained=True)
        self.conv.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.conv.avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv.fc = nn.Linear(self.conv.fc.in_features, config.num_classes, bias=True)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, input):
        x = self.conv(input)
        return self.softmax(x)

# class Model(nn.Module):
#     def __init__(self):
#         super(Model, self).__init__()
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(in_channels=1, out_channels=24, kernel_size=2, stride=(1, 1)),
#             # nn.MaxPool2d(kernel_size=(4, 2), stride=(4, 2)),
#             nn.ReLU())
#         self.conv2 = nn.Sequential(nn.Conv2d(in_channels=24, out_channels=48, kernel_size=2),
#                                    # nn.MaxPool2d(kernel_size=(4, 2), stride=(4, 2)),
#                                    nn.ReLU())
#         # self.conv3 = nn.Sequential(nn.Conv2d(in_channels=48, out_channels=48, kernel_size=5),
#         #                            nn.ReLU())
#         # self.linear1 = nn.Linear(2400, 64)
#         self.linear1 = nn.Linear(48, 64)
#         self.linear2 = nn.Linear(64, config.num_classes)

#         self.dropout = nn.Dropout(p=0.5)
#         self.relu = nn.ReLU()
#         self.softmax = nn.Softmax(dim=1)

#     def forward(self, input):
#         x = self.conv1(input)
#         x = self.conv2(x)
#         # x = self.conv3(x)
#         # print(x.shape)
#         x = nn.functional.adaptive_avg_pool2d(x, (1,1))
#         # print(x.shape)
#         x = x.view(x.size(0), -1)
#         # x = self.dropout(x)
#         # print('x', x.shape)
#         x = self.relu(self.linear1(x))
#         x = self.dropout(x)
#         x = self.softmax(self.linear2(x))
#         return x

# print(Model())
