import pyvirtualdisplay


_display = pyvirtualdisplay.Display(visible=False,  # use False with Xvfb
                                    size=(1400, 900))
_ = _display.start()
import copy
from collections import namedtuple
from itertools import count
import math
import random
import numpy as np
import time
import gym
import matplotlib.pyplot as plt


import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import gym
from gym import spaces
import cv2
cv2.ocl.setUseOpenCL(False)
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from Enviroment import *
from Train_test import *
from Models import *
from Agent import *
from Memory import *

class DQN(nn.Module):

    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.loss_list = []
        self.reward_list = []

        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        #self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        #self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        #self.bn3 = nn.BatchNorm2d(64)


        self.fc4 = nn.Linear(3136, 512)
        self.head = nn.Linear(512, outputs)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        return self.head(x)


class Dueling_DQN(nn.Module):
    def __init__(self, in_channels, num_actions):
        super(Dueling_DQN, self).__init__()
        self.loss_list = []
        self.reward_list = []
        self.num_actions = num_actions

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        self.fc1_adv = nn.Linear(in_features=7 * 7 * 64, out_features=512)
        self.fc1_val = nn.Linear(in_features=7 * 7 * 64, out_features=512)

        self.fc2_adv = nn.Linear(in_features=512, out_features=num_actions)
        self.fc2_val = nn.Linear(in_features=512, out_features=1)

        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size = x.size(0)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.view(x.size(0), -1)

        adv = self.relu(self.fc1_adv(x))
        val = self.relu(self.fc1_val(x))

        adv = self.fc2_adv(adv)
        val = self.fc2_val(val).expand(x.size(0), self.num_actions)

        x = val + adv - adv.mean(1).unsqueeze(1).expand(x.size(0), self.num_actions)
        return x