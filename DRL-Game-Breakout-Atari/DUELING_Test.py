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


# set device
model_name =True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# hyperparameters
BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.01
EXPLORE_STEP = 10000
EPS_DECAY = (EPS_START - EPS_END) / EXPLORE_STEP
TARGET_UPDATE = 100
RENDER = True
lr = 0.0001
MEMORY_SIZE = 1000000000
INITIAL_MEMORY = 600
NUM_EPISODES = 600
HEIGHT = 84
WIDTH = 84
TEST_EPISODES = 10


MODEL_PATH = 'DuelingNet_model.pt'

      # create environment
      # See wrappers.py
env = create_atari_env("Breakout-v0", episode_life=False, frame_stack=True, scale=True, clip_rewards=False)
epsilon = EPS_START
steps_done = 0
      # initialize replay memory
memory = ReplayMemory(MEMORY_SIZE)

      # create networks
action_num = env.action_space.n
  #policy_net = DQN(HEIGHT, WIDTH, action_num).to(device)
  #target_net = DQN(HEIGHT, WIDTH, action_num).to(device)

policy_net = Dueling_DQN(in_channels=4,num_actions=action_num).to(device)
target_net = Dueling_DQN(in_channels=4,num_actions=action_num).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
print(policy_net)

      # setup optimizer

#optimizer = optim.Adam(policy_net.parameters(), lr=lr)
optimizer = optim.RMSprop(policy_net.parameters(),lr=lr)

      # train model
train(env, NUM_EPISODES,model_name="Dueling_NET")
torch.save(policy_net, MODEL_PATH)


# Testing the Duell Model
policy_net = torch.load(MODEL_PATH)
test(env, TEST_EPISODES, policy_net, render=False, model_name="Duelling_Net")

rewardDuell = policy_net.reward_list

lossDuell = policy_net.loss_list