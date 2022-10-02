
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


Transition = namedtuple('Transion',
                        ('state', 'action', 'next_state', 'reward'))

def select_action(state):
    global epsilon
    global steps_done
    sample = random.random()
    epsilon -= EPS_DECAY
    if epsilon <= EPS_END:
        epsilon = EPS_END

    steps_done += 1
    if sample > epsilon:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state.to(device)).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(action_num)]], device=device, dtype=torch.long)


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    """
    zip(*transitions) unzips the transitions into
    Transition(*) creates new named tuple
    batch.state - tuple of all the states (each state is a tensor)
    batch.next_state - tuple of all the next states (each state is a tensor)
    batch.reward - tuple of all the rewards (each reward is a float)
    batch.action - tuple of all the actions (each action is an int)    
    """
    batch = Transition(*zip(*transitions))
    actions = tuple((map(lambda a: torch.tensor([[a]], device=device), batch.action)))
    rewards = tuple((map(lambda r: torch.tensor([r], device=device), batch.reward)))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device,
                                  dtype=torch.bool)
    # print(non_final_mask)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]).to(device)

    state_batch = torch.cat(batch.state).to(device)
    action_batch = torch.cat(actions)
    reward_batch = torch.cat(rewards)

    state_action_values = policy_net(state_batch.to(device)).gather(1, action_batch)  # [batch_size, 1]
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    # DQN
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    # loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    # for para in policy_net.parameters():
    #     if para.grad is not None:
    #         print("para.grad not None1")
    optimizer.step()
    return loss.item()

def get_state(obs):
    state = np.array(obs)
    # print(state.shape)  # 84*84*4
    state = state.transpose((2, 0, 1))
    state = state.astype(np.float32)
    state = torch.from_numpy(state)
    # plt.imshow(state[3])
    # plt.show()
    # time.sleep(2)
    return state.unsqueeze(0)