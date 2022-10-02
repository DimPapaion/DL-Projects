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

def train(env, n_episodes, render=False, model_name = "Dueling_NET"):
    env = gym.wrappers.Monitor(env, './train_model_scale' + model_name, force=True)
    start_optimize = False
    loss = 0.0
    score = 0
    for episode in range(1, n_episodes + 1):
        obs = env.reset()
        state = get_state(obs)
        total_reward = 0.0
        for step in count():
            action = select_action(state)
            if render:
                env.render()
            obs, reward, done, info = env.step(action)
            total_reward += reward

            if not done:
                next_state = get_state(obs)
            else:
                next_state = None
            reward = torch.tensor([reward], device=device)
            memory.push(state, action.to(device), next_state, reward.to(device))
            state = next_state
            if steps_done > INITIAL_MEMORY:
                start_optimize = True
                loss += optimize_model()
                if (steps_done - INITIAL_MEMORY) % 100 == 0:
                    policy_net.loss_list.append(loss / 100)
                    loss = 0.0
            # optimize_model()
            if steps_done % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())
            if done:
                break
            # time.sleep(0.1)
        score += total_reward
        print('Episode {}/{} Step_total {} steps: {} epsilon {} Total reward: {}'.format(episode, n_episodes, steps_done, step + 1, epsilon, total_reward))
        if episode % 10 == 0:
            policy_net.reward_list.append(score/10)
            score = 0
        if episode % 100 == 0 and start_optimize:
          if model_name=="Dueling_NET":
            show(policy_net.loss_list, 1000, "loss per 1000 steps for Dueling_Net", "loss", "steps", "loss_scale_DUNET.png")
            show(policy_net.reward_list, 10, "score per 10 episodes for Dueling_Net", "score", "episodes", "score_scale_DUNET.png")
            torch.save(policy_net, MODEL_PATH)
          else:
            show(policy_net.loss_list, 1000, "loss per 1000 steps for DQN", "loss", "steps", "loss_scale_DQN.png")
            show(policy_net.reward_list, 10, "score per 10 episodes for DQN", "score", "episodes", "score_scale_DQN.png")
            torch.save(policy_net, MODEL_PATH)
        # time.sleep(2)
    env.close()
    return

def test(env, n_episodes, policy, render=True, model_name="DQNet"):

    env = gym.wrappers.Monitor(env, './test_model'+model_name, force=True)
    for episode in range(n_episodes):
        obs = env.reset()
        state = get_state(obs)
        total_reward = 0.0
        for t in count():
            action = policy(state.to(device)).max(1)[1].view(1, 1)
            if render:
                env.render()
            obs, reward, done, info = env.step(action)
            total_reward += reward
            if not done:
                next_state = get_state(obs)
            else:
                next_state = None
            state = next_state
            # print("Episode {} life {} done{}".format(episode + 1, info["ale.lives"], done))
            if done:
                print("Finished Episode {} with steps {} reward {}".format(episode + 1, t + 1, total_reward))
                break
            # time.sleep(0.2)
        # time.sleep(1.0)
    env.close()
    return

def show(y, scale, des, ydes, xdes, path):
    x = [i*scale for i in range(len(y))]
    plt.plot(x, y, 'b-', label=des)
    plt.xlabel(xdes)
    plt.ylabel(ydes)
    plt.legend()
    plt.savefig(path)
    plt.close("all")