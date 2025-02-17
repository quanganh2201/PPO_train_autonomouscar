#!/usr/bin/env python

# Copyright (c) 2019: Jianyu Chen (jianyuchen@berkeley.edu).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import gym
import gym_carla
import carla
import torch
import warnings
import os
import sys
from collections import deque

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import gym
module_path = os.path.abspath(os.path.join('../'))
if module_path not in sys.path:
    sys.path.append(module_path)
from scripts.ppo import PPO


def main():
  # parameters for the gym_carla environment
  params = {
    'number_of_vehicles': 100,
    'number_of_walkers': 0,
    'display_size': 256,  # screen size of bird-eye render
    'max_past_step': 1,  # the number of past steps to draw
    'dt': 0.1,  # time interval between two frames
    'discrete': False,  # whether to use discrete control space
    'discrete_acc': [-3.0, 0.0, 3.0],  # discrete value of accelerations
    'discrete_steer': [-0.2, 0.0, 0.2],  # discrete value of steering angles
    'continuous_accel_range': [-3.0, 3.0],  # continuous acceleration range
    'continuous_steer_range': [-0.3, 0.3],  # continuous steering angle range
    'ego_vehicle_filter': 'vehicle.lincoln*',  # filter for defining ego vehicle
    'port': 2000,  # connection port
    'town': 'Town03',  # which town to simulate
    'task_mode': 'random',  # mode of the task, [random, roundabout (only for Town03)]
    'max_time_episode': 1000,  # maximum timesteps per episode
    'max_waypt': 12,  # maximum number of waypoints
    'obs_range': 32,  # observation range (meter)
    'lidar_bin': 0.125,  # bin size of lidar sensor (meter)
    'd_behind': 12,  # distance behind the ego vehicle (meter)
    'out_lane_thres': 2.0,  # threshold for out of lane
    'desired_speed': 8,  # desired speed (m/s)
    'max_ego_spawn_times': 200,  # maximum times to spawn ego vehicle
    'display_route': True,  # whether to render the desired route
    'pixor_size': 64,  # size of the pixor labels
    'pixor': False,  # whether to output PIXOR observation
  }

  # Set gym-carla environment
  env = gym.make('carla-v0', params=params)
  rewards_queue = deque()
  rewards_ma = list()

  max_episodes = 750
  max_trajectory_length = 200
  discount_factor = 0.99
  update_timesteps = 500
  num_timesteps = 0


  ppo = PPO(
    action_dims = env.action_space.n,
    state_dims = env.observation_space.shape[0],
    actor_lr = 0.001,
    critic_lr = 0.001,
    epsilon_clipping = 0.1,
    optimization_steps = 3,
    discount_rate = 0.99
  )
  obs = env.reset()
  warnings.simplefilter('ignore')
  for epi in tqdm(range(max_episodes)):
      state = env.reset()
      total_reward = 0.0

      for t in range(max_trajectory_length):
          num_timesteps += 1
          action, log_prob, state_value = ppo.select_action(torch.from_numpy(state.astype(np.float32)))
          state, reward, is_done, _ = env.step(action)
          ppo.record(state, action, log_prob, state_value, reward, is_done)
          total_reward += reward

          if is_done or update_timesteps == num_timesteps:
              break

      if update_timesteps == num_timesteps:
          ppo.update()
          num_timesteps = 0
          pass

      solved = total_reward > 195.0
      if len(rewards_queue) > 50:
          rewards_queue.popleft()

      rewards_queue.append(total_reward)
      mean_reward = np.mean(rewards_queue)
      rewards_ma.append(mean_reward)

      if solved:
          break
  ppo.save('./models/aws.pt')
if __name__ == '__main__':
  main()