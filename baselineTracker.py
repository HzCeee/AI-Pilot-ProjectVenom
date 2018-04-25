from DroneSimEnv_movingTarget_eval import *
import random
import time
import tensorflow as tf
import baselines.common.tf_util as U
from baselines.ddpg.ddpg import DDPG
from baselines.ddpg.models import Actor, Critic
from baselines.ddpg.memory import Memory
from baselines.ddpg.noise import *
import numpy as np

from collections import deque
import dronesim
# from matplotlib.pyplot as plt

def actiongenerator(obs):
    action, _ = agent.pi(obs, apply_noise=False, compute_Q=True)
    return action

env = DroneSimEnv()
itetime = 1000

normalize_returns = False
normalize_observations = True
critic_l2_reg = 1e-2
batch_size = 64
actor_lr = 1e-4
critic_lr = 1e-3
popart = False
gamma = 0.99
tau = 0.01
reward_scale = 1.
clip_norm = None
layer_norm = True

action_noise = None
param_noise = None
noise_type = 'adaptive-param_0.2'  # choices are adaptive-param_xx, ou_xx, normal_xx, none
for current_noise_type in noise_type.split(','):
    current_noise_type = current_noise_type.strip()
    if current_noise_type == 'none':
        pass
    elif 'adaptive-param' in current_noise_type:
        _, stddev = current_noise_type.split('_')
        param_noise = AdaptiveParamNoiseSpec(initial_stddev=float(stddev), desired_action_stddev=float(stddev))
    elif 'normal' in current_noise_type:
        _, stddev = current_noise_type.split('_')
        action_noise = NormalActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
    elif 'ou' in current_noise_type:
        _, stddev = current_noise_type.split('_')
        action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
    else:
        raise RuntimeError('unknown noise type "{}"'.format(current_noise_type))

nb_actions = env.action_space.shape[-1]
memory = Memory(limit=int(1e6), action_shape=env.action_space.shape, observation_shape=env.observation_space.shape)
critic = Critic(layer_norm=layer_norm)
actor = Actor(nb_actions, layer_norm=layer_norm)

tf.reset_default_graph()

agent = DDPG(actor, critic, memory, env.observation_space.shape, env.action_space.shape,
    gamma=gamma, tau=tau, normalize_returns=normalize_returns, normalize_observations=normalize_observations,
    batch_size=batch_size, action_noise=action_noise, param_noise=param_noise, critic_l2_reg=critic_l2_reg,
    actor_lr=actor_lr, critic_lr=critic_lr, enable_popart=popart, clip_norm=clip_norm,
    reward_scale=reward_scale)

max_iteration = 100
step_number = []
success = []
reason = {1:0, 2:0, 3:0}

queue_length = 8

coordinate_queue = None
distance_queue = None
in_fov_queue = None

with U.single_threaded_session() as sess:
    agent.initialize(sess)

    saver = tf.train.Saver()
    saver.restore(tf.get_default_session(), '/home/projectvenom/Documents/AIPilot/AIPilot-ProjectVenom-master/model_mv/exp3/Exp5_mv_best') 
    
    iteration = 0
    success_number = 0
    while iteration < max_iteration:
        iteration += 1
        print(iteration)
        # agent.reset()

        # print('error 1')

        obs = env.reset()

        position_hunter, orientation_hunter, acc_hunter, position_target, orientation_target, acc_target, thrust_hunter, velocity_hunter, _ = dronesim.siminfo()
        static_position_target = position_target
        static_orientation_target = orientation_target

        # print('error 2')

        done = False
        step = 0

        count = 1

        while not done:
            state, reward, done, dis = env.step(actiongenerator(obs))
            # print("reward: ",reward)
            # print("done: ",done)
            # print("distance: ",dis['distance'])

            if count < 2:
                position_hunter, orientation_hunter, acc_hunter, position_target, orientation_target, acc_target, thrust_hunter, velocity_hunter, _ = dronesim.siminfo()
                absolute_x, absolute_y, area_ratio, target_in_front = dronesim.projection(position_hunter, orientation_hunter, static_position_target, static_orientation_target)

                distance = np.linalg.norm(np.array(position_hunter) - np.array(static_position_target))
                is_in_fov = 1 if target_in_front and (
                    (0 < absolute_x < 256 and 0 < absolute_y < 144) 
                    or (area_ratio * 256 * 144 * distance * distance > 6)
                    ) else 0

                if target_in_front: relative_x, relative_y = absolute_x / 256, absolute_y / 144
                target_coordinate_in_view = np.array((relative_x, relative_y)).flatten() if is_in_fov == 1 else np.array((0, 0))
                
                distance = np.array([distance / 30]) if is_in_fov == 1 else np.array([0])

                if coordinate_queue is None and distance_queue is None and in_fov_queue is None:
                    coordinate_queue = deque([target_coordinate_in_view] * queue_length)
                    distance_queue = deque([distance] * queue_length)
                    in_fov_queue = deque([is_in_fov] * queue_length)
                else:
                    coordinate_queue.append(target_coordinate_in_view)
                    coordinate_queue.popleft()

                    distance_queue.append(distance)
                    distance_queue.popleft()

                    in_fov_queue.append(is_in_fov)
                    in_fov_queue.popleft()
                
                # print(distance_queue)

                coordinate_state = np.concatenate(list(coordinate_queue))
                distance_state = np.concatenate(list(distance_queue))
                in_fov_state = np.array(list(in_fov_queue))

                # define state
                static_state = np.concatenate([np.array([orientation_hunter[0] / 40, orientation_hunter[1] / 40, orientation_hunter[2] / 180]).flatten(),
                                        np.array((thrust_hunter - 4) / 6 - 1).flatten(),
                                        np.array(velocity_hunter).flatten(),
                                        coordinate_state,
                                        distance_state,
                                        in_fov_state
                                        ], 0)

                static_state = tuple(list(static_state))

                count = count + 1
            else:
                position_hunter, orientation_hunter, acc_hunter, position_target, orientation_target, acc_target, thrust_hunter, velocity_hunter, _ = dronesim.siminfo()
                static_position_target = position_target
                static_orientation_target = orientation_target

                tmp_coord_queue = []
                for x_index in range(7, 23, 2):
                    tmp_coord_queue.append(np.array((state[x_index], state[x_index+1])))
                
                tmp_dist_queue = []
                for d_index in range(23, 31):
                    tmp_dist_queue.append(np.array([state[d_index]]))

                coordinate_queue = deque(tmp_coord_queue)
                distance_queue = deque(tmp_dist_queue)
                in_fov_queue = deque(state[31:39])

                count = 1

            obs = static_state
            # obs = state

            # time.sleep(0.05)
            step += 1

            if done:
                # print('step: ', step)
                # print('done')

                reason[dis['reason']] += 1

                if dis['distance'] <= 1:
                    success.append(1)
                    success_number += 1
                    step_number.append(step)
                else:
                    success.append(0)

                # time.sleep(10)
                agent.reset()
                # obs = env.reset()

    env.stop()

# print(success_number/max_iteration)
print('----------------------')
print('average step: ', sum(step_number)/len(step_number), len(step_number), np.mean(step_number), np.var(step_number))
print('----------------------')
print('success rate: ', sum(success)/len(success))
print('----------------------')
print('result: (1 = success, 2 = max distance, 3 = max time)\n', reason)
