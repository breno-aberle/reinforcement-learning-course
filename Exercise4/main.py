import gym
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from rbf_agent import Agent as RBFAgent  # Use for Tasks 1-3
from dqn_agent import Agent as DQNAgent  # Task 4
from itertools import count
import torch
from torch.utils.tensorboard import SummaryWriter
from utils import *
from tqdm import tqdm  # progress bar

env_name = "CartPole-v0"
#env_name = "LunarLander-v2"
env = gym.make(env_name)
env.reset()

# Set hyperparameters
# Values for RBF (Tasks 1-3)
glie_a = 50
num_episodes = 1000  # 1000

# Values for DQN  (Task 4)
if "CartPole" in env_name:
    TARGET_UPDATE = 50
    glie_a = 500
    num_episodes = 2000
    hidden = 12
    gamma = 0.95
    replay_buffer_size = 500000
    batch_size = 256
elif "LunarLander" in env_name:
    TARGET_UPDATE = 4
    glie_a = 100
    num_episodes = 2000
    hidden = 64
    gamma = 0.99
    replay_buffer_size = 50000
    batch_size = 64
else:
    raise ValueError("Please provide hyperparameters for %s" % env_name)

# The output will be written to your folder ./runs/CURRENT_DATETIME_HOSTNAME,
# Where # is the consecutive number the script was run
writer = SummaryWriter()

# Get number of actions from gym action space
n_actions = env.action_space.n
state_space_dim = env.observation_space.shape[0]

# Tasks 1-3 - RBF
#agent = RBFAgent(n_actions)

# Task 4 - DQN
agent = DQNAgent(env_name, state_space_dim, n_actions, replay_buffer_size, batch_size, hidden, gamma)



# Training loop
cumulative_rewards = []
for ep in tqdm(range(num_episodes)):
    # Initialize the environment and state
    state = env.reset()
    done = False
    eps = glie_a / (glie_a + ep)
    cum_reward = 0
    while not done:
        # Select and perform an action
        action = agent.get_action(state, eps)
        next_state, reward, done, _ = env.step(action)
        cum_reward += reward

        # Task 1: TODO: Update the Q-values
        #agent.single_update(state, action, next_state, reward, done)
        # Task 2: TODO: Store transition and batch-update Q-values
        agent.store_transition(state, action, next_state, reward, done)
        #agent.update_estimator()
        # Task 4: Update the DQN
        agent.update_network()

        # Move to the next state
        state = next_state
    cumulative_rewards.append(cum_reward)
    writer.add_scalar('Training ' + env_name, cum_reward, ep)
    # Update the target network, copying all weights and biases in DQN
    # Uncomment for Task 4
    if ep % TARGET_UPDATE == 0:
        agent.update_target_network()

    # Save the policy
    # Uncomment for Task 4
    if ep % 1000 == 0:
        torch.save(agent.policy_net.state_dict(), "weights_%s_%d.mdl" % (env_name, ep))

plot_rewards(cumulative_rewards)
print('Complete')
plt.ioff()
plt.show()


# Task 3 - plot the policy

discr = 32  # reasonable of discretization bins
x_min, x_max = -2.4, 2.4  # values from last week
th_min, th_max = -0.3, 0.3

# Create discretization grid
x_grid = np.linspace(x_min, x_max, discr)
th_grid = np.linspace(th_min, th_max, discr)

# create q-grid
q_grid = np.zeros((discr, discr))  # only two-dimensional because only in terms of x and theta; x_dot and theta_dot = 0


# v1 --------------------------------
# for i in range(len(x_grid)):
#     for j in range(len(th_grid)):
#         position = x_grid[i]
#         theta = th_grid[j]
#         state = np.array([position, 0, theta, 0])  # set velocity and angle velocity = 0 like stated in exercise
#         action = agent.get_action(state)
#         q_grid[i][j] = action  # heatmap should plot policy, more specifically action
#
# plt.imshow(q_grid)
# plt.xticks(range(32)[0::4], x_grid.round(2)[0::4])
# plt.yticks(range(32)[0::4], th_grid.round(2)[0::4])
# plt.xlabel('x')
# plt.ylabel('theta')
# plt.show()


# v2 ------------------------------
# x_grid = x_grid.round(2);
# th_grid = th_grid.round(2);
#
# for i, x in enumerate(x_grid):
#     for j, th in enumerate(th_grid):
#         state = np.array([x, 0, th, 0])
#         action = agent.get_action(state)
#         #x, th = discretize_position_velocity(x, th)
#         q_grid[(i, j)] = action
# sns.heatmap(q_grid, cbar=False, xticklabels=x_grid, yticklabels=th_grid)
# plt.show()

