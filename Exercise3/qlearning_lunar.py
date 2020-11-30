import gym
import numpy as np
from matplotlib import pyplot as plt

np.random.seed(123)

env = gym.make('LunarLander-v2')
env.seed(321)

episodes = 20000
test_episodes = 10
num_of_actions = 4

# Reasonable values for Cartpole discretization
discr = 16

x_min, x_max = -1.2, 1.2
y_min, y_max = -0.3, 1.2
xdot_min, xdot_max = -2.4, 2.4
ydot_min, ydot_max = -2, 2
theta_min, theta_max = -6.28, 6.28
thetadot_min, thetadot_max = -8, 8
cl_min, cl_max = 0, 1
cr_min, cr_max = 0, 1

# For LunarLander, use the following values:
#         [  x     y    xdot ydot theta  thetadot cl  cr
# s_min = [ -1.2  -0.3  -2.4  -2  -6.28  -8       0   0 ]
# s_max = [  1.2   1.2   2.4   2   6.28   8       1   1 ]

# Parameters
gamma = 0.98
alpha = 0.1
target_eps = 0.1
a = np.rint( (target_eps * episodes) / (1 - target_eps) )  # TODO: Set the correct value. Which is b in the lecture slide
initial_q = 0  # T3: Set to 50

# Create discretization grid
x_grid = np.linspace(x_min, x_max, discr)
y_grid = np.linspace(y_min, y_max, discr)
xdot_grid = np.linspace(xdot_min, xdot_max, discr)
ydot_grid = np.linspace(ydot_min, ydot_max, discr)
th_grid = np.linspace(theta_min, theta_max, discr)
thdot_grid = np.linspace(thetadot_min, thetadot_max, discr)
cl_grid = np.linspace(cl_min, cl_max, 2) # because binary variable (0 or 1)
cr_grid = np.linspace(cr_min, cr_max, 2)

q_grid = np.zeros((discr, discr, discr, discr, discr, discr, 2, 2, num_of_actions), dtype='float32') + initial_q  # grid contains state-action-values q(S,A)


def find_nearest(array, value):
    return np.argmin(np.abs(array - value))


def get_cell_index(state):
    x = find_nearest(x_grid, state[0])
    y = find_nearest(y_grid, state[1])
    xdot = find_nearest(xdot_grid, state[2])
    ydot = find_nearest(ydot_grid, state[3])
    theta = find_nearest(th_grid, state[4])
    thetadot = find_nearest(thdot_grid, state[5])
    cl = find_nearest(cl_grid, state[6])
    cr = find_nearest(cr_grid, state[7])
    return x, y, xdot, ydot, theta, thetadot, cl, cr


def get_action(state, q_values, greedy=False):
    # TODO: Implement epsilon-greedy
    state_index = get_cell_index(state)  # get the indices of the respective state values to be able to access the q_grid

    if greedy:
        action = np.argmax(q_values[state_index])  # greedy
    else:
        action_random = np.random.randint(0, 4)  # exploration (random)
        action_greedy = np.argmax(q_values[state_index])  # greedy
        action = np.random.choice([action_greedy, action_random], p=[(1 - epsilon), epsilon])
    return action



def update_q_value(old_state, action, new_state, reward, done, q_array):
    # TODO: Implement Q-value update
    old_cell_index = get_cell_index(old_state)
    new_cell_index = get_cell_index(new_state)

    # update
    if done:
        q_array[old_cell_index][action] = q_array[old_cell_index][action] + alpha * (reward + gamma * 0 - q_array[old_cell_index][action])
    else:
        q_array[old_cell_index][action] = q_array[old_cell_index][action] + alpha * (reward + gamma * np.max(q_array[new_cell_index]) - q_array[old_cell_index][action])


q_grid0 = q_grid.copy() # variables to plot the heatmaps
q_grid1 = None
q_grid10000 = None

# Training loop
ep_lengths, epl_avg = [], []
for ep in range(episodes+test_episodes):
    test = ep > episodes
    state, done, steps = env.reset(), False, 0

    epsilon = a / (a + ep) # 0.2 # a / (a + ep)  # TODO: T1: GLIE/constant, T3: Set to 0

    while not done:
        action = get_action(state, q_grid, greedy=test)

        new_state, reward, done, _ = env.step(action)

        if not test: # training
            update_q_value(state, action, new_state, reward, done, q_grid)
        else: # testing + rendering
            #env.render()
            dummy = 0

        state = new_state
        steps += 1

    if (ep == 1): # for Question1 to plot the heatmap after specific number of iterations
        q_grid1 = q_grid.copy()
    elif (ep == 10000):
        q_grid10000 = q_grid.copy()

    ep_lengths.append(steps)
    epl_avg.append(np.mean(ep_lengths[max(0, ep-500):]))
    if ep % 200 == 0:
        print("Episode {}, average timesteps: {:.2f}".format(ep, np.mean(ep_lengths[max(0, ep-200):])))

# Save the Q-value array
np.save("q_values.npy", q_grid)  # TODO: SUBMIT THIS Q_VALUES.NPY ARRAY

# Calculate the value function
values = np.zeros(q_grid.shape[:-1])  # TODO: COMPUTE THE VALUE FUNCTION FROM THE Q-GRID
values = np.max(q_grid, axis=4) # iterate through all actions and store the highest values of the respective action # according to chapter 3.6 in the book "Sutton Barto" the optimal action-value function is returns the highest action-value q over all possible actions
np.save("value_func.npy", values)  # TODO: SUBMIT THIS VALUE_FUNC.NPY ARRAY

# Plot the heatmap
# TODO: Plot the heatmap here using Seaborn or Matplotlib


# Draw plots
plt.plot(ep_lengths)
plt.plot(epl_avg)
plt.legend(["Episode length", "500 episode average"])
plt.title("Episode lengths")
plt.show()

