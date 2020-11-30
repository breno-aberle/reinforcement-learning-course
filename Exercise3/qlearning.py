import gym
import numpy as np
from matplotlib import pyplot as plt

np.random.seed(123)

env = gym.make('CartPole-v0')
env.seed(321)

episodes = 20000
test_episodes = 10
num_of_actions = 2

# Reasonable values for Cartpole discretization
discr = 16
x_min, x_max = -2.4, 2.4 # boundaries along 1D-coordinate axis
v_min, v_max = -3, 3 # velocity
th_min, th_max = -0.3, 0.3 # angle
av_min, av_max = -4, 4 # angle velocity

# For LunarLander, use the following values:
#         [  x     y  xdot ydot theta  thetadot cl  cr
# s_min = [ -1.2  -0.3  -2.4  -2  -6.28  -8       0   0 ]
# s_max = [  1.2   1.2   2.4   2   6.28   8       1   1 ]

# Parameters
gamma = 0.98
alpha = 0.1
target_eps = 0.1
a = np.rint( (target_eps * episodes) / (1 - target_eps) )  # TODO: Set the correct value. Which is b in the lecture slide
initial_q = 50  # T3: Set to 50

# Create discretization grid
x_grid = np.linspace(x_min, x_max, discr)
print("x_grid.shape: ", x_grid.shape)
v_grid = np.linspace(v_min, v_max, discr)
th_grid = np.linspace(th_min, th_max, discr)
av_grid = np.linspace(av_min, av_max, discr)

q_grid = np.zeros((discr, discr, discr, discr, num_of_actions)) + initial_q  # grid contains state-action-values q(S,A)


def find_nearest(array, value):
    return np.argmin(np.abs(array - value))


def get_cell_index(state):
    """
    Find grid cell in discretization grid
    """
    x = find_nearest(x_grid, state[0]) # state attribute will get allocated to closest grid cell
    v = find_nearest(v_grid, state[1])
    th = find_nearest(th_grid, state[2])
    av = find_nearest(av_grid, state[3])
    return x, v, th, av


def get_action(state, q_values, greedy=False):
    """ Returns action in epsilon-greedy manner
    Parameters:
        q_values: in this case the q_grid
    """
    # TODO: Implement epsilon-greedy


    state_index = get_cell_index(state)  # get the indices of the respective state values to be able to access the q_grid

    if greedy:
        action = np.argmax( q_values[state_index] ) # greedy
    else:
        action_random = np.random.randint(0,2) # exploration (random)
        action_greedy = np.argmax( q_values[state_index] ) # greedy
        action = np.random.choice([action_greedy, action_random], p=[(1-epsilon),epsilon])
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

    epsilon = 0 # a / (a + ep) # 0.2 # a / (a + ep)  # TODO: T1: GLIE/constant, T3: Set to 0

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
values_averaged = np.average(values, axis=1) # in the exercise it is stated to average velocity (second column: axis=1) and angle velocity (fourth column: axis=3)
values_averaged = np.average(values_averaged, axis=-1)
#plt.imshow(values_averaged)
#plt.show()

# Draw plots
plt.plot(ep_lengths)
plt.plot(epl_avg)
plt.legend(["Episode length", "500 episode average"])
plt.title("Episode lengths")
plt.show()

