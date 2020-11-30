import torch
import gym
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys
from agent import Agent, Policy
from utils import get_space_dim


# Parse script arguments
def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", "-t", type=str, default=None,
                        help="Model to be tested")
    parser.add_argument("--env", type=str, default="CartPole-v0",
                        help="Environment to use")
    parser.add_argument("--train_episodes", type=int, default=500,
                        help="Number of episodes to train for")
    parser.add_argument("--render_training", action='store_true',
                        help="Render each frame during training. Will be slower.")
    parser.add_argument("--render_test", action='store_true', help="Render test")
    parser.add_argument("--position", type=float, default=0, help="pole shall balance around that position")
    return parser.parse_args(args)


# Policy training function
def train(agent, env, train_episodes, early_stop=True, render=False, silent=False, train_run_id=0):
    # Arrays to keep track of rewards
    reward_history, timestep_history = [], []
    average_reward_history = []

    # Run actual training
    for episode_number in range(train_episodes):
        reward_sum, timesteps = 0, 0
        done = False
        # Reset the environment and observe the initial state
        observation = env.reset()

        # Loop until the episode is over
        while not done:
            # Get action from the agent
            action, action_probabilities = agent.get_action(observation)
            previous_observation = observation

            # Perform the action on the environment, get new state and reward
            observation, reward, done, info = env.step(action)

            # TODO: Task 1 - change the reward function
            reward = new_reward(observation)

            # Store action's outcome (so that the agent can improve its policy)
            agent.store_outcome(previous_observation, action_probabilities, action, reward)

            # Draw the frame, if desired
            if render:
                env.render()

            # Store total episode reward
            reward_sum += reward
            timesteps += 1

        if not silent:
            print("Episode {} finished. Total reward: {:.3g} ({} timesteps)"
                  .format(episode_number, reward_sum, timesteps))

        # Bookkeeping (mainly for generating plots)
        reward_history.append(reward_sum)
        timestep_history.append(timesteps)
        if episode_number > 100:
            avg = np.mean(reward_history[-100:])
        else:
            avg = np.mean(reward_history)
        average_reward_history.append(avg)

        # If we managed to stay alive for 15 full episodes, assume it's learned
        # (in the default setting)
        if early_stop and np.mean(timestep_history[-15:]) == env._max_episode_steps:
            if not silent:
                print("Looks like it's learned. Finishing up early")
            break

        # Let the agent do its magic (update the policy)
        agent.episode_finished(episode_number)

    # Store the data in a Pandas dataframe for easy visualization
    data = pd.DataFrame({"episode": np.arange(len(reward_history)),
                         "train_run_id": [train_run_id]*len(reward_history),
                         "reward": reward_history,
                         "mean_reward": average_reward_history})
    return data


# Function to test a trained policy
def test(agent, env, episodes, render=False):
    test_reward, test_len = 0, 0
    velocity_history = [] # plot velocity
    for ep in range(episodes):
        done = False
        observation = env.reset()
        while not done:
            # Similar to the training loop above -
            # get the action, act on the environment, save total reward
            # (evaluation=True makes the agent always return what it thinks to be
            # the best action - there is no exploration at this point)
            action, _ = agent.get_action(observation, evaluation=True)
            observation, reward, done, info = env.step(action)

            velocity_history.append(observation[1]) # plot velocity
            # TODO: New reward function
            reward = new_reward(observation)
            if render:
                env.render()
            test_reward += reward
            test_len += 1
    print("Average test reward:", test_reward/episodes, "episode length:", test_len/episodes)

    # plot velocity
    #print(velocity_history[0:200])
    plt.plot(velocity_history[0:500])
    plt.xlabel('timesteps')
    plt.ylabel('velocity')
    plt.title('Velocity history (CartPole-v0)')
    plt.show()



# TODO: Definition of the modified reward function
def new_reward(state, version=1):
    """ Reward function to meet requirements for tasks

    Parameters:
        state: a four element [position   velocity   angular   angular_velocity]
        version: (1) balance pole close to center, (2) balance pole in arbitrary point of the screen, (3) drifting from left to right

    Returns:
        reward: return reward for each timestep
    """
    if (version == 1):
        #reward = 1 - abs(state[0])
        if (state[0] == 0):
            reward = 4
        else:
            reward = min(4, 1/np.abs(state[0]))

    elif (version == 2):
        target_position = args.position
        if (state[0] == target_position):
            reward = 4
        else:
            reward = min(4, 1/np.abs(state[0] - target_position))
            #reward = min(4, 1 / (state[0] - target_position)**2)

    elif (version == 3):
        velocity = state[1]

        if (-1.8 <= state[0] <= 1.8):
            reward = 2 + abs(velocity)*2
        elif (state[0] < -1.8):
            if (state[1] > 0):
                reward = 1
            else:
                reward = -1 # -1 # maybe -(velocity**2)
        elif (state[0] > 1.8):
            if (state[1] < 0):
                reward = 1
            else:
                reward = -1
        else: # just to be sure all cases are met
            reward = 0.001

    else: # version 4 or higher (not 1,2, or 3)
        reward = 1

    return reward


# The main function
def main(args):
    # Create a Gym environment
    env = gym.make(args.env)

    # Exercise 1
    # TODO: For CartPole-v0 - maximum episode length
    env._max_episode_steps = 200  # espisode length: how many steps in one episode

    # Get dimensionalities of actions and observations
    action_space_dim = get_space_dim(env.action_space)
    observation_space_dim = get_space_dim(env.observation_space)

    # Instantiate agent and its policy
    policy = Policy(observation_space_dim, action_space_dim)
    agent = Agent(policy)

    # Print some stuff
    print("Environment:", args.env)
    print("Training device:", agent.train_device)
    print("Observation space dimensions:", observation_space_dim)
    print("Action space dimensions:", action_space_dim)

    # If no model was passed, train a policy from scratch.
    # Otherwise load the policy from the file and go directly to testing.
    if args.test is None:
        training_history = train(agent, env, args.train_episodes, False, args.render_training)

        # Save the model
        model_file = "%s_params.ai" % args.env
        torch.save(policy.state_dict(), model_file)
        print("Model saved to", model_file)

        # Plot rewards
        sns.lineplot(x="episode", y="reward", data=training_history)
        sns.lineplot(x="episode", y="mean_reward", data=training_history)
        plt.legend(["Reward", "100-episode average"])
        plt.title("Reward history (%s)" % args.env)
        plt.show()
        print("Training finished.")
    else:
        print("Loading model from", args.test, "...")
        state_dict = torch.load(args.test)
        policy.load_state_dict(state_dict)
        print("Testing...")
        test(agent, env, args.train_episodes, args.render_test)


# Entry point of the script
if __name__ == "__main__":
    args = parse_args()
    main(args)

