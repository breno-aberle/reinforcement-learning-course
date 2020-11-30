import torch
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from utils import discount_rewards


class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = 64
        self.fc1 = torch.nn.Linear(state_space, self.hidden)
        self.fc2_mean = torch.nn.Linear(self.hidden, action_space)
        self.sigma = 5  # TODO: Implement accordingly (T1, T2) # T1: 5 # T2 a): 10
        #self.sigma = torch.nn.Parameter(torch.tensor([10.]))  # TODO: T2 b): torch.nn.Parameter(torch.tensor([10.]))
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x, episode_number):
        x = self.fc1(x)
        x = F.relu(x)
        action_mean = self.fc2_mean(x)
        sigma = self.sigma  # TODO: Is it a good idea to leave it like this? # T1 + T2 b): self.sigma
        #sigma = self.sigma * np.exp((-1) * 5e-4 * episode_number)  # TODO: T2 a): exponentially decaying variance

        # TODO: Instantiate and return a normal distribution
        # with mean mu and std of sigma (T1)
        action_dist = Normal(action_mean, sigma)

        return action_dist


class Agent(object):
    def __init__(self, policy):
        self.train_device = "cpu"
        self.policy = policy.to(self.train_device)
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=5e-3)
        self.gamma = 0.98
        self.states = []
        self.action_probs = []
        self.rewards = []
        self.baseline = 20 # TODO: T1 b) 20

    def episode_finished(self, episode_number):
        action_probs = torch.stack(self.action_probs, dim=0).to(self.train_device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
        self.states, self.action_probs, self.rewards = [], [], []

        # TODO: Compute discounted rewards (use the discount_rewards function)
        discounted_r = discount_rewards(rewards, self.gamma)
        #discounted_r -= torch.mean(discounted_r)  # for Task 1 c)
        #discounted_r /= torch.std(discounted_r)

        # TODO: Compute the optimization term (T1)
        #weighted_probs = action_probs * discounted_r  # REINFORCE without baseline # T1 a)+c) & T2 # from exercise 1
        weighted_probs = action_probs * (discounted_r - self.baseline)  # REINFORCE with baseline # T1 b)
        loss = torch.mean((-1) * weighted_probs)  # needs to be multiplied by (-1) because gradient is a maximize function, so we maximize negative loss to converge it towards zero

        # TODO: Compute the gradients of loss w.r.t. network parameters (T1)
        loss.backward()  # like in policy gradient tutorial 2.2 Automatic differentiation

        # TODO: Update network parameters using self.optimizer and zero gradients (T1)
        self.optimizer.step()  # like in policy gradient tutorial 2.3 Using optimizers
        self.optimizer.zero_grad()

    def get_action(self, observation, episode_number, evaluation=False):
        """
        params:
            observation: initial state values of the environment which can be observed after resetting the environment
            episode_number: iteration step of loop through episodes
            evaluation: true for testing, false for training
        returns:
            action:
            act_log_prob:
        """
        x = torch.from_numpy(observation).float().to(self.train_device)

        # TODO: Pass state x through the policy network (T1)
        aprob = self.policy.forward(x, episode_number)

        # TODO: Return mean if evaluation, else sample from the distribution
        # returned by the policy (T1)
        if evaluation:
            action = aprob.mean()
        else:
            action = aprob.sample()


        # TODO: Calculate the log probability of the action (T1)
        act_log_prob = aprob.log_prob(action)

        return action, act_log_prob

    def store_outcome(self, observation, action_prob, action_taken, reward):
        self.states.append(observation)
        self.action_probs.append(action_prob)
        self.rewards.append(torch.Tensor([reward]))

