# Copyright 2020 (c) Aalto University - All Rights Reserved
# ELEC-E8125 - Reinforcement Learning Course
# AALTO UNIVERSITY
#
#############################################################


import numpy as np
from time import sleep
from sailing import SailingGridworld

epsilon = 10e-4  # TODO: Use this criteria for Task 3

# Set up the environment
env = SailingGridworld(rock_penalty=-2)
value_est = np.zeros((env.w, env.h))
env.draw_values(value_est)


if __name__ == "__main__":
    # Reset the environment
    state = env.reset()

    # Compute state values and the policy
    # TODO: Compute the value function and policy (Tasks 1, 2 and 3)
    value_est, policy = np.zeros((env.w, env.h)), np.zeros((env.w, env.h)) # 15x10 grid

    gamma = 0.9 # discount factor value
    number_iterations = 100
    actions = [0,1,2,3]

    value_est_history = []
    policy_history = []

    for i in range(number_iterations):
        print("iteration step: ", i)

        env.clear_text() # remove previously drawn values

        value_est_copy = value_est.copy() # copy to make sure that values are only taken from current state. In next iteration the upadted value_est will be retrieved
        policy_copy = policy.copy()

        value_est_history.append(value_est_copy)
        policy_history.append(policy_copy)

        for x in range(env.w): # loop through all states/cell in the environment
            for y in range(env.h):
                state_values_of_actions = [] # keep track of calculated state values of each action
                # calculate new state value for every action and add to list above
                #for transitions in env.transitions[x,y]: # the four different actions are 1) .LEFT 2) .DOWN 3) .RIGHT 4) .UP - each transition is a list with three tuples (state, reward, done, probability)
                for action in actions:
                    transitions = env.transitions[x,y,action]
                    state_value_of_action = 0
                    for transition in transitions:
                        state_next = transition[0]
                        reward = transition[1]
                        done = transition[2]
                        probability = transition[3]
                        if (state_next == None):
                            state_value_of_action += 0
                        else:
                            state_value_of_action += probability * (reward + gamma * value_est_copy[state_next])
                    state_values_of_actions.append(state_value_of_action)
                # update value_est and policy
                value_est[x][y] = np.max(state_values_of_actions)
                policy[x][y] = np.argmax(state_values_of_actions)

        max_diff_val = np.max(abs(abs(value_est) - abs(value_est_copy)))
        if (max_diff_val < epsilon):
            print("Converged! Value state converged in iteration: ", i+1)
            break

        max_diff_policy = np.max(abs(policy_copy - policy))
        if (max_diff_policy < epsilon):
            print("Converged! Policy converged in iteration: ", i+1)



        #env.draw_values(value_est) # draw the new calculated values after every iteration
        #env.draw_actions(policy)
        #env.render()


    # Just for my understanding how the data is stored/provided
    #print(env.transitions[3,3][0]) # gives us one of the four actions
    #print(env.transitions[3,3][0][0]) # gives us one of the three tuples
    #print(env.transitions[3, 3][0][0].state) # access entries of tuple
    #print(env.transitions[3, 3][0][0][3])
    #print(env.transitions[6,3])


    # Show the values and the policy
    env.draw_values(value_est)
    env.draw_actions(policy)
    env.render()
    sleep(1)

    # Save the state values and the policy
    fnames = "values.npy", "policy.npy"
    np.save(fnames[0], value_est)
    np.save(fnames[1], policy)
    print("Saved state values and policy to", *fnames)

    # Run a single episode
    # TODO: Run multiple episodes and compute the discounted returns (Task 4)

    number_episodes = 1000
    discounted_return_history = []

    for i in range(number_episodes):
        done = False

        counter_discount = 0
        discounted_return = 0

        while not done:
            # Select a random action
            # TODO: Use the policy to take the optimal action (Task 2)
            action = policy[state]

            # Step the environment
            state, reward, done, _ = env.step(action)

            # calculate discounted reward
            discounted_return += reward * gamma**counter_discount
            counter_discount += 1

            # Render and sleep
            #env.render()
            #sleep(0.5)

        discounted_return_history.append(discounted_return)
        state = env.reset()

    print("discounted return (initial state) - mean: ", np.mean(discounted_return_history))
    print("discounted return (initial state) - std: ", np.std(discounted_return_history))