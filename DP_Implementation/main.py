# Authors, Niklas Schmid, Jared Miller, Tristan Zeller,
# Marta Fochesato, Tobias Sutter, John Lygeros 2026
#
# This source code is licensed under the license
# found in the LICENSE file in the root directory of this source tree.

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import os
import parameters
from load_world import load_world, debug_plot_world
from generate_transition_data import generate_transition_data, debug_plot_transition_data
from dynamic_programming import execute_DP, plot_value_function_and_policy, simulate_rollouts_of_policy, evaluate_policy_safety_via_value_iteration, evaluate_policy_safety_via_value_iteration_gpu, simulate_rollouts_gpu, plot_trajectory_sweep_animation
import time
from plotting_functions import plot_trajectories, plot_cost_of_jcc_and_only_safe_trajectories

# Structure:
# - Generate trajectories for every action (2D unicycle)
# - Run DP for every state and action by computing empirical expectations from the trajectory data

if __name__ == '__main__':
    # Loading safe and target set
    safe_set, target_set = load_world(parameters.TRACK_NAME)
    map = (safe_set, target_set)
    #debug_plot_world(safe_set, target_set)

    # Get one-step transitions for every action from the origin
    if parameters.RUN_OPTIONS['GENERATE_NEW_TRAJECTORY_DATA']:
        transition_data, action_set = generate_transition_data(parameters.sim_params)
        np.save("data/transition_data.npy", transition_data)
        np.save("data/action_set.npy", action_set)
    else:
        print("Loading transition data from file.")
        transition_data = np.load("data/transition_data.npy")
        action_set = np.load("data/action_set.npy")
    #debug_plot_transition_data(transition_data)


    # Execute DP to compute value function and policy for every state in the track and every action in the action set
    # by computing empirical expectations from the trajectory data.
    # Store the value function and policy in a numpy array and save to file.
    initial_state = parameters.DP_PARAMS['INITIAL_STATE']
    if parameters.RUN_OPTIONS['RUN_DP']:
        print("Running DP to compute policy and value function.")
        if parameters.DP_PARAMS['OBJECTIVE'] == 'RA':
            print("Objective: Reach-Avoid")
            pi, V = execute_DP(transition_data, initial_state, map,action_set)
            np.save("data/policy.npy", pi)
            np.save("data/value_function.npy", V)
        elif parameters.DP_PARAMS['OBJECTIVE'] == 'JCC_classic_sweep':
            print("Objective: Classical JCC")
            pi = {}
            V = {}
            for lambda_val in parameters.DP_PARAMS['LAMBDA_LIST']:
                print(f"Lambda: {lambda_val}")
                pi[lambda_val], V[lambda_val] = execute_DP(transition_data, initial_state, map,action_set, lambda_val=lambda_val)
                initial_state_idx = (int(initial_state[0]), int(initial_state[1]), 0)  # Assuming initial cost is 0
                # Evaluate safety of policy via value iteration:
                safety_value = evaluate_policy_safety_via_value_iteration_gpu(transition_data, initial_state, map,
                                                                              pi[lambda_val], lambda_val)
                print("For lambda = ", lambda_val, "safety at initial state via DP: ", safety_value)
                np.save(f"data/policy_lambda_{lambda_val}.npy", pi[lambda_val])
                np.save(f"data/value_function_lambda_{lambda_val}.npy", V[lambda_val])
        elif parameters.DP_PARAMS['OBJECTIVE'] == 'JCC_only_safe' or parameters.DP_PARAMS['OBJECTIVE'] == 'JCC_classic':
            # measure time
            start_time = time.time()

            (pi_up,pi_down), (V_up, V_down), epsilon = execute_DP(transition_data, initial_state, map,action_set, alpha=parameters.DP_PARAMS['alpha'])

            end_time = time.time()
            print("Time taken for DP: ", end_time - start_time, " seconds.")

            # np.save(f"data/policy_lambda_{lambda_val}.npy", pi[lambda_val])
            # np.save(f"data/value_function_lambda_{lambda_val}.npy", V[lambda_val])
            # Safe data is mode specific directory eg np.save(f"data/JCC_classic/policy_lambda_{lambda_val}.npy", pi[lambda_val])

            np.save(f"data/{parameters.DP_PARAMS['OBJECTIVE']}/{parameters.TRACK_NAME}/policy_up.npy", pi_up)
            np.save(f"data/{parameters.DP_PARAMS['OBJECTIVE']}/{parameters.TRACK_NAME}/policy_down.npy", pi_down)
            np.save(f"data/{parameters.DP_PARAMS['OBJECTIVE']}/{parameters.TRACK_NAME}/value_function_up.npy", V_up)
            np.save(f"data/{parameters.DP_PARAMS['OBJECTIVE']}/{parameters.TRACK_NAME}/value_function_down.npy", V_down)
            np.save(f"data/{parameters.DP_PARAMS['OBJECTIVE']}/{parameters.TRACK_NAME}/epsilon.npy", epsilon)
            pi = (pi_up, pi_down, epsilon)

        elif parameters.DP_PARAMS['OBJECTIVE'] == 'JCC_onlySafe_sweep' or parameters.DP_PARAMS['OBJECTIVE'] == 'JCC_classic_sweep':
            print("Objective: Only Safe JCC")
            pi = {}
            V = {}
            for lambda_val in parameters.DP_PARAMS['LAMBDA_LIST']:
                print(f"Lambda: {lambda_val}")
                # Measure computation time:
                start_time = time.time()
                pi[lambda_val], V[lambda_val] = execute_DP(transition_data, initial_state, map, action_set, lambda_val=lambda_val)
                end_time = time.time()
                # Print in big red letters into console computation time:
                print(
                    "\033[91m" + "Hey Tristan, I need this computation time for the dp recursion: ",
                    end_time - start_time, " seconds." + "\033[0m")

                initial_state_idx = (int(initial_state[0]), int(initial_state[1]), 0)  # Assuming initial cost is 0
                # Evaluate safety of policy via value iteration:
                safety_value = evaluate_policy_safety_via_value_iteration_gpu(transition_data, initial_state, map,
                                                                              pi[lambda_val], lambda_val)
                 #safety_value = simulate_rollouts_of_policy(V[lambda_val], pi[lambda_val], action_set, map, initial_state, num_rollouts=1000)
                print("For lambda = ", lambda_val, "safety at initial state via DP: ", safety_value)

                #np.save(f"data/policy_lambda_{lambda_val}.npy", pi[lambda_val])
                #np.save(f"data/value_function_lambda_{lambda_val}.npy", V[lambda_val])
                # Safe data is mode specific directory eg np.save(f"data/JCC_classic/policy_lambda_{lambda_val}.npy", pi[lambda_val])
                np.save(f"data/{parameters.DP_PARAMS['OBJECTIVE']}/{parameters.TRACK_NAME}/policy_lambda_{lambda_val}.npy", pi[lambda_val])
                np.save(f"data/{parameters.DP_PARAMS['OBJECTIVE']}/{parameters.TRACK_NAME}/value_function_lambda_{lambda_val}.npy", V[lambda_val])
    else:
        if parameters.DP_PARAMS['OBJECTIVE'] == 'RA':
            print("Objective: Reach-Avoid")
            print("Loading policy and value function from file.")

            pi = np.load(f"data/{parameters.DP_PARAMS['OBJECTIVE']}/{parameters.TRACK_NAME}/policy.npy")
            V = np.load(f"data/{parameters.DP_PARAMS['OBJECTIVE']}/{parameters.TRACK_NAME}/value_function.npy")
        elif parameters.DP_PARAMS['OBJECTIVE'] == 'JCC_only_safe' or parameters.DP_PARAMS['OBJECTIVE'] == 'JCC_classic':
            print("Objective: "+ parameters.DP_PARAMS['OBJECTIVE'])
            # Check if file policy_up exists:
            if not ("policy_up.npy" in os.listdir(f"data/{parameters.DP_PARAMS['OBJECTIVE']}/{parameters.TRACK_NAME}")):
                pi_down = np.load(f"data/{parameters.DP_PARAMS['OBJECTIVE']}/{parameters.TRACK_NAME}/policy.npy")
                pi_up = np.load(f"data/{parameters.DP_PARAMS['OBJECTIVE']}/{parameters.TRACK_NAME}/policy.npy")
                V_up = np.load(f"data/{parameters.DP_PARAMS['OBJECTIVE']}/{parameters.TRACK_NAME}/value_function.npy")
                V_down = np.load(f"data/{parameters.DP_PARAMS['OBJECTIVE']}/{parameters.TRACK_NAME}/value_function.npy")
                epsilon = 1
            else:
                pi_up = np.load(f"data/{parameters.DP_PARAMS['OBJECTIVE']}/{parameters.TRACK_NAME}/policy_up.npy")
                pi_down = np.load(f"data/{parameters.DP_PARAMS['OBJECTIVE']}/{parameters.TRACK_NAME}/policy_down.npy")
                V_up = np.load(f"data/{parameters.DP_PARAMS['OBJECTIVE']}/{parameters.TRACK_NAME}/value_function_up.npy")
                V_down = np.load(f"data/{parameters.DP_PARAMS['OBJECTIVE']}/{parameters.TRACK_NAME}/value_function_down.npy")
                epsilon = np.load(f"data/{parameters.DP_PARAMS['OBJECTIVE']}/{parameters.TRACK_NAME}/epsilon.npy")
            pi = (pi_up, pi_down, epsilon)
            V = (V_up, V_down)

        elif parameters.DP_PARAMS['OBJECTIVE'] == 'JCC_classic_sweep' or parameters.DP_PARAMS['OBJECTIVE'] == 'JCC_onlySafe_sweep':

            #print("Running animation of Sweep")
            #plot_trajectory_sweep_animation(map)
            #print("Finished animation of Sweep")

            #data_only_safe, data_jcc_classic = plot_cost_of_jcc_and_only_safe_trajectories(transition_data, initial_state)
            print("Objective: "+ parameters.DP_PARAMS['OBJECTIVE'])
            pi = {}
            V = {}
            for lambda_val in parameters.DP_PARAMS['LAMBDA_LIST']:
                pi[lambda_val] = np.load(f"data/{parameters.DP_PARAMS['OBJECTIVE']}/{parameters.TRACK_NAME}/policy_lambda_{lambda_val}.npy")
                V[lambda_val] = np.load(f"data/{parameters.DP_PARAMS['OBJECTIVE']}/{parameters.TRACK_NAME}/value_function_lambda_{lambda_val}.npy")
                # Evaluate safety of policy via value iteration:transition_data, initial_state, maps, pi, lambda_val
                #safety_value = evaluate_policy_safety_via_value_iteration(transition_data, initial_state, map, pi[lambda_val], lambda_val)
                #safety_value = evaluate_policy_safety_via_value_iteration_gpu(transition_data, initial_state, map, pi[lambda_val], lambda_val)
                safety_value, _ = simulate_rollouts_gpu(pi[lambda_val], action_set, map, initial_state, num_rollouts=1000000)
                print("For lambda = ", lambda_val, "safety at initial state via DP: ", safety_value)


    if parameters.DP_PARAMS['OBJECTIVE'] == 'RA' or parameters.DP_PARAMS['OBJECTIVE'] == 'JCC_only_safe' or parameters.DP_PARAMS['OBJECTIVE'] == 'JCC_classic':
        #plot_value_function_and_policy(V, pi, action_set, map)
        safety_value, trajectories = simulate_rollouts_gpu(pi, action_set, map, initial_state, num_rollouts=1000)
        np.save(f"data/{parameters.DP_PARAMS['OBJECTIVE']}/{parameters.TRACK_NAME}/trajectories.npy", trajectories)
        #plot_trajectories(animate_evolution=True)
        plot_trajectories()
    else:
        for lambda_val in parameters.DP_PARAMS['LAMBDA_LIST']:
            print(f"Lambda: {lambda_val}")
            #plot_value_function_and_policy(V[lambda_val], pi[lambda_val], action_set, map)
            safety_value, trajectories = simulate_rollouts_gpu(pi[lambda_val], action_set, map, initial_state, num_rollouts=1000)
            np.save(f"data/{parameters.DP_PARAMS['OBJECTIVE']}/{parameters.TRACK_NAME}/trajectories_lambda_{lambda_val}.npy", trajectories)
            plot_trajectories(lambda_val)
