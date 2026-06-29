# Authors, Niklas Schmid, Jared Miller, Tristan Zeller,
# Marta Fochesato, Tobias Sutter, John Lygeros 2026
#
# This source code is licensed under the license
# found in the LICENSE file in the root directory of this source tree.


# import numpy as np
# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
# from numba import njit
#
# @njit(parallel=True)
# def gen_transit_data_core(NUM_ACTIONS, NUM_SAMPLES_PER_ACTION, w1_all, w2_all):
#     """
#     Core function to generate transition data, compiled with Numba for performance.
#     """
#     # The actions are 2D with range (u_{1,k},u_{2,k})\in[0,3]\times[0,2\pi]
#     # Sample trajectories for each action in the action set from the initial state (0,0).
#     # The dynamics are
#     #   x_{k+1} = x_k + u_{1,k}[\cos(u_{2,k} + w_{2,k})  \sin(u_{2,k} + w_{2,k})]^T + w_{1,k},
#     # where w_{1,k}=N([0 0]^T, diag(0.4, 0.4)), w_{2,k}=N(0, 0.2)
#
#     # Create numpy array to store one-step transitions
#     transition_data = np.zeros((NUM_ACTIONS, NUM_SAMPLES_PER_ACTION, 2))  # state_dim=2 for (x,y)
#     action_set = np.zeros((NUM_ACTIONS, 2))  # To store the action parameters (u1, u2) for each action
#
#     for action_idx in range(NUM_ACTIONS):
#         u1 = np.random.uniform(0, 3)
#         u2 = np.random.uniform(0, 2 * np.pi)
#         action_set[action_idx, :] = [u1, u2]
#         for sample_idx in range(NUM_SAMPLES_PER_ACTION):
#             w1 = w1_all[action_idx, sample_idx, :]
#             w2 = w2_all[action_idx, sample_idx]
#             x_next = u1 * np.array([np.cos(u2 + w2), np.sin(u2 + w2)]) + w1
#             transition_data[action_idx, sample_idx, :] = x_next
#
#     return transition_data, action_set
#
# def generate_transition_data(SIM_PARAMS):
#     """
#     Generates trajectory data for a 2D unicycle model by sampling trajectories for each action in the action set.
#     :param NUM_ACTIONS: Size of the action set
#     :param NUM_SAMPLES_PER_ACTION: Number of one-step trajectories sampled for each action set
#     :return: A numpy array containing one-step trajectory data of shape (NUM_ACTIONS, NUM_SAMPLES_PER_ACTION, state_dim)
#     """
#
#     NUM_ACTIONS = SIM_PARAMS['NUM_ACTIONS']
#     NUM_SAMPLES_PER_ACTION = SIM_PARAMS['NUM_SAMPLES_PER_ACTION']
#     NOISE_COVARIANCE = SIM_PARAMS['NOISE_COVARIANCE']
#     NOISE_STD = SIM_PARAMS['NOISE_STD']
#
#     # Generate noise for all samples in advance for autocompilation with numba.
#     # This is necessary because numba does not support random number generation inside njit functions.
#     w1_all = np.random.multivariate_normal(mean=[0, 0], cov=NOISE_COVARIANCE, size=(NUM_ACTIONS, NUM_SAMPLES_PER_ACTION))
#     w2_all = np.random.normal(loc=0, scale=NOISE_STD, size=(NUM_ACTIONS, NUM_SAMPLES_PER_ACTION))
#
#     transition_data, action_set = gen_transit_data_core(NUM_ACTIONS, NUM_SAMPLES_PER_ACTION, w1_all, w2_all)
#
#     return transition_data, action_set
#
# def debug_plot_transition_data(transition_data):
#     """Plots the generated trajectory data for 10 randomly selected actions."""
#     plt.figure(figsize=(10, 10))
#     for action_idx in range(min(10, transition_data.shape[0])):  # Plot trajectories for up to 10 actions
#         samples = transition_data[action_idx]
#         plt.scatter(samples[:, 0], samples[:, 1], label=f'Action {action_idx}', alpha=0.5)
#     plt.title('Tranisitions from the origin under 10 different actions.')
#     plt.xlabel('x')
#     plt.ylabel('y')
#     plt.legend()
#     plt.grid()
#     plt.show()


import numpy as np
import time
import matplotlib
matplotlib.use('TkAgg') # Or 'Qt5Agg' if you have PyQt installed
import matplotlib.pyplot as plt
from numba import njit, prange

@njit(parallel=True)
def gen_transit_data_core(NUM_ACTIONS, NUM_SAMPLES_PER_ACTION, w1_all, w2_all):
    """
    Core function to generate transition data, compiled with Numba for performance.
    """
    # The actions are 2D with range (u_{1,k},u_{2,k})\in[0,3]\times[0,2\pi]
    # Sample trajectories for each action in the action set from the initial state (0,0).
    # The dynamics are
    #   x_{k+1} = x_k + u_{1,k}[\cos(u_{2,k} + w_{2,k})  \sin(u_{2,k} + w_{2,k})]^T + w_{1,k},
    # where w_{1,k}=N([0 0]^T, diag(0.4, 0.4)), w_{2,k}=N(0, 0.2)

    # Create numpy array to store one-step transitions
    transition_data = np.zeros((NUM_ACTIONS, NUM_SAMPLES_PER_ACTION, 3))  # state_dim=2 for (x,y,cost)
    action_set = np.zeros((NUM_ACTIONS, 2))  # To store the action parameters (u1, u2) for each action

    # OPTIMIZATION: Use prange for the outer loop to enable multi-threading
    for action_idx in prange(NUM_ACTIONS):
        u1 = np.random.uniform(0, 3)
        if action_idx== 0:
            u1 = 0      # Ensure there is one zero input
        u2 = np.random.uniform(0, 2 * np.pi)

        # FIX: Assign scalar values explicitly to avoid type casting errors with lists
        action_set[action_idx, 0] = u1
        action_set[action_idx, 1] = u2

        for sample_idx in range(NUM_SAMPLES_PER_ACTION):
            # OPTIMIZATION: Avoid intermediate np.array creations inside the inner loop
            w2 = w2_all[action_idx, sample_idx]
            cos_val = np.cos(u2 + w2)
            sin_val = np.sin(u2 + w2)

            # Unroll the mathematical update into scalar operations for maximum Numba speed
            transition_data[action_idx, sample_idx, 0] = u1 * cos_val + w1_all[action_idx, sample_idx, 0]
            transition_data[action_idx, sample_idx, 1] = u1 * sin_val + w1_all[action_idx, sample_idx, 1]
            transition_data[action_idx, sample_idx, 2] = u1 # =abs(u1), since u1 is non-negative

    return transition_data, action_set


def generate_transition_data(SIM_PARAMS):
    """
    Generates trajectory data for a 2D unicycle model by sampling trajectories for each action in the action set.
    """

    print("Generating transition data for a 2D unicycle model with the following parameters:")
    print(f"Number of actions: {SIM_PARAMS['NUM_ACTIONS']}")
    print(f"Number of samples per action: {SIM_PARAMS['NUM_SAMPLES_PER_ACTION']}")
    print(f"Noise covariance (w1): {SIM_PARAMS['NOISE_COVARIANCE']}")
    print(f"Noise standard deviation (w2): {SIM_PARAMS['NOISE_STD']}")

    NUM_ACTIONS = SIM_PARAMS['NUM_ACTIONS']
    NUM_SAMPLES_PER_ACTION = SIM_PARAMS['NUM_SAMPLES_PER_ACTION']
    NOISE_COVARIANCE = SIM_PARAMS['NOISE_COVARIANCE']
    NOISE_STD = SIM_PARAMS['NOISE_STD']

    # Measure time taken to generate dataset
    start_time = time.time()

    # Generate noise for all samples in advance for autocompilation with numba.
    w1_all = np.random.multivariate_normal(mean=[0, 0], cov=NOISE_COVARIANCE,
                                           size=(NUM_ACTIONS, NUM_SAMPLES_PER_ACTION))
    w2_all = np.random.normal(loc=0, scale=NOISE_STD, size=(NUM_ACTIONS, NUM_SAMPLES_PER_ACTION))

    transition_data, action_set = gen_transit_data_core(NUM_ACTIONS, NUM_SAMPLES_PER_ACTION, w1_all, w2_all)

    end_time = time.time()
    print(f"Transition data generation completed in {end_time - start_time:.2f} seconds.")

    return transition_data, action_set


def debug_plot_transition_data(transition_data):
    """Plots the generated trajectory data for 10 randomly selected actions."""
    plt.figure(figsize=(6, 6))
    for action_idx in range(min(10, transition_data.shape[0])):  # Plot trajectories for up to 10 actions
        samples = transition_data[action_idx]
        plt.scatter(samples[:, 0], samples[:, 1], label=f'Action {action_idx}', alpha=0.5)
    plt.title('Transitions from the origin under 10 different actions.')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid()
    plt.show()

    # Generate 3D Plot
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    for action_idx in range(min(20, transition_data.shape[0])):  # Plot trajectories for up to 10 actions
        samples = transition_data[action_idx]
        ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2], label=f'Action {action_idx}', alpha=0.5)
    ax.set_title('Transitions from the origin under 10 different actions (3D).')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('cost')
    ax.legend()
    plt.show()
