# Authors, Niklas Schmid, Jared Miller, Tristan Zeller,
# Marta Fochesato, Tobias Sutter, John Lygeros 2026
#
# This source code is licensed under the license
# found in the LICENSE file in the root directory of this source tree.


# This file containts three dynamic programming implementations:
# 1. Maximize the reach-avoid probability of the system
# 2. Minimize the expected cost subject to a constraint on the reach-avoid probability
# 3. Minimize the expected cost subject to a constraint on the reach-avoid probability.
#    But the cost only includes trajectories that satisfy the reach-avoid specification.
#    For this case, the accumulated cost of the trajectory is captured in a third state component.

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from numba import njit, prange

from parameters import DP_PARAMS, sim_params, get_discretization, TRACK_NAME
from tqdm import tqdm
import matplotlib.ticker as ticker

# Implementation notes: collected data is from zero cost and origin, just add to all transitions since cost and dynamics independent of state
# 1. and 2. dont require the third state. To maintain consistency just ignore third state component and  duplicate policy and value function over all cost levels at the end.


def execute_DP(transition_data, initial_state, map,action_set, lambda_val=1.0, alpha = 0.5):
    """
    Wrapper for the DP execution. This function will call the appropriate DP implementation based on the objective specified in DP_PARAMS.

    :param transition_data:
    :param initial_state:
    :return:
    """
    print("DP objective: " + DP_PARAMS['OBJECTIVE'])
    print("Time horizon (N): " + str(DP_PARAMS['N']))
    print("State space size: " + str(DP_PARAMS['state_space_physical_size']))
    print("Initial state: " + str(initial_state))
    print("State space size: " + str(get_discretization(map[0])) + " (x_dim, y_dim, c_dim)")
    epsilon = 0

    if DP_PARAMS['OBJECTIVE'] == 'RA':
        policy, value_function = DP_reach_avoid(transition_data, map)
    elif DP_PARAMS['OBJECTIVE'] == 'JCC_classic':
        print("Objective: Classic JCC")
        policy, value_function, epsilon = DP_JCC_classic(transition_data, initial_state, map,action_set, alpha)
    elif DP_PARAMS['OBJECTIVE'] == 'JCC_classic_sweep':
        policy, value_function = DP_JCC_classic_fixed_lambda(transition_data, map, lambda_val)
    elif DP_PARAMS['OBJECTIVE'] == 'JCC_only_safe':
        print("Objective: Only Safe JCC")
        policy, value_function, epsilon = DP_JCC_only_safe(transition_data, initial_state, map,action_set, alpha)
    elif DP_PARAMS['OBJECTIVE'] == 'JCC_onlySafe_sweep':
        policy, value_function = DP_JCC_only_safe_fixed_lambda_gpu(transition_data, map, lambda_val)

    if DP_PARAMS['OBJECTIVE']=='JCC_classic' or DP_PARAMS['OBJECTIVE']=='JCC_only_safe':
        return policy, value_function, epsilon
    else:
        return policy, value_function

def DP_JCC_only_safe(transition_data, initial_state, map,action_set,alpha):
    """
    Dynamic programming implementation to minimize the expected cost subject to a constraint on the reach-avoid probability.
    But the cost only includes trajectories that satisfy the reach-avoid specification. For this case, the accumulated cost of the trajectory is captured in a third state component.

    :param transition_data:
    :param initial_state:
    :return:
    """
    lambda_max = 22.84
    lambda_min = 0
    lambda_val = lambda_max
    max_iter = 30


    pi_up = None
    pi_down = None
    V_up = None
    V_down = None
    safety_pi_up = 1
    safety_pi_down = 0
    epsilon= 0


    print("Alpha constraint: " + str(alpha))
    for iter in range(max_iter):
        policy, value_function = DP_JCC_only_safe_fixed_lambda_gpu(transition_data, map, lambda_val)

        safety_value, _ = simulate_rollouts_gpu(policy, action_set, map, initial_state, num_rollouts=1000000)

        if safety_value - alpha > 0:
            if safety_pi_up > safety_value:
                pi_up = policy
                V_up = value_function
                safety_pi_up = safety_value
                # pi_up is played with probability epsilon; we want safety alpha in total
        else:
            if safety_pi_down < safety_value:
                pi_down = policy
                V_down = value_function
                safety_pi_down = safety_value

        epsilon = (alpha-safety_pi_down) / (safety_pi_up-safety_pi_down)


        # Instead of bisection multiply lambda by a factor that diminishes over iterations:
        factor = 1 + 0.5 * (max_iter - iter) / max_iter * (alpha-safety_value)
        lambda_val = lambda_val * factor
        if abs(safety_value-alpha) < 0.01:
            break
        print(f"Iteration {iter+1} completed with lambda = {lambda_val}, safety value = {safety_value}")
    # check if pi_up is still none:
    if pi_up is None:
        pi_up = pi_down
        V_up = V_down
    elif pi_down is None:
        pi_down = pi_up
        V_down = V_up
    print(f"Converged after {iter+1} iterations with lambda = {lambda_val}, safety value = {safety_value}")
    return (pi_up,pi_down), (V_up, V_down), epsilon


def DP_JCC_classic(transition_data, initial_state, map, action_set, alpha):
    """
    Dynamic programming implementation to minimize the expected cost subject to a constraint on the reach-avoid probability.
    But the cost only includes trajectories that satisfy the reach-avoid specification. For this case, the accumulated cost of the trajectory is captured in a third state component.

    :param transition_data:
    :param initial_state:
    :return:
    """
    lambda_max = 22
    lambda_min = 0
    lambda_val = lambda_max
    max_iter = 30
    velocity = 0
    beta = 0.9  # Smoothing factor (0.5 to 0.9)

    pi_up = None
    pi_down = None
    V_up = None
    V_down = None
    safety_pi_up = 1
    safety_pi_down = 0
    epsilon= 0

    print("Alpha constraint: " + str(alpha))
    for iter in range(max_iter):
        policy, value_function = DP_JCC_classic_fixed_lambda(transition_data, map, lambda_val)

        safety_value, _ = simulate_rollouts_gpu(policy, action_set, map, initial_state, num_rollouts=1000000)
        print(f"Iteration {iter+1} completed with lambda = {lambda_val}, safety value = {safety_value}")

        # Instead of bisection multiply lambda by a factor that diminishes over iterations:
        progress_ratio = (max_iter - iter) / max_iter
        # Before the loop

        if safety_value - alpha > 0:
            if safety_pi_up > safety_value:
                pi_up = policy
                V_up = value_function
                safety_pi_up = safety_value
                # pi_up is played with probability epsilon; we want safety alpha in total
        else:
            if safety_pi_down < safety_value:
                pi_down = policy
                V_down = value_function
                safety_pi_down = safety_value

        epsilon = (alpha-safety_pi_down) / (safety_pi_up-safety_pi_down)

        # Inside the loop
        error = alpha - safety_value
        update = 0.1 * (progress_ratio ** 2) * error
        velocity = beta * velocity + (1 - beta) * update
        factor = 1 + velocity
        lambda_val = lambda_val * factor


        if abs(safety_value-alpha) < 0.01:
            break

    # check if pi_up is still none:
    if pi_up is None:
        pi_up = pi_down
        V_up = V_down
    elif pi_down is None:
        pi_down = pi_up
        V_down = V_up
    print(f"Converged after {iter+1} iterations with lambda = {lambda_val}, safety value = {safety_value}")
    return (pi_up,pi_down), (V_up, V_down), epsilon


def plot_value_function_and_policy(V, pi, action_set, map):
    safe_set, target_set = map

    # Plot the value function and policy.
    plt.figure(0)
    plt.imshow(V[0,:,:,0], cmap='hot', interpolation='nearest')
    plt.title("Value Function at Initial Time Step")

    # Plot policy as arrows. For better visualization, we can plot the policy as arrows indicating the direction of the action to take at each state.
    plt.figure(1)
    x_dim, y_dim = safe_set.shape
    Y, X = np.meshgrid(np.arange(x_dim), np.arange(y_dim))
    U = np.zeros_like(X, dtype=float)
    V = np.zeros_like(Y, dtype=float)
    for x_idx in range(x_dim):
        for y_idx in range(y_dim):
            action_idx = pi[0, x_idx, y_idx,0]  # Get the action index from the policy
            if action_idx >= 0:  # Check if the action index is valid
                u1, u2 = action_set[action_idx]  # Get the action parameters (u1, u2)
                U[x_idx, y_idx] = u1 * np.cos(u2)  # Compute the x component of the arrow
                V[x_idx, y_idx] = u1 * np.sin(u2)  # Compute the y component of the arrow
    plt.imshow(safe_set, cmap='gray', interpolation='nearest')  # Plot the safe set as background
    plt.imshow(target_set, cmap='Blues', alpha=0.5, interpolation='nearest')  # Plot the target set as overlay
    plt.quiver(Y, X, V, -U)
    plt.title("Policy Visualization as Arrows")
    plt.show()


import numpy as np
from numba import cuda
import math

import numpy as np
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_normal_float32
import math

import numpy as np
import math
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_normal_float32

import numpy as np
import math
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_normal_float32

import numpy as np
import math
import warnings
from numba import cuda, NumbaPerformanceWarning
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_normal_float32

# Suppress the occupancy warning if we know we are running a small workload
warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)


def simulate_rollouts_gpu(pi, action_set, map_tuple, initial_state, num_rollouts=1000):
    # 1. Unpack and determine rollout counts
    if isinstance(pi, np.ndarray):
        policies_to_run = [(pi, num_rollouts)]
    else:
        pi_up, pi_down, epsilon = pi
        n_up = int(num_rollouts * epsilon)
        n_down = num_rollouts - n_up
        policies_to_run = [(pi_up, n_up), (pi_down, n_down)]

    all_trajectories = []
    all_successes = []

    # Common GPU transfers (Constants for both runs)
    d_action_set = cuda.to_device(np.array(action_set, dtype=np.float32))
    d_safe_map = cuda.to_device(map_tuple[0].astype(np.int32))
    d_target_map = cuda.to_device(map_tuple[1].astype(np.int32))
    d_initial_state = cuda.to_device(np.array(initial_state, dtype=np.float32))

    # 2. Loop over the policies
    for current_pi, n_current in policies_to_run:
        if n_current <= 0:
            continue

        N = current_pi.shape[0]
        d_pi = cuda.to_device(current_pi.astype(np.int32))

        # Bundle configuration
        config = np.array([
            DP_PARAMS['state_space_physical_size'][0] / map_tuple[0].shape[0],
            DP_PARAMS['state_space_physical_size'][1] / map_tuple[0].shape[1],
            DP_PARAMS['cost_space_physical_size'] / current_pi.shape[3],
            math.sqrt(sim_params['NOISE_COVARIANCE'][0, 0]),
            sim_params['NOISE_STD']
        ], dtype=np.float32)
        d_config = cuda.to_device(config)

        # Allocate buffers for this specific batch
        d_rollouts = cuda.device_array((n_current, N + 1, 5), dtype=np.float32)
        d_success = cuda.device_array(n_current, dtype=np.bool_)
        rng_states = create_xoroshiro128p_states(n_current, seed=np.random.randint(0, 10000))

        # Execution Configuration
        threads_per_block = 256
        blocks_per_grid = (n_current + (threads_per_block - 1)) // threads_per_block

        # 3. Launch the existing kernel (No changes needed to rollout_kernel_cuda)
        rollout_kernel_cuda[blocks_per_grid, threads_per_block](
            d_pi, d_action_set, d_safe_map, d_target_map,
            d_initial_state, d_rollouts, d_success,
            rng_states, d_config, N
        )

        # 4. Pull data back and store
        all_trajectories.append(d_rollouts.copy_to_host())
        all_successes.append(d_success.copy_to_host())

    # 5. Aggregate Results
    trajectories = np.concatenate(all_trajectories, axis=0)
    success_flags = np.concatenate(all_successes, axis=0)
    empirical_success_rate = np.mean(success_flags)

    return empirical_success_rate, trajectories


# def simulate_rollouts_gpu(pi, action_set, map_tuple, initial_state, num_rollouts=1000):
#     # if pi is not just an array, unpack
#     (pi_up, pi_down, epsilon) = (pi, pi, 1) if isinstance(pi, np.ndarray) else pi
#     rollouts_using_pi_up = int(num_rollouts * epsilon)
#     rollouts_using_pi_down = num_rollouts - rollouts_using_pi_up
#
#
#
#     safe_map, target_map = map_tuple
#     N = pi.shape[0]
#
#     # 1. Cast and move all arrays to Device
#     # Ensure float32 for speed; float64 is often 10-32x slower on consumer GPUs
#     d_pi = cuda.to_device(pi.astype(np.int32))
#     d_action_set = cuda.to_device(np.array(action_set, dtype=np.float32))
#     d_safe_map = cuda.to_device(safe_map.astype(np.int32))
#     d_target_map = cuda.to_device(target_map.astype(np.int32))
#     d_initial_state = cuda.to_device(np.array(initial_state, dtype=np.float32))
#
#     # Bundle configuration to avoid individual host-to-device transfers
#     config = np.array([
#         DP_PARAMS['state_space_physical_size'][0] / safe_map.shape[0],  # cx
#         DP_PARAMS['state_space_physical_size'][1] / safe_map.shape[1],  # cy
#         DP_PARAMS['cost_space_physical_size'] / pi.shape[3],  # cc
#         math.sqrt(sim_params['NOISE_COVARIANCE'][0, 0]),  # n_std_pos
#         sim_params['NOISE_STD']  # n_std_rot
#     ], dtype=np.float32)
#     d_config = cuda.to_device(config)
#
#     # 2. Allocate output buffers directly on the GPU
#     d_rollouts = cuda.device_array((num_rollouts, N + 1, 5), dtype=np.float32)
#     d_success = cuda.device_array(num_rollouts, dtype=np.bool_)
#
#     # 3. Setup Random States (One per rollout)
#     rng_states = create_xoroshiro128p_states(num_rollouts, seed=np.random.randint(0, 10000))
#
#     # 4. Execution Configuration
#     # Using 128 or 256 threads per block is usually the sweet spot for occupancy
#     threads_per_block = 256
#     blocks_per_grid = (num_rollouts + (threads_per_block - 1)) // threads_per_block
#
#     # 5. Launch
#     rollout_kernel_cuda[blocks_per_grid, threads_per_block](
#         d_pi, d_action_set, d_safe_map, d_target_map,
#         d_initial_state, d_rollouts, d_success,
#         rng_states, d_config, N
#     )
#
#     # 6. Synchronization and Retrieval
#     cuda.synchronize()
#
#     empirical_success_rate = np.mean(d_success.copy_to_host())
#     trajectories = d_rollouts.copy_to_host()
#
#     return empirical_success_rate, trajectories


@cuda.jit
def rollout_kernel_cuda(pi, action_set, safe_map, target_map,
                        initial_state, rollouts, success_of_rollout,
                        rng_states, config, N):
    idx = cuda.grid(1)
    if idx >= rollouts.shape[0]:
        return

    # Cache config values in registers
    cx, cy, cc = config[0], config[1], config[2]
    n_std_pos, n_std_rot = config[3], config[4]

    # Local registers for state
    curr_x = initial_state[0]
    curr_y = initial_state[1]
    curr_c = 0.0

    # Store initial step
    rollouts[idx, 0, 0] = curr_x
    rollouts[idx, 0, 1] = curr_y
    rollouts[idx, 0, 2] = curr_c

    success_of_rollout[idx] = False
    is_terminated = False

    for k in range(N):
        if is_terminated:
            break

        # Discretized Lookup
        xi = max(0, min(int(math.floor(curr_x / cx)), pi.shape[1] - 1))
        yi = max(0, min(int(math.floor(curr_y / cy)), pi.shape[2] - 1))
        ci = max(0, min(int(math.floor(curr_c / cc)), pi.shape[3] - 1))

        # Get Control
        act_idx = pi[k, xi, yi, ci]
        u1 = action_set[act_idx, 0]
        u2 = action_set[act_idx, 1]

        # Physics + Noise
        # xoroshiro128p_normal_float32 generates a standard normal N(0,1)
        w1_x = xoroshiro128p_normal_float32(rng_states, idx) * n_std_pos
        w1_y = xoroshiro128p_normal_float32(rng_states, idx) * n_std_pos
        w2_n = xoroshiro128p_normal_float32(rng_states, idx) * n_std_rot

        next_x = curr_x + u1 * math.cos(u2 + w2_n) + w1_x
        next_y = curr_y + u1 * math.sin(u2 + w2_n) + w1_y

        # Discretized cost state logic
        raw_c = curr_c + u1
        next_c = (int(math.floor(raw_c / cc))) * cc + (cc / 2.0)

        # Collision Check
        # Uses your provided Amanatides & Woo voxel traversal
        status, _, _ = collision_check_line_gpu(
            safe_map, target_map, curr_x / cx, curr_y / cy, next_x / cx, next_y / cy
        )

        curr_x, curr_y, curr_c = next_x, next_y, next_c

        if status == 1:  # Target Reached
            success_of_rollout[idx] = True
            is_terminated = True
        elif status == 0:  # Unsafe / Out of Bounds
            is_terminated = True

        # Trajectory recording
        if is_terminated:
            # Fill remaining steps with the terminal state
            for fill_idx in range(k + 1, N + 1):
                rollouts[idx, fill_idx, 0] = curr_x
                rollouts[idx, fill_idx, 1] = curr_y
                rollouts[idx, fill_idx, 2] = curr_c
                rollouts[idx, k, 3] = 0
                rollouts[idx, k, 4] = 0
        else:
            rollouts[idx, k + 1, 0] = curr_x
            rollouts[idx, k + 1, 1] = curr_y
            rollouts[idx, k + 1, 2] = curr_c
            rollouts[idx, k, 3] = action_set[act_idx, 0]
            rollouts[idx, k, 4] = action_set[act_idx, 1]

def simulate_rollouts_of_policy(V, pi, action_set, map, initial_state, num_rollouts=100, verbose = False):
    """
    Simulate rollouts of the system under the computed policy to empirically evaluate the reach-avoid probability and expected cost.
    :param V:
    :param pi:
    :param action_set:
    :param map:
    :param num_rollouts:
    :return:
    """
    print("Number of rollouts: " + str(num_rollouts))
    safe_map, target_map = map
    x_dim, y_dim, c_dim = get_discretization(safe_map)
    N = DP_PARAMS['N']
    initial_state_with_cost = np.array([initial_state[0], initial_state[1], 0])  # Assuming zero initial cost
    rollouts = np.zeros((num_rollouts, N+1, 3))  # To store the state trajectories of the rollouts
    success_of_rollout = np.zeros(num_rollouts, dtype=bool)  # To store whether each rollout successfully reached the target while avoiding unsafe states

    cell_size_x = DP_PARAMS['state_space_physical_size'][0] / x_dim
    cell_size_y = DP_PARAMS['state_space_physical_size'][1] / y_dim
    cell_size_c = DP_PARAMS['cost_space_physical_size'] / c_dim

    for rollout_idx in range(num_rollouts):
        rollouts[rollout_idx, 0, :] = np.array(initial_state_with_cost)
        for k in range(N):
            x = rollouts[rollout_idx, k, 0]
            y = rollouts[rollout_idx, k, 1]
            c = rollouts[rollout_idx, k, 2]

            # Compute x_idx, y_idx, c_idx for the current state
            x_idx = int(np.floor(x / cell_size_x))
            y_idx = int(np.floor(y / cell_size_y))
            c_idx = int(np.floor(c / cell_size_c))

            action_idx = pi[k, x_idx, y_idx, c_idx]
            u1, u2 = action_set[action_idx]

            # Simulate one step of the system dynamics under the action and noise
            w1 = np.random.multivariate_normal(mean=[0, 0], cov=sim_params['NOISE_COVARIANCE'])
            w2 = np.random.normal(loc=0, scale=sim_params['NOISE_STD'])
            next_x = x + u1 * np.cos(u2 + w2) + w1[0]
            next_y = y + u1 * np.sin(u2 + w2) + w1[1]
            next_c = c + u1
            # For technical reasons, the cost state must propagate by the discretized dynamics.
            # Compute next c_idx:
            next_c_idx = int(next_c / cell_size_c)
            # Convert next_c_idx back to next_c for storage in rollouts:
            next_c = next_c_idx * cell_size_c + cell_size_c / 2
            #rollouts[rollout_idx, k+1, :] = np.array([next_x, next_y, next_c])
            # Check if the rollout has hit the target or unsafe set or exceeded state space limit; terminate trajectory at this point if so.
            # Using the function collision_check_line to check for collisions along the trajectory from the current state to the next state.
            x_grid_size = x / cell_size_x
            y_grid_size = y / cell_size_y

            next_x_grid_sized = next_x / cell_size_x
            next_y_grid_sized = next_y / cell_size_y

            status, (curr_x, curr_y) = collision_check_line(safe_map, target_map, x_grid_size, y_grid_size, next_x_grid_sized, next_y_grid_sized)
            if (next_x < 0 or next_x >= DP_PARAMS['state_space_physical_size'][0] or next_y < 0 or next_y >= DP_PARAMS['state_space_physical_size'][1]) and not status == 'target' and not status == 'unsafe':
                blla = 213
                status, (curr_x, curr_y) = collision_check_line(safe_map, target_map, rollouts[rollout_idx, k, 0], rollouts[rollout_idx, k, 1], next_x, next_y)

            if status == "target" or status == "unsafe":
                for remaining_k in range(k+1, N+1):
                    rollouts[rollout_idx, remaining_k, :] = np.array([next_x, next_y, next_c])
                if status == "target":
                    success_of_rollout[rollout_idx] = True
                break
            else:
                rollouts[rollout_idx, k+1, :] = np.array([next_x, next_y, next_c])

    empirical_success_rate = np.mean(success_of_rollout)
    if verbose:
        # Print safety according to value function:
        initial_x_idx = int(np.floor(initial_state[0] / cell_size_x))
        initial_y_idx = int(np.floor(initial_state[1] / cell_size_y))
        initial_c_idx = int(np.floor(0 / cell_size_c))  # Assuming zero initial cost
        print("Value function at initial state: " + str(V[0, initial_x_idx, initial_y_idx, initial_c_idx]))
        # Print empirical success rate:
        print("Empirical success rate of rollouts: " + str(empirical_success_rate))
        # Plot rollouts
        plt.figure(figsize=(10, 10))
        # Scale imshow to correct physical size of state space:
        true_x = np.linspace(0, DP_PARAMS['state_space_physical_size'][0], safe_map.shape[0])
        true_y = np.linspace(0, DP_PARAMS['state_space_physical_size'][1], safe_map.shape[1])
        plt.imshow(safe_map.T, extent=(0, DP_PARAMS['state_space_physical_size'][0], 0, DP_PARAMS['state_space_physical_size'][1]), origin='lower', cmap='gray', interpolation='nearest')  # Plot the safe set as background
        plt.imshow(target_map.T, extent=(0, DP_PARAMS['state_space_physical_size'][0], 0, DP_PARAMS['state_space_physical_size'][1]), origin='lower', cmap='Blues', alpha=0.5, interpolation='nearest')  # Plot the target set as overlay
        for rollout_idx in range(num_rollouts):
            if success_of_rollout[rollout_idx]:
                plt.plot(rollouts[rollout_idx, :, 0], rollouts[rollout_idx, :, 1], color='green', alpha=0.5)
            else:
                plt.plot(rollouts[rollout_idx, :, 0], rollouts[rollout_idx, :, 1], color='red', alpha=0.5)
        plt.title(f"Rollouts of the System under the Computed Policy\n"
                  f"(Value Function, Empirical Safety) = ({V[0, initial_x_idx, initial_y_idx, initial_c_idx]:.2f}, {empirical_success_rate:.2f}) "
                  f"over {num_rollouts} rollouts")
        plt.xlabel('x')
        plt.ylabel('y')
        # Limit to physical state space size
        plt.xlim(0, DP_PARAMS['state_space_physical_size'][0])
        plt.ylim(0, DP_PARAMS['state_space_physical_size'][1])
        plt.grid()
        plt.show()

    return empirical_success_rate
@njit
def collision_check_line(safe_map, target_map, x1, y1, x2, y2):
    """
    Checks collisions with target or safe set using method described in "A Fast Voxel Traversal Algorithm for Ray Tracing" by John Amanatides and Andrew Woo
    :param safe_map:
    :param target_map:
    :param x1:
    :param y1:
    :param x2:
    :param y2:
    :return:
    """
    rows, cols = safe_map.shape

    # Starting grid coordinates
    curr_x, curr_y = int(np.floor(x1)), int(np.floor(y1))
    end_x, end_y = int(np.floor(x2)), int(np.floor(y2))

    dx = x2 - x1
    dy = y2 - y1

    step_x = 1 if dx > 0 else -1 if dx < 0 else 0
    step_y = 1 if dy > 0 else -1 if dy < 0 else 0

    # Distance to move to cross the first vertical/horizontal grid line
    if step_x != 0:
        t_max_x = (float(curr_x + (1 if step_x > 0 else 0)) - x1) / dx
        t_delta_x = abs(1.0 / dx)
    else:
        t_max_x = 1e10  # Infinity
        t_delta_x = 1e10

    if step_y != 0:
        t_max_y = (float(curr_y + (1 if step_y > 0 else 0)) - y1) / dy
        t_delta_y = abs(1.0 / dy)
    else:
        t_max_y = 1e10
        t_delta_y = 1e10

    while True:
        # 1. Bounds Check
        if not (0 <= curr_x < cols and 0 <= curr_y < rows):
            return "unsafe", (curr_x, curr_y)

        # 2. Collision Checks
        if target_map[curr_x, curr_y] == 1:
            return "target", (curr_x, curr_y)

        if safe_map[curr_x, curr_y] == 0:
            return "unsafe", (curr_x, curr_y)

        # 3. Check if reached destination
        if curr_x == end_x and curr_y == end_y:
            break

        # 4. Step to next cell
        if t_max_x < t_max_y:
            t_max_x += t_delta_x
            curr_x += step_x
        else:
            t_max_y += t_delta_y
            curr_y += step_y

    return "safe_no_target", (end_x, end_y)

@njit(parallel=True)
def collision_check_line_mine(safe_map, target_map, x1, y1, x2, y2):
    """
    Checks if a line segment hits a target or unsafe cell in a grid.
    Returns: (status, (grid_x, grid_y))
    """
    rows, cols = safe_map.shape

    # Place initial positions in center of cell
    x1 = x1 + 0.5
    x2 = x2 + 0.5
    y1 = y1 + 0.5
    y2 = y2 + 0.5


    # Direction and step
    dx, dy = x2 - x1, y2 - y1
    step_x = 1 if dx >= 0 else -1
    step_y = 1 if dy >= 0 else -1

    # 2. Calculate t_delta: how far along the ray to move one full cell width/height
    #t_delta_x = abs(1 / dx) if dx != 0 else float('inf')
    #t_delta_y = abs(1 / dy) if dy != 0 else float('inf')

    end_x, end_y = int(x2), int(y2)

    # 4. Traversal Loop
    while True:
        # Initialize start/end grid coordinates
        curr_x, curr_y = int(x1), int(y1)

        # Check if out of bounds (Unsafe)
        if not (0 <= curr_x < cols and 0 <= curr_y < rows):
            return "unsafe", (curr_x, curr_y)

        # Check Target first (if hitting both, target usually takes priority)
        if target_map[curr_y, curr_x] == 1:
            return "target", (curr_x, curr_y)

        # Check Unsafe (safe_map == 0 means unsafe)
        if safe_map[curr_y, curr_x] == 0:
            return "unsafe", (curr_x, curr_y)

        # Check if we reached the destination cell without hitting anything
        if curr_x == end_x and curr_y == end_y:
            break

        # 3. Calculate distance along line to the next grid boundary
        if dx > 0:
            t_max_x = ((curr_x + 1) - x1) / dx
        elif dx < 0:
            t_max_x = (curr_x - x1) / dx
        else:
            t_max_x = float('inf')

        if dy > 0:
            t_max_y = ((curr_y + 1) - y1) / dy
        elif dy < 0:
            t_max_y = (curr_y - y1) / dy
        else:
            t_max_y = float('inf')

        t_max = min(t_max_x, t_max_y)

        # Move to the next cell boundary
        x1 += t_max * dx
        y1 += t_max * dy

    return "safe_no_target", (end_x, end_y)


import numpy as np
from numba import njit, prange
from tqdm import tqdm


@njit(parallel=True)
def DP_reach_avoid_step_core(V_next, safe_map, target_map, transition_data,
                             x_dim, y_dim, u_dim, samples_per_action,
                             cell_size_x, cell_size_y):
    """
    Core computation for a single time step k.
    """
    pi_k = np.zeros((x_dim, y_dim), dtype=np.int32)
    V_k = np.zeros((x_dim, y_dim))

    for x_idx in prange(x_dim):
        for y_idx in range(y_dim):
            # 1. Handle terminal/static cases first
            if target_map[x_idx, y_idx] == 1:
                V_k[x_idx, y_idx] = 1.0
                continue

            if safe_map[x_idx, y_idx] == 0:
                V_k[x_idx, y_idx] = 0.0
                continue

            # 2. Optimization for safe states outside target
            expected_values = np.zeros(u_dim)

            for action_idx in range(u_dim):
                sum_val = 0.0
                for realization in range(samples_per_action):
                    # Compute one-step transition
                    transition = transition_data[action_idx, realization, 0:2]

                    # Current and Next positions in grid-scale
                    x_grid_sized = x_idx + 0.5
                    y_grid_sized = y_idx + 0.5
                    x_next_grid_sized = x_grid_sized + (transition[0] / cell_size_x)
                    y_next_grid_sized = y_grid_sized + (transition[1] / cell_size_y)

                    # Logic check via collision function
                    # Note: collision_check_line must also be @njit compatible!
                    status, (curr_x, curr_y) = collision_check_line(
                        safe_map, target_map,
                        x_grid_sized, y_grid_sized,
                        x_next_grid_sized, y_next_grid_sized
                    )

                    if status == "safe_no_target" or status == "target":
                        sum_val += V_next[int(curr_x), int(curr_y)]
                    # "unsafe" adds 0, so we omit it

                expected_values[action_idx] = sum_val / samples_per_action

            best_action_idx = np.argmax(expected_values)
            pi_k[x_idx, y_idx] = best_action_idx
            V_k[x_idx, y_idx] = expected_values[best_action_idx]

    return pi_k, V_k


def DP_reach_avoid(transition_data, maps):
    safe_map, target_map = maps
    x_dim, y_dim, c_dim = get_discretization(safe_map)
    u_dim = transition_data.shape[0]
    samples_per_action = transition_data.shape[1]

    N = DP_PARAMS['N']
    cell_size_x = DP_PARAMS['state_space_physical_size'][0] / x_dim
    cell_size_y = DP_PARAMS['state_space_physical_size'][1] / y_dim

    # Initialize V with the target map at k=N
    V = np.zeros((N + 1, x_dim, y_dim))
    V[N] = target_map.astype(float)
    pi = np.zeros((N, x_dim, y_dim), dtype=np.int32)

    # Time-step loop remains here to maintain tqdm
    for k in tqdm(range(N - 1, -1, -1)):
        pi_k, V_k = DP_step_core(
            V[k + 1], safe_map, target_map, transition_data,
            x_dim, y_dim, u_dim, samples_per_action,
            cell_size_x, cell_size_y, lambda_val = 1.0, jcc=False
        )
        pi[k] = pi_k
        V[k] = V_k

    # Broadcast to c_dim for compatibility
    pi_expanded = np.repeat(pi[:, :, :, np.newaxis], c_dim, axis=3)
    V_expanded = np.repeat(V[:, :, :, np.newaxis], c_dim, axis=3)

    return pi_expanded, V_expanded

@njit(parallel=True)
def DP_step_core(V_next, safe_map, target_map, transition_data,
                             x_dim, y_dim, u_dim, samples_per_action,
                             cell_size_x, cell_size_y, lambda_val, jcc=False):
    """
    Core computation for a single time step k.
    """
    pi_k = np.zeros((x_dim, y_dim), dtype=np.int32)
    V_k = np.zeros((x_dim, y_dim))

    for x_idx in prange(x_dim):
        for y_idx in range(y_dim):
            # 1. Handle terminal/static cases first; No stage cost from these states
            if target_map[x_idx, y_idx] == 1:
                V_k[x_idx, y_idx] = lambda_val
                continue

            if safe_map[x_idx, y_idx] == 0:
                V_k[x_idx, y_idx] = 0.0
                continue

            # 2. Optimization for safe states outside target
            expected_values = np.zeros(u_dim)

            for action_idx in range(u_dim):
                sum_val = 0.0
                for realization in range(samples_per_action):
                    # Compute one-step transition
                    transition = transition_data[action_idx, realization, 0:2]

                    # Current and Next positions in grid-scale
                    x_grid_sized = x_idx + 0.5
                    y_grid_sized = y_idx + 0.5
                    x_next_grid_sized = x_grid_sized + (transition[0] / cell_size_x)
                    y_next_grid_sized = y_grid_sized + (transition[1] / cell_size_y)

                    # Logic check via collision function
                    # Note: collision_check_line must also be @njit compatible!
                    status, (curr_x, curr_y) = collision_check_line(
                        safe_map, target_map,
                        x_grid_sized, y_grid_sized,
                        x_next_grid_sized, y_next_grid_sized
                    )

                    if status == "target" or status == "safe_no_target":
                        sum_val += V_next[int(curr_x), int(curr_y)]
                    if jcc==True:
                        sum_val -= transition_data[action_idx, realization, 2]  # Subtract cost term; transition_data[:,:,2] contains the cost of the transition
                    # "unsafe" adds 0, so we omit it

                expected_values[action_idx] = sum_val / samples_per_action

            best_action_idx = np.argmax(expected_values)
            pi_k[x_idx, y_idx] = best_action_idx
            V_k[x_idx, y_idx] = expected_values[best_action_idx]

    return pi_k, V_k


def DP_JCC_classic_fixed_lambda(transition_data, maps, lambda_val):
    """
    Dynamic programming implementation to minimize the expected cost subject to a constraint on the reach-avoid probability.

    :param transition_data:
    :param initial_state:
    :return:
    """

    safe_map, target_map = maps
    x_dim, y_dim, c_dim = get_discretization(safe_map)
    u_dim = transition_data.shape[0]
    samples_per_action = transition_data.shape[1]

    N = DP_PARAMS['N']
    cell_size_x = DP_PARAMS['state_space_physical_size'][0] / x_dim
    cell_size_y = DP_PARAMS['state_space_physical_size'][1] / y_dim

    # Initialize V with the target map at k=N
    V = np.zeros((N + 1, x_dim, y_dim))
    V[N] = target_map.astype(float)*lambda_val  # Scale the terminal value by lambda since we are in the Lagrangian formulation
    pi = np.zeros((N, x_dim, y_dim), dtype=np.int32)

    # Time-step loop remains here to maintain tqdm
    for k in tqdm(range(N - 1, -1, -1)):
        pi_k, V_k = DP_step_core(
            V[k + 1], safe_map, target_map, transition_data,
            x_dim, y_dim, u_dim, samples_per_action,
            cell_size_x, cell_size_y, lambda_val, jcc=True
        )
        pi[k] = pi_k
        V[k] = V_k

    # Broadcast to c_dim for compatibility
    pi_expanded = np.repeat(pi[:, :, :, np.newaxis], c_dim, axis=3)
    V_expanded = np.repeat(V[:, :, :, np.newaxis], c_dim, axis=3)

    return pi_expanded, V_expanded


@njit(parallel=True)
def DP_step_core_3D(V_next, safe_map, target_map, transition_data,
                             x_dim, y_dim, c_dim, u_dim, samples_per_action,
                             cell_size_x, cell_size_y, cell_size_c):
    """
    Core computation for a single time step k.
    """
    pi_k = np.zeros((x_dim, y_dim, c_dim), dtype=np.int32)
    V_k = np.zeros((x_dim, y_dim, c_dim))

    for x_idx in prange(x_dim):
        for y_idx in range(y_dim):
            for c_idx in range(c_dim):
                # 1. Handle terminal/static cases first; No stage cost from these states
                if target_map[x_idx, y_idx] == 1:
                    V_k[x_idx, y_idx, c_idx] = V_next[x_idx, y_idx, c_idx]  # No moving once in target set
                    continue

                if safe_map[x_idx, y_idx] == 0:
                    V_k[x_idx, y_idx, c_idx] = V_next[x_idx, y_idx, c_idx]  # No moving once in unsafe set
                    continue

                # 2. Optimization for safe states outside target
                expected_values = np.zeros(u_dim)

                for action_idx in range(u_dim):
                    sum_val = 0.0
                    for realization in range(samples_per_action):
                        # Compute one-step transition
                        transition = transition_data[action_idx, realization, 0:2]

                        # Current and Next positions in grid-scale
                        x_grid_sized = x_idx + 0.5
                        y_grid_sized = y_idx + 0.5
                        c_grid_sized = c_idx + 0.5
                        x_next_grid_sized = x_grid_sized + (transition[0] / cell_size_x)
                        y_next_grid_sized = y_grid_sized + (transition[1] / cell_size_y)
                        c_next_grid_sized = c_grid_sized + (transition[2] / cell_size_c)

                        # Logic check via collision function
                        # Note: collision_check_line must also be @njit compatible!
                        status, (curr_x, curr_y) = collision_check_line(
                            safe_map, target_map,
                            x_grid_sized, y_grid_sized,
                            x_next_grid_sized, y_next_grid_sized
                        )
                        if c_next_grid_sized >= c_dim:
                            c_next_grid_sized = c_dim - 1.0
                        curr_c = c_next_grid_sized

                        sum_val += V_next[int(curr_x), int(curr_y), int(curr_c)] - transition[2]
                        # "unsafe" adds 0, so we omit it

                    expected_values[action_idx] = sum_val / samples_per_action

                best_action_idx = np.argmax(expected_values)
                pi_k[x_idx, y_idx, c_idx] = best_action_idx
                V_k[x_idx, y_idx, c_idx] = expected_values[best_action_idx]

    return pi_k, V_k

def DP_JCC_only_safe_fixed_lambda(transition_data, maps, lambda_val):
    """
    Dynamic programming implementation to minimize the expected cost subject to a constraint on the reach-avoid probability.
    But the cost only includes trajectories that satisfy the reach-avoid specification. For this case, the accumulated cost of the trajectory is captured in a third state component.

    :param transition_data:
    :param initial_state:
    :return:
    """

    safe_map, target_map = maps
    x_dim, y_dim, c_dim = get_discretization(safe_map)
    u_dim = transition_data.shape[0]
    samples_per_action = transition_data.shape[1]

    N = DP_PARAMS['N']
    cell_size_x = DP_PARAMS['state_space_physical_size'][0] / x_dim
    cell_size_y = DP_PARAMS['state_space_physical_size'][1] / y_dim
    cell_size_c = DP_PARAMS['cost_space_discretization']

    # Initialize V with the target map at k=N
    V = np.zeros((N + 2, x_dim, y_dim, c_dim))
    pi = np.zeros((N+1, x_dim, y_dim, c_dim), dtype=np.int32)

    # There is an additional time-step N+1, where terminal cost is always zero; but the action at time-step N achieves
    # incurs the lambda-cost_state under action "1" if in the target, and zero cost otherwise.
    # Consequently, the optimal policy is to pay always when the cost_state is smaller than lambda and the state is in
    # the target, and never pay otherwise.
    cost_state_up_to_which_its_worth_paying = int(lambda_val / cell_size_c)
    for c_idx in range(cost_state_up_to_which_its_worth_paying + 1):
        V[N, :,:, c_idx] = target_map.astype(float)*lambda_val + (c_idx + 0.5) * cell_size_c

    # Time-step loop remains here to maintain tqdm
    for k in tqdm(range(N - 1, -1, -1)):
        pi_k, V_k = DP_step_core_3D(
            V[k + 1,:,:,:], safe_map, target_map, transition_data,
            x_dim, y_dim, c_dim, u_dim, samples_per_action,
            cell_size_x, cell_size_y, cell_size_c
        )
        pi[k] = pi_k
        V[k] = V_k

    return pi, V



def evaluate_policy_safety_via_value_iteration(transition_data, initial_state, maps, pi, lambda_val):
    """
    Evaluate the reach-avoid probability of a given policy via value iteration.
    :param transition_data:
    :param initial_state:
    :param maps:
    :param pi:
    :return:
    """
    safe_map, target_map = maps
    x_dim, y_dim, c_dim = get_discretization(safe_map)
    u_dim = transition_data.shape[0]
    samples_per_action = transition_data.shape[1]

    N = DP_PARAMS['N']
    cell_size_x = DP_PARAMS['state_space_physical_size'][0] / x_dim
    cell_size_y = DP_PARAMS['state_space_physical_size'][1] / y_dim
    cell_size_c = DP_PARAMS['cost_space_discretization']

    # Initialize V with the target map at k=N
    V = np.zeros((N + 1, x_dim, y_dim, c_dim))

    # Set value function to target map (2d) for all cost layers (first dimension of V) and time-steps (fourth dimension of V)
    V[:, :,:, :] = np.repeat(target_map[:, :, np.newaxis], c_dim, axis=2)[np.newaxis, :, :, :]

    # Time-step loop remains here to maintain tqdm
    for k in tqdm(range(N - 1, -1, -1)):
        V_k = evaluate_safety_via_value_iteration_core(
            V[k + 1,:,:,:], safe_map, target_map, transition_data,
            x_dim, y_dim, c_dim, u_dim, samples_per_action,
            cell_size_x, cell_size_y, cell_size_c, pi, k
        )
        V[k] = V_k
    initial_x_idx = int(np.floor(initial_state[0] / cell_size_x))
    initial_y_idx = int(np.floor(initial_state[1] / cell_size_y))
    initial_c_idx = int(np.floor(0 / cell_size_c))  # Assuming zero initial cost
    print("Value function at initial state: " + str(V[0, initial_x_idx, initial_y_idx, initial_c_idx]))
    return V[0, initial_x_idx, initial_y_idx, initial_c_idx]

@njit(parallel=True)
def evaluate_safety_via_value_iteration_core(V_next, safe_map, target_map, transition_data, x_dim, y_dim, c_dim, u_dim, samples_per_action, cell_size_x, cell_size_y, cell_size_c, pi, k):
    V_k = np.zeros((x_dim, y_dim, c_dim))
    for x_idx in prange(x_dim):
        for y_idx in range(y_dim):
            for c_idx in range(c_dim):
                # 1. Handle terminal/static cases first; No stage cost from these states
                if target_map[x_idx, y_idx] == 1:
                    V_k[x_idx, y_idx, c_idx] = 1  # No moving once in target set
                    continue

                if safe_map[x_idx, y_idx] == 0:
                    V_k[x_idx, y_idx, c_idx] = 0  # No moving once in unsafe set
                    continue

                action_idx = pi[k, x_idx, y_idx, c_idx]

                sum_val = 0.0
                for realization in range(samples_per_action):
                    # Compute one-step transition
                    transition = transition_data[action_idx, realization, 0:3]

                    # Current and Next positions in grid-scale
                    x_grid_sized = x_idx + 0.5
                    y_grid_sized = y_idx + 0.5
                    c_grid_sized = c_idx + 0.5
                    x_next_grid_sized = x_grid_sized + (transition[0] / cell_size_x)
                    y_next_grid_sized = y_grid_sized + (transition[1] / cell_size_y)
                    c_next_grid_sized = c_grid_sized + (transition[2] / cell_size_c)

                    # Logic check via collision function
                    # Note: collision_check_line must also be @njit compatible!
                    status, (curr_x, curr_y) = collision_check_line(
                        safe_map, target_map,
                        x_grid_sized, y_grid_sized,
                        x_next_grid_sized, y_next_grid_sized
                    )
                    if c_next_grid_sized >= c_dim:
                        c_next_grid_sized = c_dim - 1.0
                    curr_c = c_next_grid_sized

                    if status == "target" or status == "safe_no_target":
                        sum_val += V_next[int(curr_x), int(curr_y), int(curr_c)]
                    # "unsafe" adds 0, so we omit it

                V_k[x_idx, y_idx, c_idx] = sum_val / samples_per_action
    return V_k


from numba import cuda
import numpy as np

import math


@cuda.jit
def evaluate_safety_gpu_kernel(V_next, safe_map, target_map, transition_data,
                               x_dim, y_dim, c_dim, u_dim, samples_per_action,
                               cell_size_x, cell_size_y, cell_size_c, pi, k, V_k, mode=0):
    # Determine the absolute thread position in the 3D grid
    x_idx, y_idx, c_idx = cuda.grid(3)

    # Boundary check (essential as grids are often slightly larger than data)
    if x_idx < x_dim and y_idx < y_dim and c_idx < c_dim:

        # 1. Handle terminal/static cases
        if target_map[x_idx, y_idx] == 1 or safe_map[x_idx, y_idx] == 0:
            V_k[x_idx, y_idx, c_idx] = V_next[x_idx, y_idx, c_idx]
            return  # Thread finishes early

        action_idx = pi[k, x_idx, y_idx, c_idx]
        sum_val = 0.0

        # The inner-most loop (samples) remains a loop within each thread
        for realization in range(samples_per_action):
            transition = transition_data[action_idx, realization, 0:3]

            x_grid_sized = x_idx + 0.5
            y_grid_sized = y_idx + 0.5
            c_grid_sized = c_idx + 0.5

            x_next_grid_sized = x_grid_sized + (transition[0] / cell_size_x)
            y_next_grid_sized = y_grid_sized + (transition[1] / cell_size_y)
            c_next_grid_sized = c_grid_sized + (transition[2] / cell_size_c)

            # NOTE: collision_check_line must also be decorated with @cuda.jit(device=True)
            status, curr_x, curr_y = collision_check_line_gpu(
                safe_map, target_map,
                x_grid_sized, y_grid_sized,
                x_next_grid_sized, y_next_grid_sized
            )

            # Simple clamping for C dimension
            curr_c = min(c_next_grid_sized, c_dim - 1.0)

            if status == 1 or status == 2:  # Use integers for status on GPU (Strings are heavy/unsupported)
                V_k[x_idx, y_idx, c_idx] += V_next[int(curr_x), int(curr_y), int(curr_c)]
            # Todo: For evaluation of all costs and unsafe costs, associate value to status==0 here:
            # For safety evaluation and cost of safe trajectories, we could just set the value to zero.
            # For cost of unsafe trajectories, and cost of all trajectories, we set the cost to curr_c
            if status == 0:
                if mode==0 or mode == 1: # Evaluate safety or cost of safe trajectories:
                    V_k[x_idx, y_idx, c_idx] += 0.0
                elif mode==2 or mode == 3: # Evaluate cost of unsafe or all trajectories
                    V_k[x_idx, y_idx, c_idx] += (int(curr_c) + 0.5) * cell_size_c

        V_k[x_idx, y_idx, c_idx] /= samples_per_action

@cuda.jit(device=True)
def collision_check_line_gpu(safe_map, target_map, x1, y1, x2, y2):
    """
    Device-side implementation of Amanatides & Woo voxel traversal.
    Returns: (status_code, curr_x, curr_y)
    Status codes: 0 = unsafe, 1 = target, 2 = safe_no_target
    """
    rows, cols = safe_map.shape

    # Starting grid coordinates
    curr_x = int(math.floor(x1))
    curr_y = int(math.floor(y1))
    end_x = int(math.floor(x2))
    end_y = int(math.floor(y2))

    dx = x2 - x1
    dy = y2 - y1

    step_x = 1 if dx > 0 else (-1 if dx < 0 else 0)
    step_y = 1 if dy > 0 else (-1 if dy < 0 else 0)

    # Distance to move to cross the first vertical/horizontal grid line
    if step_x != 0:
        # Use math.fabs and float literals for GPU compatibility
        t_max_x = (float(curr_x + (1 if step_x > 0 else 0)) - x1) / dx
        t_delta_x = math.fabs(1.0 / dx)
    else:
        t_max_x = 1e10
        t_delta_x = 1e10

    if step_y != 0:
        t_max_y = (float(curr_y + (1 if step_y > 0 else 0)) - y1) / dy
        t_delta_y = math.fabs(1.0 / dy)
    else:
        t_max_y = 1e10
        t_delta_y = 1e10

    # Voxel traversal loop
    # Note: On GPU, we must be careful with infinite loops;
    # but the logic here is bounded by the grid dimensions.
    while True:
        # 1. Bounds Check
        if not (0 <= curr_x < cols and 0 <= curr_y < rows):
            return 0, curr_x, curr_y # "unsafe"

        # 2. Collision Checks
        if target_map[curr_x, curr_y] == 1:
            return 1, curr_x, curr_y # "target"

        if safe_map[curr_x, curr_y] == 0:
            return 0, curr_x, curr_y # "unsafe"

        # 3. Check if reached destination
        if curr_x == end_x and curr_y == end_y:
            break

        # 4. Step to next cell
        if t_max_x < t_max_y:
            t_max_x += t_delta_x
            curr_x += step_x
        else:
            t_max_y += t_delta_y
            curr_y += step_y

    return 2, end_x, end_y # "safe_no_target"


from numba import cuda
import numpy as np
from tqdm import tqdm


def evaluate_policy_safety_via_value_iteration_gpu(transition_data, initial_state, maps, pi, lambda_val, mode = 'safety'):
    safe_map, target_map = maps
    x_dim, y_dim, c_dim = get_discretization(safe_map)
    u_dim, samples_per_action, _ = transition_data.shape
    mode_idx = 0 if mode == 'safety' else 1 if mode == 'cost_safe' else 2 if mode == 'cost_unsafe' else 3

    N = DP_PARAMS['N']
    cell_size_x = DP_PARAMS['state_space_physical_size'][0] / x_dim
    cell_size_y = DP_PARAMS['state_space_physical_size'][1] / y_dim
    cell_size_c = DP_PARAMS['cost_space_discretization']

    # 1. Allocate V on Host (CPU) for final storage
    V = np.zeros((N + 1, x_dim, y_dim, c_dim))
    if mode == 'safety': # Define safety as the probability of being in the target set below cost threshold lambda

        cost_state_up_to_which_its_worth_paying = int(lambda_val / cell_size_c)
        for c_idx in range(cost_state_up_to_which_its_worth_paying + 1):
            # Put cost only in target states:
            V[N, :, :, c_idx] = target_map.astype(float)

    elif mode == 'cost_safe':
        # Put cost only at terminal time based on the policy pi at time-step N
        cost_state_up_to_which_its_worth_paying = int(lambda_val / cell_size_c)
        for c_idx in range(cost_state_up_to_which_its_worth_paying + 1):
            # Put cost only in target states:
            V[N, :, :, c_idx] = target_map.astype(float)*(c_idx + 0.5) * cell_size_c
    elif mode == 'cost_unsafe':
        # Put cost only at terminal time based on policy pi at time-step N, but all in contrast to cost_safe only those which are above the value worth paying
        cost_state_up_to_which_its_worth_paying = int(lambda_val / cell_size_c)
        # for all above cost worth paying assign the cost, and for all below assign zero cost unless not in target
        for c_idx in range(c_dim):
            # Outside target, always put cost
            V[N, :, :, c_idx] = (1 - target_map.astype(float))*(c_idx + 0.5) * cell_size_c
            # Above cost_state worth paying, put cost in target states as well
            if c_idx > cost_state_up_to_which_its_worth_paying:
                # Put cost only outside target states:
                V[N, :, :, c_idx] = (c_idx + 0.5) * cell_size_c

    elif mode == 'cost_all':
        for c_idx in range(c_dim):
            V[N, :, :, c_idx] = (c_idx + 0.5) * cell_size_c
    else:
        raise ValueError("Invalid mode. Choose from 'safety', 'cost_safe', 'cost_unsafe', 'cost_all'.")

    # 2. PRE-MOVE Static Data to GPU (Do this ONCE)
    # Ensure types are explicit (float32 is often faster on consumer GPUs)
    d_safe_map = cuda.to_device(safe_map.astype(np.float32))
    d_target_map = cuda.to_device(target_map.astype(np.float32))
    d_transition_data = cuda.to_device(transition_data.astype(np.float32))
    d_pi = cuda.to_device(pi.astype(np.int32))

    # 3. Configure CUDA Grid
    threadsperblock = (8, 8, 4)  # 256 threads
    blockspergrid = (
        (x_dim + threadsperblock[0] - 1) // threadsperblock[0],
        (y_dim + threadsperblock[1] - 1) // threadsperblock[1],
        (c_dim + threadsperblock[2] - 1) // threadsperblock[2]
    )

    # 4. Value Iteration Loop
    # We move the current V_next to the GPU
    d_V_next = cuda.to_device(V[N].astype(np.float32))

    for k in tqdm(range(N-1, -1, -1)):
        # Create an empty array on GPU for the result of this timestep
        d_V_k = cuda.device_array((x_dim, y_dim, c_dim), dtype=np.float32)

        # Launch Kernel
        evaluate_safety_gpu_kernel[blockspergrid, threadsperblock](
            d_V_next, d_safe_map, d_target_map, d_transition_data,
            x_dim, y_dim, c_dim, u_dim, samples_per_action,
            cell_size_x, cell_size_y, cell_size_c, d_pi, k, d_V_k, mode_idx
        )

        # Copy result back to CPU storage
        V[k] = d_V_k.copy_to_host()

        # The current result becomes the 'next' value for the next iteration
        # We keep it on the GPU to avoid a Host->Device transfer next loop
        d_V_next = d_V_k

    # Final Indexing
    initial_x_idx = int(np.floor(initial_state[0] / cell_size_x))
    initial_y_idx = int(np.floor(initial_state[1] / cell_size_y))
    initial_c_idx = 0

    #print(f"Value function at initial state: {V[0, initial_x_idx, initial_y_idx, initial_c_idx]}")
    return V[0, initial_x_idx, initial_y_idx, initial_c_idx]


import numpy as np
import math
from numba import cuda


@cuda.jit
def DP_step_core_3D_gpu(V_next, safe_map, target_map, transition_data,
                        x_dim, y_dim, c_dim, u_dim, samples_per_action,
                        cell_size_x, cell_size_y, cell_size_c,
                        pi_k, V_k):
    """
    GPU Kernel: Each thread computes one (x, y, c) cell.
    """
    # 1. Get current thread position
    x_idx, y_idx, c_idx = cuda.grid(3)

    # 2. Bounds check (in case grid size > array size)
    if x_idx < x_dim and y_idx < y_dim and c_idx < c_dim:

        # Handle terminal/static cases
        if target_map[x_idx, y_idx] == 1:
            V_k[x_idx, y_idx, c_idx] = V_next[x_idx, y_idx, c_idx]
            pi_k[x_idx, y_idx, c_idx] = 0  # Default action
            return

        if safe_map[x_idx, y_idx] == 0:
            V_k[x_idx, y_idx, c_idx] = V_next[x_idx, y_idx, c_idx]
            pi_k[x_idx, y_idx, c_idx] = 0
            return

        # 3. Optimization over actions
        best_val = -1e15  # Negative infinity
        best_action = 0

        for action_idx in range(u_dim):
            sum_val = 0.0
            for realization in range(samples_per_action):
                # Fetch transition (assuming transition_data is [u, samples, 3])
                tx = transition_data[action_idx, realization, 0]
                ty = transition_data[action_idx, realization, 1]
                tc = transition_data[action_idx, realization, 2]

                x_grid_sized = x_idx + 0.5
                y_grid_sized = y_idx + 0.5
                c_grid_sized = c_idx + 0.5

                x_next = x_grid_sized + (tx / cell_size_x)
                y_next = y_grid_sized + (ty / cell_size_y)
                c_next = c_grid_sized + (tc / cell_size_c)

                # Use your existing device function
                status, curr_x, curr_y = collision_check_line_gpu(
                    safe_map, target_map,
                    x_grid_sized, y_grid_sized,
                    x_next, y_next
                )

                # Clip cost dimension
                curr_c = c_next
                if curr_c >= c_dim:
                    curr_c = float(c_dim - 1)
                elif curr_c < 0:
                    curr_c = 0.0

                # Status codes: 0 = unsafe, 1 = target, 2 = safe_no_target
                if status == 1 or status == 2:
                    sum_val += V_next[int(curr_x), int(curr_y), int(curr_c)]

            expected_val = sum_val / samples_per_action

            if expected_val > best_val:
                best_val = expected_val
                best_action = action_idx

        # 4. Write results to global memory
        pi_k[x_idx, y_idx, c_idx] = best_action
        V_k[x_idx, y_idx, c_idx] = best_val


def DP_JCC_only_safe_fixed_lambda_gpu(transition_data, maps, lambda_val):
    safe_map, target_map = maps
    x_dim, y_dim, c_dim = get_discretization(safe_map)
    u_dim, samples_per_action, _ = transition_data.shape
    N = DP_PARAMS['N']

    # 1. Move static data to GPU
    d_safe_map = cuda.to_device(safe_map.astype(np.float32))
    d_target_map = cuda.to_device(target_map.astype(np.float32))
    d_transition_data = cuda.to_device(transition_data.astype(np.float32))

    # 2. Allocate V and pi on CPU (or GPU if memory permits)
    V = np.zeros((N + 1, x_dim, y_dim, c_dim), dtype=np.float32)
    pi = np.zeros((N, x_dim, y_dim, c_dim), dtype=np.int32)

    # Initialize Terminal Cost
    cost_cutoff = int(lambda_val / DP_PARAMS['cost_space_discretization'])
    # Bound the cost_cutoff to c_dim to avoid indexing issues
    if cost_cutoff >= c_dim:
        cost_cutoff = c_dim - 1


    for c_idx in range(cost_cutoff + 1):
        V[N, :, :, c_idx] = target_map * lambda_val - target_map *(c_idx + 0.5) * DP_PARAMS['cost_space_discretization']

    # Move the "Next V" to device
    d_V_next = cuda.to_device(V[N])

    # 3. Configure GPU Blocks/Threads
    threads_per_block = (8, 8, 8)
    blocks_per_grid_x = math.ceil(x_dim / threads_per_block[0])
    blocks_per_grid_y = math.ceil(y_dim / threads_per_block[1])
    blocks_per_grid_z = math.ceil(c_dim / threads_per_block[2])
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y, blocks_per_grid_z)


    cell_size_x = DP_PARAMS['state_space_physical_size'][0] / x_dim
    cell_size_y = DP_PARAMS['state_space_physical_size'][1] / y_dim
    cell_size_c = DP_PARAMS['cost_space_discretization']

    # 4. Main Time Loop
    for k in tqdm(range(N - 1, -1, -1)):
        d_V_k = cuda.device_array((x_dim, y_dim, c_dim), dtype=np.float32)
        d_pi_k = cuda.device_array((x_dim, y_dim, c_dim), dtype=np.int32)

        DP_step_core_3D_gpu[blocks_per_grid, threads_per_block](
            d_V_next, d_safe_map, d_target_map, d_transition_data,
            x_dim, y_dim, c_dim, u_dim, samples_per_action,
            cell_size_x, cell_size_y, cell_size_c,
            d_pi_k, d_V_k
        )

        # Copy back results for this timestep
        V[k] = d_V_k.copy_to_host()
        pi[k] = d_pi_k.copy_to_host()

        # Reuse d_V_k as the next V
        d_V_next = d_V_k

    return pi, V



import os
from matplotlib.colors import ListedColormap

from plotting_functions import plot_trajectories
def plot_trajectory_sweep_animation(map):
    lambda_values = DP_PARAMS['LAMBDA_LIST']
    directory = os.path.join("data", DP_PARAMS['OBJECTIVE'], TRACK_NAME)
    font_size = 30

    # 1. Colors & Colormaps# A cohesive, color-blind safe palette (Okabe-Ito / Wong inspired)
    CB_TARGET      = '#0072B2'  # A strong, professional Blue (Clearer than DarkBlue)
    CB_SAFE_TRAJ   = '#009E73'  # A vibrant, distinct Green (Bluish-green, very safe)
    CB_UNSAFE_TRAJ = '#D55E00'  # Vermillion (The gold standard for "Red" in CVD)

    # Custom colormaps for layered transparency
    unsafe_mask_cmap = ListedColormap([(0, 0, 0, 0), 'white'])
    #unsafe_mask_cmap = ListedColormap([(0, 0, 0, 0), 'black'])
    target_map_cmap = ListedColormap([(0, 0, 0, 0), CB_TARGET])
    safe_background_cmap = ListedColormap([(0, 0, 0, 0), 'white'])

    import numpy as np
    fig, ax = plt.subplots(figsize=(8, 8))
    # Load all trajectory data for all lambda
    idx = 0
    for lambda_val in lambda_values:
        #trajectories = np.load( os.path.join(directory,f"trajectories_lambda_{lambda_val}.npy"))
        # Create and save animation for this lambda

        plot_trajectories(lambda_val = lambda_val, plot = True, plot_number = 0, lambda_title = lambda_val)
        if idx == 0:
            plt.pause(10)
        else:
            plt.pause(0.1)

        idx = idx+1