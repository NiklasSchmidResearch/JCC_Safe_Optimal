# Authors, Niklas Schmid, Jared Miller, Tristan Zeller,
# Marta Fochesato, Tobias Sutter, John Lygeros 2026
#
# This source code is licensed under the license
# found in the LICENSE file in the root directory of this source tree.


import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import parameters
import os
from parameters import DP_PARAMS
from dynamic_programming import collision_check_line, evaluate_policy_safety_via_value_iteration_gpu
from load_world import load_world

# CHANGE THESE TWO VALUES AS NEEDED:
# Where your trajectory data is stored:
file_name = f"data/{parameters.DP_PARAMS['OBJECTIVE']}/{parameters.TRACK_NAME}/trajectories.npy"
# Which background picture to use for the track:
track_name = parameters.TRACK_NAME


safe_set, target_set = load_world(track_name)
map = (safe_set, target_set)
font_size = 30

import matplotlib.ticker as ticker

import matplotlib.ticker as ticker


def plot_trajectories(lambda_val = 0, plot = True, file_to_load_from = None, plot_number = None, animate_evolution = False, lambda_title=None):
    from matplotlib.colors import ListedColormap
    import numpy as np
    if file_to_load_from is not None:
        file_name = file_to_load_from
    else:
        if parameters.DP_PARAMS['OBJECTIVE'] == 'RA' or parameters.DP_PARAMS['OBJECTIVE'] == 'JCC_only_safe' or parameters.DP_PARAMS['OBJECTIVE'] == 'JCC_classic':
            file_name = f"data/{parameters.DP_PARAMS['OBJECTIVE']}/{parameters.TRACK_NAME}/trajectories.npy"
        else:
            file_name = f"data/{parameters.DP_PARAMS['OBJECTIVE']}/{parameters.TRACK_NAME}/trajectories_lambda_{lambda_val}.npy"

    if not parameters.DP_PARAMS['OBJECTIVE'] == 'RL':
        trajectories = np.load(file_name)
    else:
        file_name = f"data/{parameters.DP_PARAMS['OBJECTIVE']}/{parameters.TRACK_NAME}/trajectories_lambda_{lambda_val}.npy"
        trajectories = np.load(file_name)

        # Add an additional state to trajectories that is zero.




    # 1. Colors & Colormaps# A cohesive, color-blind safe palette (Okabe-Ito / Wong inspired)
    CB_TARGET      = '#0072B2'  # A strong, professional Blue (Clearer than DarkBlue)
    CB_SAFE_TRAJ   = '#009E73'  # A vibrant, distinct Green (Bluish-green, very safe)
    CB_UNSAFE_TRAJ = '#D55E00'  # Vermillion (The gold standard for "Red" in CVD)

    # Custom colormaps for layered transparency
    unsafe_mask_cmap = ListedColormap([(0, 0, 0, 0), 'white'])
    #unsafe_mask_cmap = ListedColormap([(0, 0, 0, 0), 'black'])
    target_map_cmap = ListedColormap([(0, 0, 0, 0), CB_TARGET])
    safe_background_cmap = ListedColormap([(0, 0, 0, 0), 'white'])

    safe_map, target_map = map
    num_rollouts = trajectories.shape[0]
    success_of_rollout = np.zeros(num_rollouts, dtype=bool)

    if plot:
        if plot_number is None:
            fig, ax = plt.subplots(figsize=(8, 8))
            #print("Pausing for 5 seconds for allow for time to start recording the plot with OBS...")
            #plt.pause(10)
        else:
            # This selects the figure (creating it if it doesn't exist)
            fig = plt.figure(plot_number, figsize=(8, 8))
            # This gets the current axes or creates one if the figure is empty
            ax = fig.gca()
            # clear the plot
            ax.clear()


    # --- PRE-PROCESSING MAPS ---
    background_mask = np.where(safe_map == 1, 1, 0)
    unsafe_mask = np.where(safe_map == 0, 1, 0)

    # --- LAYER 1: Background (Safe Area) ---









    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    import numpy as np
    import matplotlib.patches as patches

    # Increase the line thickness globally so stripes are distinct from "blobs"
    plt.rcParams['hatch.linewidth'] = 1.5

    def add_striped_patches(ax, mask, hatch_pattern, color, zorder=10):
        width, height = DP_PARAMS['state_space_physical_size']
        rows, cols = mask.shape
        dx, dy = width / cols, height / rows

        data = mask.T

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if data[i, j] == 1:
                    if parameters.TRACK_NAME != "ra_50_labyrinth" or hatch_pattern == '/':
                        rect = patches.Rectangle(
                            xy=(i * dx, j * dy),
                            width=dx,
                            height=dy,
                            facecolor="white",  # Solid background
                            edgecolor=color,  # Stripe color
                            hatch=hatch_pattern,  # The pattern
                            linewidth=0,  # REMOVES THE BOX BORDER
                            antialiased=False,  # Prevents tiny 'ghost' lines between cells
                            zorder=zorder
                        )
                        ax.add_patch(rect)
                    else:
                        rect = patches.Rectangle(
                            xy=(i * dx, j * dy),
                            width=dx,
                            height=dy,
                            facecolor="black",  # Changed from "white" to "black"
                            edgecolor="none",  # Set to "none" since you have linewidth=0 anyway
                            hatch=None,  # Explicitly remove the pattern for a solid fill
                            linewidth=0,
                            antialiased=False,
                            zorder=zorder
                        )
                        ax.add_patch(rect)
    # 1. Unsafe Map: Crosses (Hatched)
    # Using 'xxx' for a dense cross-hatch

    plt.imshow(unsafe_mask.T, cmap=unsafe_mask_cmap)
    if plot:
        # Make thick stripes for the unsafe areas using hatch patterns. 'x' is not dense so we add thicknessimport matplotlib as mpl
        #
        # # This changes the thickness of ALL hatches/stripes drawn afterwards
        if not parameters.TRACK_NAME == "ra_50_labyrinth":
            matplotlib.rcParams['hatch.linewidth'] = 4.0  # Increase this if you want them even thicker
            pattern = 'x'
        else:
            pattern = ''
        add_striped_patches(ax, unsafe_mask.T - target_map.T, pattern, color='black', zorder=10)

    # 2. Target Map: Single Slanted Stripes (Striped)
    # Using '///' for clear diagonal stripes
    if plot and animate_evolution==False:
        matplotlib.rcParams['hatch.linewidth'] = 8.0  # Increase this if you want them even thicker
        add_striped_patches(ax, target_map.T, '/', color=CB_TARGET, zorder=11)

        # --- LAYER 2: Trajectories ---
        if not parameters.DP_PARAMS['OBJECTIVE'] == 'RL':
            cost_of_safe_trajectories = 0
            cost_of_unsafe_trajectories = 0

            for rollout_idx in range(num_rollouts):
                cost_of_trajectory = trajectories[rollout_idx, -1, 2]
                for step_idx in range(trajectories.shape[1] - 1):
                    x, y = trajectories[rollout_idx, step_idx, 0], trajectories[rollout_idx, step_idx, 1]
                    next_x, next_y = trajectories[rollout_idx, step_idx + 1, 0], trajectories[
                        rollout_idx, step_idx + 1, 1]

                    scale_x = DP_PARAMS['state_space_physical_size'][0] / safe_map.shape[0]
                    scale_y = DP_PARAMS['state_space_physical_size'][1] / safe_map.shape[1]

                    status, (end_x, end_y) = collision_check_line(safe_map, target_map, x / scale_x, y / scale_y,
                                                                  next_x / scale_x, next_y / scale_y)
                    # status = "safe"
                    if status in ["unsafe", "target"]:
                        for fill_idx in range(step_idx + 1, trajectories.shape[1]):
                            trajectories[rollout_idx, fill_idx, 0] = end_x * scale_x
                            trajectories[rollout_idx, fill_idx, 1] = end_y * scale_y
                        if status == "target":
                            success_of_rollout[rollout_idx] = True
                        break

                color = CB_SAFE_TRAJ if success_of_rollout[rollout_idx] else CB_UNSAFE_TRAJ
                if plot:
                    ax.plot(trajectories[rollout_idx, :, 0], trajectories[rollout_idx, :, 1], color=color, alpha=0.6, linewidth=1.5, zorder=2)

                if success_of_rollout[rollout_idx]:
                    cost_of_safe_trajectories += cost_of_trajectory
                else:
                    cost_of_unsafe_trajectories += cost_of_trajectory
        else:
            for rollout_idx in range(num_rollouts):
                color = CB_SAFE_TRAJ if trajectories[rollout_idx, 0, 3] == 1 else CB_UNSAFE_TRAJ
                # if trajectories[rollout_idx, 0, 3] == 0:
                ax.plot(trajectories[rollout_idx, :, 0], trajectories[rollout_idx, :, 1],
                        color=color, alpha=0.6, linewidth=1.5, zorder=2)

    if plot and animate_evolution:
        matplotlib.rcParams['hatch.linewidth'] = 8.0  # Increase this if you want them even thicker

        add_striped_patches(ax, target_map.T, '/', color=CB_TARGET, zorder=11)
        plt.xlim(0, DP_PARAMS['state_space_physical_size'][0])
        plt.ylim(0, DP_PARAMS['state_space_physical_size'][1])

        # --- LAYER 2: Trajectories ---
        # 1. PRE-CALCULATION PHASE
        # We must determine success/failure for ALL trajectories before we start animating
        if not parameters.DP_PARAMS['OBJECTIVE'] == 'RL':
            cost_of_safe_trajectories = 0
            cost_of_unsafe_trajectories = 0

            for rollout_idx in range(num_rollouts):
                # Determine the outcome of this rollout
                for step_idx in range(trajectories.shape[1] - 1):
                    x, y = trajectories[rollout_idx, step_idx, 0], trajectories[rollout_idx, step_idx, 1]
                    next_x, next_y = trajectories[rollout_idx, step_idx + 1, 0], trajectories[
                        rollout_idx, step_idx + 1, 1]

                    scale_x = DP_PARAMS['state_space_physical_size'][0] / safe_map.shape[0]
                    scale_y = DP_PARAMS['state_space_physical_size'][1] / safe_map.shape[1]

                    status, (end_x, end_y) = collision_check_line(safe_map, target_map, x / scale_x, y / scale_y,
                                                                  next_x / scale_x, next_y / scale_y)

                    if status in ["unsafe", "target"]:
                        # Clip trajectory to collision point
                        for fill_idx in range(step_idx + 1, trajectories.shape[1]):
                            trajectories[rollout_idx, fill_idx, 0] = end_x * scale_x
                            trajectories[rollout_idx, fill_idx, 1] = end_y * scale_y
                        if status == "target":
                            success_of_rollout[rollout_idx] = True
                        break

                # Accumulate costs
                cost_of_trajectory = trajectories[rollout_idx, -1, 2]
                if success_of_rollout[rollout_idx]:
                    cost_of_safe_trajectories += cost_of_trajectory
                else:
                    cost_of_unsafe_trajectories += cost_of_trajectory

        # 2. ANIMATION PHASE
        if plot:
            # Pre-assign colors based on the results from Phase 1
            rollout_colors = []
            for r_idx in range(num_rollouts):
                if parameters.DP_PARAMS['OBJECTIVE'] == 'RL':
                    is_safe = (trajectories[r_idx, 0, 3] == 1)
                else:
                    is_safe = success_of_rollout[r_idx]

                rollout_colors.append(CB_SAFE_TRAJ if is_safe else CB_UNSAFE_TRAJ)

            # Plot one timestep at a time for ALL rollouts simultaneously
            for step_idx in range(trajectories.shape[1] - 1):
                for r_idx in range(num_rollouts):
                    ax.plot(trajectories[r_idx, step_idx:step_idx + 2, 0],
                            trajectories[r_idx, step_idx:step_idx + 2, 1],
                            color=rollout_colors[r_idx],
                            alpha=0.6,
                            linewidth=1.5,
                            zorder=2)

                # Pause after drawing this step for every trajectory
                plt.pause(0.01)

    if plot:
        # --- GRID & AXIS FORMATTING ---
        ax.set_xlabel('x', fontsize=font_size)
        ax.set_ylabel('y', fontsize=font_size)

        # 1. Labels/Major Ticks every 2 units
        ax.xaxis.set_major_locator(ticker.MultipleLocator(2.0))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(2.0))

        # 2. Minor Ticks every 1 unit (for the grid lines)
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(1.0))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(1.0))

        ax.set_xlim(0, DP_PARAMS['state_space_physical_size'][0])
        ax.set_ylim(0, DP_PARAMS['state_space_physical_size'][1])

        ax.tick_params(axis='both', which='major', labelsize=font_size)

        # 3. Enable grid for BOTH major and minor ticks
        ax.grid(which='both', color='gray', linestyle='--', linewidth=0.5, alpha=0.7, zorder=1.5)
        if lambda_title is not None:
            ax.set_title(f"$\lambda=$ {lambda_title}", fontsize=font_size)
        plt.tight_layout()
        plt.show()

    # Stats
    if not parameters.DP_PARAMS['OBJECTIVE'] == 'RL':
        safe_count = np.sum(success_of_rollout)
        unsafe_count = np.sum(~success_of_rollout)
        empirical_safety= safe_count / num_rollouts
        expected_cost_safe = cost_of_safe_trajectories / safe_count if safe_count > 0 else 'N/A'
        expected_cost_unsafe = cost_of_unsafe_trajectories / unsafe_count if unsafe_count > 0 else 'N/A'
        expected_cost_all = (cost_of_safe_trajectories + cost_of_unsafe_trajectories)/(unsafe_count+safe_count) if unsafe_count > 0 and safe_count > 0 else 'N/A'
    else:
        safe_count = np.sum(trajectories[:, 0, 3] == 1)
        unsafe_count = np.sum(trajectories[:, 0, 3] == 0)
        empirical_safety= safe_count / num_rollouts
        expected_cost_safe = np.mean(trajectories[trajectories[:, 0, 3] == 1][:, -1, 2])
        expected_cost_unsafe = np.mean(trajectories[trajectories[:, 0, 3] == 0][:, -1, 2])
        expected_cost_all = np.mean(trajectories[:, -1, 2])

    print(f"Empirical safety probability: {empirical_safety:.2f} ({safe_count} safe out of {num_rollouts} total)")
    print(f"Expected cost of safe trajectories: {expected_cost_safe}")
    print(f"Expected cost of unsafe trajectories: {expected_cost_unsafe}")
    print(f"Expected cost of all trajectories: {expected_cost_all}")
    # Save plot as png in figs directory using the mode and trackname
    track_name = parameters.TRACK_NAME
    mode = DP_PARAMS['OBJECTIVE']
    if parameters.DP_PARAMS['OBJECTIVE'] == 'JCC_classic_sweep' or parameters.DP_PARAMS['OBJECTIVE'] == 'JCC_onlySafe_sweep':
        # Replace dots in lambda_val due to floating point by "_"
        lambda_val_text = str(lambda_val).replace('.', '_')
        mode += f"_lambda_{lambda_val_text}"
    if plot:
        plt.savefig(f"figs/{mode}_{track_name}")
    return empirical_safety, expected_cost_safe, expected_cost_unsafe, expected_cost_all


def plot_cost_of_jcc_and_only_safe_trajectories(transition_data, initial_state, recompute_front = False):
    print("Computing Pareto Fronts of Only-Safe and Classical JCC Approach")
    if not recompute_front:
        data_only_safe = np.load(f"data/expected_cost_vs_safety_probability_only_safe.npy")
        data_jcc_classic = np.load(f"data/expected_cost_vs_safety_probability_jcc_classic.npy")

        # Make the data more scarce, only pick one point every 0.03 safety probability
        def pick_points(data):
            # first sort data by safety probability
            data = data[:, data[0].argsort()]


            picked_data = []
            last_safety = -1
            for i in range(data.shape[1]):
                safety = data[0, i]
                if safety - last_safety >= 0.01:
                    picked_data.append(data[:, i])
                    last_safety = safety
            return np.array(picked_data).T
        data_only_safe = pick_points(data_only_safe)
        data_jcc_classic = pick_points(data_jcc_classic)



        # Print data as text for latex coordinate plots:
        print("Only Safe JCC Data (Safety, Expected Cost Safe, Expected Cost Unsafe, Expected Cost All):")
        for i in range(data_only_safe.shape[1]):
            print(f"{data_only_safe[0, i]:.4f} {data_only_safe[1, i]:.4f} {data_only_safe[2, i]:.4f} {data_only_safe[3, i]:.4f}")
        print("Classic JCC Data (Safety, Expected Cost Safe, Expected Cost Unsafe, Expected Cost All):")
        for i in range(data_jcc_classic.shape[1]):
            print(f"{data_jcc_classic[0, i]:.4f} {data_jcc_classic[1, i]:.4f} {data_jcc_classic[2, i]:.4f} {data_jcc_classic[3, i]:.4f}")

    else:
        data_only_safe_file = "data/JCC_onlySafe_sweep/ra_50_pass/"
        # load all files in the directory that start with "trajectories_lambda" and end with ".npy"
        files = [f for f in os.listdir(data_only_safe_file) if f.startswith("trajectories_lambda") and f.endswith(".npy")]
        lambda_values = []
        data_only_safe=np.zeros((4, len(files)))
        pi = {}
        for file in files:
            lambda_val = float(file[len("trajectories_lambda_"):-len(".npy")].replace('_', '.'))
            lambda_values.append(lambda_val)
            # load policy from data_only_safe_file+f"policy_lambda_{lambda_val}.npy":
            pi[lambda_val] = np.load(data_only_safe_file + f"policy_lambda_{lambda_val}.npy")
            empirical_safety, expected_cost_safe, expected_cost_unsafe, expected_cost_all = plot_trajectories(lambda_val=lambda_val, plot=False, file_to_load_from=data_only_safe_file+file)

            #empirical_safety = evaluate_policy_safety_via_value_iteration_gpu(transition_data, initial_state, map,
            #                                                                      pi[lambda_val], lambda_val)
            #expected_cost_safe = evaluate_policy_safety_via_value_iteration_gpu(transition_data, initial_state, map,
            #                                                                      pi[lambda_val], lambda_val, mode="cost_safe")
            #expected_cost_unsafe = evaluate_policy_safety_via_value_iteration_gpu(transition_data, initial_state, map,
            #                                                                        pi[lambda_val], lambda_val, mode="cost_unsafe")
            #expected_cost_all = evaluate_policy_safety_via_value_iteration_gpu(transition_data, initial_state, map,
            #                                                                        pi[lambda_val], lambda_val, mode="cost_all")
            #expected_cost_safe = expected_cost_safe / empirical_safety if empirical_safety > 0 else 0
            #expected_cost_unsafe = expected_cost_unsafe / (1 - empirical_safety) if empirical_safety < 1 else 0


            # Correct expectation by safety probability since cost of zero is assined to unsafe trajectories in the cost_safe case, and cost of zero is assigned to safe trajectories in the cost_unsafe case
            print(f"Lambda: {lambda_val}, Safety: {empirical_safety}, Expected Cost Safe: {expected_cost_safe}, Expected Cost Unsafe: {expected_cost_unsafe}, Expected Cost All: {expected_cost_all}")
            if empirical_safety > 0 and empirical_safety < 1:
                data_only_safe[:, lambda_values.index(lambda_val)] = [empirical_safety, expected_cost_safe, expected_cost_unsafe, expected_cost_all]

        pi = {}
        data_jcc_classic_file = "data/JCC_classic_sweep/ra_50_pass/"
        files = [f for f in os.listdir(data_jcc_classic_file) if f.startswith("trajectories_lambda") and f.endswith(".npy")]
        lambda_values_classic = []
        data_jcc_classic=np.zeros((4, len(files)))
        for file in files:
            lambda_val = float(file[len("trajectories_lambda_"):-len(".npy")].replace('_', '.'))
            lambda_values_classic.append(lambda_val)
            pi[lambda_val] = np.load(data_jcc_classic_file + f"policy_lambda_{lambda_val}.npy")
            empirical_safety, expected_cost_safe, expected_cost_unsafe, expected_cost_all = plot_trajectories(lambda_val=lambda_val, plot=False, file_to_load_from=data_jcc_classic_file+file)

            #empirical_safety = evaluate_policy_safety_via_value_iteration_gpu(transition_data, initial_state, map,
            #                                                                      pi[lambda_val], lambda_val)
            #expected_cost_safe = evaluate_policy_safety_via_value_iteration_gpu(transition_data, initial_state, map,
            #                                                                      pi[lambda_val], lambda_val, mode="cost_safe")
            #expected_cost_unsafe = evaluate_policy_safety_via_value_iteration_gpu(transition_data, initial_state, map,
            #                                                                        pi[lambda_val], lambda_val, mode="cost_unsafe")
            #expected_cost_all = evaluate_policy_safety_via_value_iteration_gpu(transition_data, initial_state, map,
            #                                                                        pi[lambda_val], lambda_val, mode="cost_all")

            #expected_cost_safe = expected_cost_safe / empirical_safety if empirical_safety > 0 else 0
            #expected_cost_unsafe = expected_cost_unsafe / (1 - empirical_safety) if empirical_safety < 1 else 0
            # convert any N/A to nan so that we can plot them
            if expected_cost_safe == 'N/A':
                expected_cost_safe = np.nan
            if expected_cost_unsafe == 'N/A':
                expected_cost_unsafe = np.nan
            if expected_cost_all == 'N/A':
                expected_cost_all = np.nan
            if empirical_safety == 'N/A':
                empirical_safety = np.nan

            print(f"Lambda: {lambda_val}, Safety: {empirical_safety}, Expected Cost Safe: {expected_cost_safe}, Expected Cost Unsafe: {expected_cost_unsafe}, Expected Cost All: {expected_cost_all}")
            if empirical_safety > 0 and empirical_safety < 1:
                data_jcc_classic[:, lambda_values_classic.index(lambda_val)] = [empirical_safety, expected_cost_safe, expected_cost_unsafe, expected_cost_all]
        np.save(f"data/expected_cost_vs_safety_probability_only_safe.npy", data_only_safe)
        np.save(f"data/expected_cost_vs_safety_probability_jcc_classic.npy", data_jcc_classic)


    # Plot expected cost of safe and unsafe trajectories against safety probability for both methods
    plt.figure(figsize=(8, 6))
    plt.plot(data_only_safe[0, :], data_only_safe[1, :], label='Only Safe JCC (safe)', marker='o', linestyle='None', markersize=10, markeredgewidth=2, markerfacecolor='none', markeredgecolor='green')
    plt.plot(data_only_safe[0, :], data_only_safe[2, :], label='Only Safe JCC (Unsafe)', marker='o', linestyle='None', markersize=10, markeredgewidth=2, markerfacecolor='none', markeredgecolor='red')
    plt.plot(data_only_safe[0, :], data_only_safe[3, :], label='Only Safe JCC (all)', marker='o', linestyle='None', markersize=10, markeredgewidth=2, markerfacecolor='none', markeredgecolor='black')
    plt.plot(data_jcc_classic[0, :], data_jcc_classic[1, :], label='Classic JCC (safe)', marker='x', linestyle='None', markersize=10, markeredgewidth=2, markerfacecolor='none', markeredgecolor='green')
    plt.plot(data_jcc_classic[0, :], data_jcc_classic[2, :], label='Classic JCC (Unsafe)', marker='x', linestyle='None', markersize=10, markeredgewidth=2, markerfacecolor='none', markeredgecolor='red')
    plt.plot(data_jcc_classic[0, :], data_jcc_classic[3, :], label='Classic JCC (all)', marker='x', linestyle='None', markersize=10, markeredgewidth=2, markerfacecolor='none', markeredgecolor='black')
    plt.xlabel('Empirical Safety Probability')
    plt.ylabel('Expected Cost')
    plt.title('Expected Cost of Safe and Unsafe Trajectories vs. Empirical Safety Probability')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"figs/expected_cost_vs_safety_probability.png")

    plt.show(block=True)

    return data_only_safe, data_jcc_classic
