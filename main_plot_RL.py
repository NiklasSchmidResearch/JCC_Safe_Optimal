# Authors, Niklas Schmid, Jared Miller, Tristan Zeller,
# Marta Fochesato, Tobias Sutter, John Lygeros 2026
#
# This source code is licensed under the license
# found in the LICENSE file in the root directory of this source tree.


from plotting_functions import plot_trajectories
import parameters
import numpy as np

def refine_RL_data(trajectories, lambda_val = 0):
    # dimensions are N, NUM_STATES + NUM_ACTIONS, SAMPLES SIZE, but should be  SAMPLES SIZE, N, NUM_STATES + COST_STATE
    trajectories = np.transpose(trajectories, (2, 0, 1))
    # First x,y state is nan, but should be the initial state (0.5, 1)
    trajectories[:, 0, 0] = parameters.DP_PARAMS['INITIAL_STATE'][0]
    trajectories[:, 0, 1] = parameters.DP_PARAMS['INITIAL_STATE'][1]

    # The states become nan at some point; just take the value of the previous timestep
    for samples in range(trajectories.shape[0]):
        for step in range(1, trajectories.shape[1]):
            if np.isnan(trajectories[samples, step, 0]):
                trajectories[samples, step, 0] = trajectories[samples, step - 1, 0]
            if np.isnan(trajectories[samples, step, 1]):
                trajectories[samples, step, 1] = trajectories[samples, step - 1, 1]

    # for the inputs, just replace nans with zeros
    for samples in range(trajectories.shape[0]):
        for step in range(trajectories.shape[1]):
            if np.isnan(trajectories[samples, step, 3]):
                trajectories[samples, step, 3] = 0
            if np.isnan(trajectories[samples, step, 4]):
                trajectories[samples, step, 4] = 0

    # Overwrite the third state with cost based on inputs
    for samples in range(trajectories.shape[0]):
        cost = 0
        for step in range(trajectories.shape[1]):
            if step > 0:
                # cost is the second input
                cost += trajectories[samples, step-1, 3]
            trajectories[samples, step, 2] = cost

    # Remove the inputs from the trajectories, so that the dimensions are SAMPLES SIZE, N, NUM_STATES
    trajectories = trajectories[:, :, :3]
    return trajectories


if __name__ == '__main__':
    file_name = f"data/{parameters.DP_PARAMS['OBJECTIVE']}/{parameters.TRACK_NAME}/trajectories.npy"
    trajectories_source = np.load(file_name)
    trajectories_refined = refine_RL_data(trajectories_source)

    file_name_label = f"data/{parameters.DP_PARAMS['OBJECTIVE']}/{parameters.TRACK_NAME}/labels.npy"
    labels = np.load(file_name_label)


    # Add labels as an additional state
    #trajectories = np.insert(trajectories_refined, 3, 0, axis=2)
    #  labals are vectors of size SAMPLES SIZE; I want them to be added as an additional state to the trajectories (2nd dimension) at every time-step

    labels_expanded = labels[:, np.newaxis, np.newaxis]

    # 2. Create a 'tiled' version of the labels that matches TimeSteps
    # np.tile(array, (1, TimeSteps, 1)) repeats the label for every step
    labels_tiled = np.tile(labels_expanded, (1, trajectories_refined.shape[1], 1))

    # 3. Concatenate along the state/feature axis (axis 2)
    trajectories_with_labels = np.concatenate((trajectories_refined, labels_tiled), axis=2)

    file_name_destination = f"data/{parameters.DP_PARAMS['OBJECTIVE']}/{parameters.TRACK_NAME}/trajectories_lambda_0.npy"
    np.save(file_name_destination, trajectories_with_labels)
    plot_trajectories()