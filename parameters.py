# Authors, Niklas Schmid, Jared Miller, Tristan Zeller,
# Marta Fochesato, Tobias Sutter, John Lygeros 2026
#
# This source code is licensed under the license
# found in the LICENSE file in the root directory of this source tree.


import numpy as np

# Slit: alpha = 0.55
# RA:

import os
import re

def load_lambda_values_from_files(objective, track_name):

    # The directory containing your files
    directory = os.path.join("data", objective, track_name)

    # List to store the extracted values
    lambda_list = []

    # Regex pattern to find lambda values.
    # This looks for "lambda" followed by an "=" or ":" and then a number.
    # You may need to tweak this depending on exactly how it is written in your files.
    pattern = re.compile(r"lambda\s*[=:]\s*([-+]?[0-9]*\.?[0-9]+)")

    if os.path.exists(directory):
        # Loop through all files in the directory
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)

            if os.path.isfile(filepath):
                match = re.search(r"([-+]?[0-9]*\.?[0-9]+)", filename)
                if match:
                    lambda_list.append(float(match.group(1)))

        print(f"Found {len(lambda_list)} lambda values:")
        print(lambda_list)
    else:
        print(f"Could not find the directory: {directory}")

    return lambda_list

RUN_OPTIONS = {
    'GENERATE_NEW_TRAJECTORY_DATA': True,  # Whether generate new trajectory data or load from file
    'RUN_DP': True  # Whether to run the DP recursion or load the value function and policy from file
}

TRACK_NAME = "ra_50_pass"

# Simulation parameters:
sim_params = {
    'NUM_ACTIONS': 50,  # Size of the action set
    'NUM_SAMPLES_PER_ACTION': 500,  # Number of trajectories sampled for each action set
    'NOISE_COVARIANCE': np.diag([0.025, 0.025]),#np.diag([0.025, 0.025]),  # Covariance of the noise w_{1,k}
    'NOISE_STD': np.sqrt(0.01),  #0.01 Value in sqrt is variance of the noise w_{2,k}
}

# DP parameters:
DP_PARAMS = {
    'N': 15,  # Number of time-steps for DP recursion
    'cost_space_physical_size': 3*50, # Dimension of the cost space (c) is max cost times timesteps
    'OBJECTIVE': 'JCC_onlySafe_sweep', # 'RA' only reach-avoid, 'JCC_classic' classical JCC problems, 'JCC_classic_sweep', 'JCC_only_safe', 'JCC_onlySafe_sweep' our JCC variant
    'state_space_physical_size': (10,10), # Dimension of the physical state space (x,y)
    'cost_space_discretization': 0.2, # Discretization of the cost space (c)
    'INITIAL_STATE': (0.5, 1),#(0.5, 1), # Initial state for the problem
    'LAMBDA_LIST': [18.0] ,#[12, 12.5, 13, 13.5, 14]#[10]#[20, 22, 24, 26, 28], # List of lambda values for classical JCC problems
    'alpha': 0.6#75
}

#DP_PARAMS['LAMBDA_LIST'] = load_lambda_values_from_files(DP_PARAMS['OBJECTIVE'], TRACK_NAME)

# slit alpha = 0.6
# RA classic: [20, 22, 24, 26, 28]

# RA ours: [10]

def get_discretization(safe_map):
    """Get the discretization of the state space based on the shape of the safe map."""
    x_dim, y_dim = safe_map.shape
    c_dim = int(DP_PARAMS['N']*3 / DP_PARAMS['cost_space_discretization']) # Basically means 5 different levels of the input
    return x_dim, y_dim, c_dim