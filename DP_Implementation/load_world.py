# Authors, Niklas Schmid, Jared Miller, Tristan Zeller,
# Marta Fochesato, Tobias Sutter, John Lygeros 2026
#
# This source code is licensed under the license
# found in the LICENSE file in the root directory of this source tree.


import numpy as np
import imageio.v3 as iio
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Generate map of track by loading the respective png
def load_world(track_name):
    """
    Loads a track from a PNG image and creates safety and target sets for a
    control or path planning problem.

    The function assumes the track is represented by a PNG image where the red
    channel (im_r) represents the 'safe' area and the blue channel (im_b)
    represents the 'target' area. Both channels are used to create a safety_map.
    The blue channel is then modified to represent the target set by subtracting
    the red channel, effectively isolating the blue area.

    Args:
        track_name (str): The name of the track file (without the '.png'
                           extension).

    Returns:
        list: A list containing two NumPy arrays:
              - safe_set: A binary NumPy array representing the safe area.
              - target_set: A binary NumPy array representing the target area.
    """

    print("Loading track.")
    im = iio.imread("worlds/" + track_name + ".png")   # Loads track from png file
    im_r = np.array(im[:, :, 0]) / 255.0
    im_b = np.array(im[:, :, 2]) / 255.0

    card_X_x = np.shape(im)[0]
    card_X_y = np.shape(im)[1]
    safety_map = np.zeros((card_X_x,card_X_y))
    for x_idx in range(card_X_x):  # for all states in x
        for y_idx in range(card_X_y):  # for all states in y
            if im_r[x_idx,y_idx]>0 or im_b[x_idx,y_idx]>0:
                safety_map[x_idx,y_idx] = 1

    im_b = im_b - im_r
    #plt.figure(0)
    #plt.imshow(safety_map, cmap='hot', interpolation='nearest')
    #plt.show() # Plots track
    safe_set = im_r
    target_set = im_b
    return safe_set, target_set

def debug_plot_world(safe_set, target_set):
    plt.figure(0)
    plt.imshow(safe_set, cmap='hot', interpolation='nearest')
    plt.title("Safe Set")
    plt.figure(1)
    plt.imshow(target_set, cmap='hot', interpolation='nearest')
    plt.title("Target Set")
    plt.show()