import numpy as np
import copy
import os
from concurrent.futures import ThreadPoolExecutor

from SACAgent import SACAgent
from unicycle_env import UnicycleGymEnv
from math import pi
from networks_github import ActorNetworkSAC, CriticNetworkSAC

import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import torch
import argparse

from plot import plot_traj, plot_costs

def train(agent: SACAgent, env: UnicycleGymEnv, num_ep, max_steps, start_steps, print_rate, path, lambda_0, lambda_lr, target_safety, 
          window_size, deterministic_data_collection = False, updates_per_step = 1, dual_after = 20_000, method = "new"):

    """
    Trains a soft actor-critic agent to solve the jcc reach-avoid task.

    inputs:
        - agent:            SAC agent
        - env:              Gymnasium-like environment for reach-avoid task
        - num_ep:           Number of training episodes
        - max_steps:        Number of steps in one episode before termination
        - start_steps:      Number of steps with unoformally sampled actions at the start of training
        - print_rate:       How often to print training updates to terminal
        - path:             Path to neural network weights
        - lambda_0:         Inital value for dual variable
        - lambda_lr:        Dual learning rate
        - target_safety:    Targeted success-rate for reach-avoid task
        - window_size:      Number of samples to estimate dual gradient

        - determinist_data_collection:  Whether to use mean of gaussian policy or sampling policy to collect data
        - updates_per_step:             Number of SAC updates per environment step
        - dual after:                   Number of episodes with fixed dual variable before dual updates commence
        - method:                       whether to use classical approach with immediate rewards or new approach with accumulated rewards

    outputs:
        None
    """

    reward_tracker = []

    t_start = time.time()

    print(f"Start at {datetime.fromtimestamp(t_start, tz=ZoneInfo("Europe/Zurich")).strftime("%H:%M:%S")}")
    print("Commencing Training...")

    success_tracker = np.zeros(num_ep)
    total_steps = 0

    alpha = agent.alpha

    dual = lambda_0

    for ep in range(num_ep):

        state, _ = env.reset()
        aug_state = np.append(state, 0)

        avg_reward = 0
        avg_a_loss = 0
        avg_c_loss = 0

        for it in range(max_steps):

            if total_steps <= start_steps:
                action = env.action_space.sample()

            else:
                action = agent.get_action(aug_state, deterministic_data_collection)
                action = np.clip(action, env.action_space.low, env.action_space.high).reshape(-1)


            next_state, reward, terminated, truncated, info = env.step(action)
            aug_next_state = np.append(next_state, reward + aug_state[2])

            success = info["success"]

            if success:
                success_tracker[ep] = 1
                aug_reward = reward + aug_state[2]
            else:
                if method == "new":
                    # Safe Trajectories
                    aug_reward = 0.0

                else:
                    # Classic JCC
                    aug_reward = reward

            agent.store_transition(aug_state, action, aug_reward, aug_next_state, terminated, success)

            avg_reward += reward

            if len(agent.replay_buffer) > agent.batch_size + 1 and total_steps > start_steps:
                a_loss, c_loss = 0, 0
                
                for _ in range(updates_per_step):
                    a_loss, c_loss, alpha_loss, alpha = agent.sac_update(dual = dual)

                agent.soft_target_update()

                avg_a_loss += a_loss
                avg_c_loss += c_loss

            if terminated or truncated:
                avg_a_loss /= (it + 1)
                avg_c_loss /= (it + 1)

                break

            state = next_state
            aug_state = aug_next_state
            total_steps += 1

        if ep >= dual_after/2:
            env.random_init = 0

        # dual update
        if ep >= dual_after and ep >= window_size - 1:
            success_rate = np.mean(success_tracker[ep - window_size + 1 : ep + 1])
            dual = max(0.0, dual + (target_safety - success_rate) * lambda_lr)
            
        reward_tracker.append(avg_reward)

        if (ep + 1)%print_rate == 0:
            print(f"actor: {avg_a_loss:.2f}\tcritic: {avg_c_loss:.2f}\ttemp: {alpha:.6f}\tlambda: {dual:.2f}\treward: {np.mean(reward_tracker[ep - (print_rate-1): ep-1]):.2f}\tsuccess rate: {np.sum(success_tracker[ep - (print_rate-1): ep-1])/print_rate}\tlen(buffer): {len(agent.replay_buffer)} | t = {timedelta(seconds=round(time.time() - t_start))} | ({ep+1}/{num_ep}) | {datetime.fromtimestamp(time.time(), tz=ZoneInfo("Europe/Zurich")).strftime("%H:%M:%S")}")

    agent.store_model(path)

    t_end = time.time()
    print(f"End at {datetime.fromtimestamp(t_end, tz=ZoneInfo("Europe/Zurich")).strftime("%H:%M:%S")}. \nTotal Time: {(t_end - t_start)//60:.0f}:{(t_end - t_start)%60:.1f}")
    

def test(agent: SACAgent, env: UnicycleGymEnv, path, test_eps, max_steps, deterministic_evaluation = False, random_init = 0, 
         plot = False, print_rate = 100):
    
    """
    Evaluates a Policy on a reach-avoid task.

    inputs:
        - agent:            SAC agent
        - env:              Gymnasium-like environment for reach-avoid task
        - path:             Path to neural network weights
        - test_eps:         Number of test episodes
        - max_steps:        Number of steps in one episode before termination

        - determinist_data_collection:  Whether to use mean of gaussian policy or sampling policy for trajectories
        - random_init:                  Probability to initialize not at specified inital state but at uniformally sampled point in safe set
        - plot:                         Number of trajectories to plot directly during evaluation
        - print_rate:       How often to print training updates to terminal

    outputs:
        - output_traj:      Array containing trajectories
        - success_tracker_plot: Array containing labels for trajectories (success -> 1, fail -> 0, timed out -> -1)
    """

    print("Commencing Testing...")

    agent.load_model(path)
    env.random_init = random_init
    env.goal_reward = 0

    rewards_per_episode = np.zeros(test_eps)
    success_tracker = np.zeros(test_eps)

    if plot:
        traj = np.empty((max_steps, 2, plot)) * np.nan
        success_tracker_plot = np.zeros(plot)
        output_traj = np.empty((max_steps, 5, plot)) * np.nan

        plot_every_n = test_eps//plot

    for ep in range(test_eps):

        state, _ = env.reset()
        aug_state = np.append(state, 0)

        success = False
        crash = False
        truncated = False

        if plot and ep%plot_every_n == 0:
            traj[0,:,ep//plot_every_n] = state
            output_traj[0,0:2,ep//plot_every_n] = state
            step = 0

        for _ in range(max_steps):

            action = agent.get_action(aug_state, deterministic_evaluation)
            action = np.clip(action, env.action_space.low, env.action_space.high).reshape(-1)

            next_state, reward, terminated, truncated, info = env.step(action)

            success = info["success"]

            aug_next_state = np.append(next_state, reward + aug_state[2])

            aug_state = aug_next_state

            if plot and not truncated:
                if ep%plot_every_n == 0:
                    traj[step+1, :2, ep//plot_every_n] = next_state
                    output_traj[step+1, :3, ep//plot_every_n] = aug_next_state
                    output_traj[step, 3:, ep//plot_every_n] = action
                step += 1

            rewards_per_episode[ep] += reward

            if terminated or truncated:
                break

        if success:
            success_tracker[ep] = 1
            if ep%plot_every_n == 0:
                success_tracker_plot[ep//plot_every_n] = 1

        if truncated and ep%plot_every_n == 0:
            success_tracker_plot[ep//plot_every_n] = -1

        if plot and (ep+1)%print_rate == 0:
            print(f'iteration = {ep+1}, number of successes = {np.sum(success_tracker)}')

    a = np.sum(success_tracker)/test_eps

    if plot:
        plot_traj(output_traj[:,0:2,:], success_tracker_plot, file_name, env.initial_state, env.goal_set, rectangles=env.unsafe_rect, circles=env.unsafe_circ, polygons= env.unsafe_polygons, boundary=env.state_space, params = (a, 1))
        plot_costs(rewards_per_episode, success_tracker, params=(a, 100), file_name=file_name)

    return output_traj, success_tracker_plot

def safety_type(x):
    x = float(x)
    if not (0.0 <= x <= 1.0):
        raise argparse.ArgumentError("Safety must be between 0 and 1.")
    return x

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Train policy for desired task with specified target safety using our method or classical jcc.")

    parser.add_argument("--safety", type=safety_type, default=0.5, help="Desired success rate (between 0.0, 1.0).")
    parser.add_argument("--method", choices=["classical","new"], default="new", help="Method used for training: classical JCC ('classical') or our method ('new').")
    parser.add_argument("--task", choices=["rectangle", "circles", "slit", "maze", "triangles"], default="rectangle", help="The environment layout for the navigation task ('rectangle', 'circles', 'slit', 'maze', triangles)")

    args = parser.parse_args()

    safety = args.safety
    method = args.method
    task = args.task

    ############################################################################
    # Param dicts for different tasks
    params_rectangle = {
        # Env params
        "initial state": (0.5, 1.0),
        "dist var": (0.025,0.025),
        "input var": 0.01,
        "state space": [10, 10],
        "max steps": 16,
        "random init": 0.75,
        # NN params
        "a h1": 16,
        "a h2": 16,
        "a lr": 1e-4,
        "c h1": 32,
        "c h2": 32,
        "c lr": 4e-4,
        "temp0": 0.2,
        "temp lr":3e-4,
        "compile": "max-autotune",
        # Training params
        "stage cost": 1,
        "num eps train": 200_000,
        "start steps": 20_000,
        "update freq": 2,   # was 2
        "buffer size": 1e6,
        "batch size": 256,
        "gamma": 1,
        "tau": 1e-3,
        "deterministic": True,
        # Dual params
        "dual after": 80_000,
        "lambda0": 19,
        "lambda lr": 1e-4,
        "window size": 1_000,
        # Test params
        "num eps test": 1_000,
        "num plot": 1_000,
        # World layout
        "target set": [[np.array([0, 7]), 2.2, 3]],
        "rect obstacles": [[np.array([0, 2.8]), 6.2, 7-2.8]],
        "circ obstacles": None,
        "poly obstacles": None
    }

    params_slit = {
        # Env params
        "initial state": (0.5, 1.0),
        "dist var": (0.025,0.025),
        "input var": 0.01,
        "state space": [10, 10],
        "max steps": 16,
        "random init": 0.75,
        # NN params
        "a h1": 16,
        "a h2": 16,
        "a lr": 1e-4,
        "c h1": 32,
        "c h2": 32,
        "c lr": 3e-4,
        "temp0": 0.2,
        "temp lr":3e-4,
        "compile": "max-autotune",
        # Training params
        "stage cost": 1,
        "num eps train": 200_000,
        "start steps": 20_000,
        "update freq": 2,
        "buffer size": 1e6,
        "batch size": 256,
        "gamma": 1,
        "tau": 1e-3,
        "deterministic": True,
        # Dual params
        "dual after": 80_000,
        "lambda0": 19,
        "lambda lr": 1e-4,
        "window size": 1_000,
        # Test params
        "num eps test": 1_000,
        "num plot": 1_000,
        # World layout
        "target set": [[np.array([0, 7]), 2.2, 3]],
        "rect obstacles": [[np.array([0, 2.8]), 4.2, 7-2.8],
                           [np.array([4.8, 2.8]), 1.4, 7-2.8]],
        "circ obstacles": None,
        "poly obstacles": None
    }

    params_triangles = {
        # Env params
        "initial state": (0.5, 1.0),
        "dist var": (0.025,0.025),
        "input var": 0.01,
        "state space": [10, 10],
        "max steps": 16,
        "random init": 0.9,
        # NN params
        "a h1": 8,
        "a h2": 8,
        "a lr": 1e-4,
        "c h1": 16,
        "c h2": 16,
        "c lr": 3e-4,
        "temp0": 0.2,
        "temp lr":3e-4,
        "compile": "max-autotune",
        # Training params
        "stage cost": 1,
        "num eps train": 250_000,
        "start steps": 20_000,
        "update freq": 1,   # was 2
        "buffer size": 1e6,
        "batch size": 256,
        "gamma": 1,
        "tau": 1e-3,
        "deterministic": True,
        # Dual params
        "dual after": 10_000,
        "lambda0": 20,
        "lambda lr": 1e-4,
        "window size": 2_000,
        # Test params
        "num eps test": 1_000,
        "num plot": 1_000,
        # World layout
        "target set": [[np.array([0, 8.4]), 1.2, 1.6]],
        "rect obstacles": None,
        "circ obstacles": None,
        "poly obstacles":[np.array([[10,0],[10,4.8],[9.6,4.8],[4.4,2.4],[4.4,2.2],[9.2,0]]),
                          np.array([[10,10],[8,10],[4.6,8.4],[4.6,8.2],[9.8,5.8],[10,5.8]]),
                          np.array([[0,2.6],[0.4,2.6],[5.6,5],[5.6,5.2],[0.4,7.6],[0,7.6]])]
    }

    params_circles = {
        # Env params
        "initial state": (0.5, 1.0),
        "dist var": (0.025,0.025),
        "input var": 0.01,
        "state space": [10, 10],
        "max steps": 16,
        "random init": 0.9,
        # NN params
        "a h1": 8,
        "a h2": 8,
        "a lr": 2e-4,
        "c h1": 16,
        "c h2": 16,
        "c lr": 6e-4,
        "temp0": 0.2,
        "temp lr":3e-4,
        "compile": "max-autotune",
        # Training params
        "stage cost": 1,
        "num eps train": 250_000,
        "start steps": 20_000,
        "update freq": 2,   # was 2
        "buffer size": 1e6,
        "batch size": 256,
        "gamma": 1,
        "tau": 1e-3,
        "deterministic": True,
        # Dual params
        "dual after": 100_000,
        "lambda0": 15,
        "lambda lr": 1e-4,
        "window size": 1_000,
        # Test params
        "num eps test": 1_000,
        "num plot": 1_000,
        # World layout
        "target set": [[np.array([8.8, 9.2]), 1.2, 0.8],
                       [np.array([8.2, 0.6]), 1, 0.8]],
        "rect obstacles": None,
        "circ obstacles": [[np.array([1.5, 10-0.04545455]), 0.49842728],
                           [np.array([7.3, 10-0.24303797]), 0.67332849],
                           [np.array([3.2, 10-2.1]), 1.04641579],
                           [np.array([9.43333333, 10-2.5]), 0.73786479],
                           [np.array([6.7, 10-2.7]), 1.00925301],
                           [np.array([0.10805369, 10-3.9]), 0.60838473],
                           [np.array([4.9, 10-5.5]), 1.00925301],
                           [np.array([8.9, 10-5.5]), 1.00925301],
                           [np.array([1.7, 10-7.4]), 0.94406974],
                           [np.array([6.5, 10-8.2]), 0.94406974],
                           [np.array([3.9, 10-9.43333333]), 0.73786479]],
        "poly obstacles": None
    }

    if task == "rectangle":
        p = params_rectangle
    elif task == "circles":
        p = params_circles
    elif task == "slit":
        p = params_slit
    elif task == "triangles":
        p = params_triangles

    file_name = f"models/{task}_{method}_{safety}_2"

    
    env = UnicycleGymEnv(random_init=p["random init"], goal_reward=1, stage_cost=p["stage cost"], initial_state=p["initial state"],
                         state_space=p["state space"], max_steps=p["max steps"], goal_set=p["target set"], dist_var = p["dist var"], input_var=p["input var"])

    scale = torch.tensor((env.action_space.high - env.action_space.low)/2, dtype=torch.float32)
    offset = torch.tensor((env.action_space.high + env.action_space.low)/2, dtype=torch.float32)

    agent = SACAgent(env.observation_space.shape[-1]+1, env.action_space.shape[-1], scale, offset, p["batch size"], alpha_0 = p["temp0"],
                     a_h1 = p["a h1"], a_h2 = p["a h2"], c_h1 = p["c h1"], c_h2 = p["c h2"], a_lr = p["a lr"], c_lr = p["c lr"], alpha_lr=p["temp lr"], tau = p["tau"], discount = p["gamma"],
                     compile = p["compile"])

    if p["rect obstacles"] != None:
        for rect in p["rect obstacles"]:
            env.add_rect_set(rect[0], rect[1], rect[2])

    if p["circ obstacles"] != None:
        for circ in p["circ obstacles"]:
            env.add_circ_set(circ[0], circ[1])

    if p["poly obstacles"] != None:
        for polygon in p["poly obstacles"]:
            env.add_poly_set(polygon)

    try:
        train(agent, env, max_steps=p["max steps"], num_ep=p["num eps train"], start_steps=p["start steps"], print_rate=5_000, path = file_name,
        lambda_0=p["lambda0"], lambda_lr=p["lambda lr"], target_safety=safety, window_size=p["window size"], updates_per_step=p["update freq"], dual_after=p["dual after"], method = method, deterministic_data_collection=p["deterministic"])
        
        trajs, successes = test(agent, env, path = file_name, test_eps=p["num eps test"], max_steps=p["max steps"], plot = p["num plot"], print_rate=1_000, random_init=0, deterministic_evaluation=p["deterministic"])
        np.save(file_name + "traj", trajs)
        np.save(file_name + "success", successes)
    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted. Saving model weights...")
        agent.store_model(file_name + "_temp")
        print("[INFO] Weights saved successfully. Exiting.") 

    env.close()
