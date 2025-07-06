import os

import numpy as np
from stable_baselines3 import PPO
import re
import argparse
import scipy.ndimage
from collections import defaultdict
import matplotlib.pyplot as pp
import cv2
import torch

from uitb.utils.logger import StateLogger, ActionLogger
from uitb.simulator import Simulator

from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter

from concurrent.futures import ProcessPoolExecutor
from functools import partial
import multiprocessing as mp

import gc

def _parallel_eval_wrapper(args):
    """Unpacks arguments for parallel processing."""
    print(f"📦 Worker received args: {args[:2]}")  # Only print alpha, beta
    partial_func, alpha, beta = args
    result = partial_func(alpha, beta)
    print(f"✅ Worker finished: {result}")
    return partial_func(alpha, beta)

def flatten_params(state_dict, keys):
    """Flatten only selected parameters (keys) from the state_dict."""
    return np.concatenate([
        state_dict[k].detach().cpu().numpy().flatten() for k in keys
    ])

def update_params(state_dict, keys, flat_vector):
    """Update the state_dict with new flattened values only for selected keys."""
    offset = 0
    for k in keys:
        shape = state_dict[k].shape
        numel = np.prod(shape)
        new_vals = flat_vector[offset:offset + numel].reshape(shape)
        state_dict[k] = torch.tensor(new_vals, dtype=state_dict[k].dtype, device=state_dict[k].device)
        offset += numel
    return state_dict

def evaluate_policy_return(model, env, num_episodes=1):
    total_reward = 0
    for _ in range(num_episodes):
        reset_result = env.reset()
        obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        done = False
        episode_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            step_result = env.step(action)
            if len(step_result) == 5:
                obs, reward, terminated, truncated, _ = step_result
                done = terminated or truncated
            else:
                obs, reward, done, _ = step_result
            episode_reward += reward
        total_reward += episode_reward
    return total_reward / num_episodes


def _evaluate_point(alpha, beta, model_path, env_config, lambda_scale=0.01):
    env = None
    model = None
    try:
        # Set up environment
        config = dict(env_config)
        run_params = config.get("run_parameters", {}).copy()
        run_params["reward_parameters"] = {'alpha': alpha, 'beta': beta}
        config["run_parameters"] = run_params

        env = Simulator.get(**config)
        model = PPO.load(model_path, device='cpu')

        # Evaluate
        reward = evaluate_policy_return(model, env)

        # Load weights and compute penalty
        weights = model.policy.state_dict()["action_net.weight"]
        penalty = lambda_scale * torch.sum(torch.abs(weights)).item() 

        # Subtract penalty from reward
        penalized_reward = reward - penalty

        return (alpha, beta, penalized_reward)

    finally:
        if env is not None:
            try:
                env.close()
            except Exception as e:
                print(f"[WARN] Failed to close env: {e}")
        del env
        del model
        gc.collect()
        torch.cuda.empty_cache() 


def visualize_reward_landscape_2d_parallel(model, env, state_dict, keys, model_path, env_config,
                                           alpha_range=np.linspace(-3.0, 3.0, 10), num_episodes=5):

    original_vector = flatten_params(state_dict, keys)

    # Get PCA directions
    perturbations = [original_vector + 0.01 * np.random.randn(*original_vector.shape) for _ in range(100)]
    pca = PCA(n_components=2)
    pca.fit(perturbations)
    dir1, dir2 = pca.components_

    grid_points = [(alpha, beta) for alpha in alpha_range for beta in alpha_range]
    '''
    partial_eval = partial(_evaluate_point, original_vector=original_vector,
                           dir1=dir1, dir2=dir2,
                           state_dict=state_dict,
                           keys=keys,
                           model_path=model_path,
                           num_episodes=num_episodes,
                           env_config=env_config)
    '''
    partial_eval = partial(_evaluate_point,
                       model_path=model_path,
                       env_config=env_config)

    print(f"🌍 Evaluating {len(grid_points)} points in parallel...")

    results = []
    with ProcessPoolExecutor(max_workers=16) as executor:
        args_list = [(partial_eval, alpha, beta) for alpha, beta in grid_points]
        for res in executor.map(_parallel_eval_wrapper, args_list):
            results.append(res)

    # Fill result matrix
    rewards = np.zeros((len(alpha_range), len(alpha_range)))
    alpha_to_idx = {round(a, 5): i for i, a in enumerate(alpha_range)}
    for alpha, beta, reward in results:
        i, j = alpha_to_idx[round(alpha, 5)], alpha_to_idx[round(beta, 5)]
        rewards[j, i] = reward

    # Smooth
    rewards = gaussian_filter(rewards, sigma=1.0)

    # Plot
    X, Y = np.meshgrid(alpha_range, alpha_range)
    pp.figure(figsize=(6, 5))
    levels = np.linspace(np.min(rewards), np.max(rewards), 25)
    contour = pp.contour(X, Y, rewards, levels=levels, cmap='viridis', linewidths=1.5)
    pp.clabel(contour, inline=True, fontsize=8, fmt="%.2f")
    pp.imshow(rewards, extent=[alpha_range[0], alpha_range[-1], alpha_range[0], alpha_range[-1]],
              origin='lower', cmap='viridis', alpha=0.6)
    pp.colorbar(label='Average Reward')
    pp.xlabel("PCA Direction 1")
    pp.ylabel("PCA Direction 2")
    pp.title("2D Reward Landscape (Parallel + Smoothed)")
    pp.gca().set_aspect('equal')
    pp.tight_layout()
    pp.savefig("reward_landscape_2d_parallel.png", dpi=300)
    pp.close()


def visualize_reward_landscape_2d(model, env, state_dict, keys, alpha_range=np.linspace(-1.0, 1.0, 41)):
    original_vector = flatten_params(state_dict, keys)

    perturbations = [original_vector + 0.01 * np.random.randn(*original_vector.shape) for _ in range(100)]
    pca = PCA(n_components=2)
    pca.fit(perturbations)
    dir1, dir2 = pca.components_

    rewards = np.zeros((len(alpha_range), len(alpha_range)))

    for i, alpha in enumerate(alpha_range):
        for j, beta in enumerate(alpha_range):
            new_vector = original_vector + alpha * dir1 + beta * dir2
            new_sd = update_params(state_dict.copy(), keys, new_vector)
            model.policy.load_state_dict(new_sd)
            avg_reward = evaluate_policy_return(model, env)
            rewards[j, i] = avg_reward  # match axis orientation

    # Smooth
    rewards = gaussian_filter(rewards, sigma=1.0)

    # Plot
    X, Y = np.meshgrid(alpha_range, alpha_range)
    pp.figure(figsize=(6, 5))
    levels = np.linspace(np.min(rewards), np.max(rewards), 25)
    cp = pp.contour(X, Y, rewards, levels=levels, cmap='viridis', linewidths=1.5)
    pp.clabel(cp, fmt="%.2f", inline=True, fontsize=8)
    pp.imshow(rewards, extent=[alpha_range[0], alpha_range[-1], alpha_range[0], alpha_range[-1]],
              origin='lower', cmap='viridis', alpha=0.6)
    pp.colorbar(label='Average Reward')
    pp.xlabel("PCA Direction 1")
    pp.ylabel("PCA Direction 2")
    pp.title("2D Reward Landscape")
    pp.gca().set_aspect('equal')
    pp.tight_layout()
    pp.savefig("reward_landscape_2d.png", dpi=300)
    pp.close()

def worker(gpu_id, task, model_path):
    # Limit this process to one GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # Now any torch.cuda calls will see only this GPU
    device = torch.device('cuda:0')  # '0' because CUDA_VISIBLE_DEVICES remaps visible devices
    
    # Load your model on this device
    model = PPO.load(model_path, device=device)
    
    # Run your evaluation on this GPU
    result = model(task.to(device))
    return result.cpu()

def main(model_path):
    tasks = [...]  # your 441 tasks
    num_gpus = torch.cuda.device_count()

    with ProcessPoolExecutor(max_workers=num_gpus) as executor:
        futures = []
        for i, task in enumerate(tasks):
            gpu_id = i % num_gpus  # distribute tasks round-robin on GPUs
            futures.append(executor.submit(worker, gpu_id, task, model_path))
        
        results = [f.result() for f in futures]

    
def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

if __name__ == "__main__":

    
    
    mp.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser(description='Evaluate a policy.')
    parser.add_argument('simulator_folder', type=str,
                        help='the simulation folder')
    parser.add_argument('--action_sample_freq', type=float, default=20,
                        help='action sample frequency (how many times per second actions are sampled from policy, default: 20)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='filename of a specific checkpoint (default: None, latest checkpoint is used)')
    parser.add_argument('--num_episodes', type=int, default=10,
                        help='how many episodes are evaluated (default: 10)')
    parser.add_argument('--uncloned', dest="cloned", action='store_false', help='use source code instead of files from cloned simulator module')
    parser.add_argument('--app_condition', type=str, default=None,
                        help="can be used to override the 'condition' argument passed to a Unity app")
    parser.add_argument('--record', action='store_true', help='enable recording')
    parser.add_argument('--out_file', type=str, default='evaluate.mp4',
                        help='output file for recording if recording is enabled (default: ./evaluate.mp4)')
    parser.add_argument('--logging', action='store_true', help='enable logging')
    parser.add_argument('--state_log_file', default='state_log',
                        help='output file for state log if logging is enabled (default: ./state_log)')
    parser.add_argument('--action_log_file', default='action_log',
                        help='output file for action log if logging is enabled (default: ./action_log)')
    parser.add_argument('--heatmap', default=False, action='store_true',help='Create heatmap')
    parser.add_argument('--reward_landscape', default=False, action='store_true',
                        help='Create reward landscape')
    
    args = parser.parse_args()

    # Define directories
    checkpoint_dir = os.path.join(args.simulator_folder, 'checkpoints')
    evaluate_dir = os.path.join(args.simulator_folder, 'evaluate')

    # Make sure output dir exists
    os.makedirs(evaluate_dir, exist_ok=True)

    # Override run parameters
    run_params = dict()
    run_params["action_sample_freq"] = args.action_sample_freq
    run_params["evaluate"] = True

    run_params["unity_record_gameplay"] = args.record
    run_params["unity_logging"] = True
    run_params["unity_output_folder"] = evaluate_dir
    if args.app_condition is not None:
        run_params["app_args"] = ['-condition', args.app_condition]

    render_mode_perception = "separate" if run_params["unity_record_gameplay"] else "embed"

    deterministic = False

    simulator = Simulator.get(args.simulator_folder, render_mode="rgb_array_list", render_mode_perception=render_mode_perception, run_parameters=run_params, use_cloned=args.cloned)

    lamda = simulator.config["rl"]["reg"]["l1"]

    print(f"run parameters are: {simulator.run_parameters}")

    _policy_loaded = False
    if args.checkpoint is not None:
        model_file = args.checkpoint
        _policy_loaded = True
    else:
        try:
            files = natural_sort(os.listdir(checkpoint_dir))
            model_file = files[-1]
            _policy_loaded = True
        except (FileNotFoundError, IndexError):
            print("No checkpoint found. Will continue evaluation with randomly sampled controls.")

    if _policy_loaded:
        print(f'Loading model: {os.path.join(checkpoint_dir, model_file)}\n')
        model = PPO.load(os.path.join(checkpoint_dir, model_file))
        simulator.update_callbacks(model.num_timesteps)

    if args.logging:
        state_logger = StateLogger(args.num_episodes, keys=simulator.get_state().keys())
        action_logger = ActionLogger(args.num_episodes)

    
    
    params = model.policy.state_dict()

    key = "action_net.weight"

    np_params = {k: v.cpu().numpy() for k, v in params.items()}

    if key in np_params:
        weights = np_params[key]
        print(f"🔍 Values for '{key}':")
        print(weights)
        print(f"Shape: {weights.shape}")
    else:
        print(f"❌ Key '{key}' not found in model parameters.")

    pp.figure(figsize=(10, 6))
    pp.imshow(weights, cmap='coolwarm', aspect='auto')
    pp.colorbar(label='Weight Value')
    pp.title(f"Heatmap of Weights: {key}")
    pp.xlabel("Input Features")
    pp.ylabel("Output Neurons")
    pp.tight_layout()
    pp.savefig("heatmap_policy_net_0_weight.png", dpi=300)

    keys = ["action_net.weight"]

    flat_vector = flatten_params(params, keys)
    direction = np.random.randn(*flat_vector.shape)

    # Save model for multiprocessing use
    model_path = os.path.join(evaluate_dir, "temp_model.zip")
    model.save(model_path)

    # Save model for multiprocessing use
    model_path = os.path.join(evaluate_dir, "temp_model.zip")
    model.save(model_path)

    params = model.policy.state_dict()

    key = "action_net.weight"
    np_params = {k: v.cpu().numpy() for k, v in params.items()}


    if args.heatmap:
        if key in np_params:
            weights = np_params[key]
            print(f"🔍 Values for '{key}':")
            print(weights)
            print(f"Shape: {weights.shape}")
    
            pp.figure(figsize=(10, 6))
            pp.imshow(weights, cmap='coolwarm', aspect='auto')
            pp.colorbar(label='Weight Value')
            pp.title(f"Heatmap of Weights: {key}")
            pp.xlabel("Input Features")
            pp.ylabel("Output Neurons")
            pp.tight_layout()
            pp.savefig("heatmap_policy_net_0_weight.png", dpi=300)
        else:
            print(f"❌ Key '{key}' not found in model parameters.")

    # PCA keys to visualize
    keys = ["action_net.weight"]

    # Environment config for Simulator.get()
    env_config = {
        "simulator_folder": args.simulator_folder,
        "render_mode": "rgb_array_list",
        "render_mode_perception": render_mode_perception,
        "run_parameters": run_params,
        "use_cloned": args.cloned
    }

    if args.reward_landscape:
        
        visualize_reward_landscape_2d_parallel(
            model=None,
            env=None,
            state_dict=params,
            keys=keys,
            model_path=model_path,
            env_config=env_config,
            num_episodes=args.num_episodes
        )

    simulator.close()