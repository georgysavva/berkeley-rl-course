import copy
from collections import OrderedDict
from typing import Dict, List, Tuple

import cv2
import gym
import numpy as np

from cs285.infrastructure import pytorch_util as ptu
from cs285.networks.policies import MLPPolicy

############################################
############################################


def sample_trajectory(
    env: gym.Env,
    policy: MLPPolicy,
    max_length: int,
    render: bool = False,
    deterministic_predict: bool = False,
) -> Dict[str, np.ndarray]:
    """Sample a rollout in the environment from a policy."""
    ob = env.reset()
    obs, acs, rewards, next_obs, terminals, image_obs = [], [], [], [], [], []
    steps = 0
    while True:
        # render an image
        if render:
            if hasattr(env, "sim"):
                img = env.sim.render(camera_name="track", height=500, width=500)[::-1]
            else:
                img = env.render(mode="single_rgb_array")
            image_obs.append(
                cv2.resize(img, dsize=(250, 250), interpolation=cv2.INTER_CUBIC)
            )

        ac: np.ndarray = policy.get_action(ob, deterministic=deterministic_predict)

        next_ob, rew, done, _ = env.step(ac)

        steps += 1
        rollout_done: bool = steps >= max_length or done

        # record result of taking that action
        obs.append(ob)
        acs.append(ac)
        rewards.append(rew)
        next_obs.append(next_ob)
        terminals.append(rollout_done)

        ob = next_ob  # jump to next timestep

        # end the rollout if the rollout ended
        if rollout_done:
            break

    return {
        "observation": np.array(obs, dtype=np.float32),
        "image_obs": np.array(image_obs, dtype=np.uint8),
        "reward": np.array(rewards, dtype=np.float32),
        "action": np.array(acs, dtype=np.float32),
        "next_observation": np.array(next_obs, dtype=np.float32),
        "terminal": np.array(terminals, dtype=np.float32),
    }


def sample_trajectories(
    env: gym.Env,
    policy: MLPPolicy,
    min_timesteps_per_batch: int,
    max_length: int,
    render: bool = False,
    deterministic_predict: bool = False,
) -> Tuple[List[Dict[str, np.ndarray]], int]:
    """Collect rollouts using policy until we have collected min_timesteps_per_batch steps."""
    timesteps_this_batch = 0
    trajs = []
    while timesteps_this_batch < min_timesteps_per_batch:
        # collect rollout
        traj = sample_trajectory(
            env, policy, max_length, render, deterministic_predict=deterministic_predict
        )
        trajs.append(traj)

        # count steps
        timesteps_this_batch += get_traj_length(traj)
    return trajs, timesteps_this_batch


def sample_n_trajectories(
    env: gym.Env,
    policy: MLPPolicy,
    ntraj: int,
    max_length: int,
    render: bool = False,
    deterministic_predict: bool = False,
):
    """Collect ntraj rollouts."""
    trajs = []
    for _ in range(ntraj):
        # collect rollout
        traj = sample_trajectory(
            env, policy, max_length, render, deterministic_predict=deterministic_predict
        )
        trajs.append(traj)
    return trajs


def compute_metrics(trajs_groups):
    """Compute metrics for logging."""

    # returns, for logging

    # decide what to log
    logs = OrderedDict()
    for name, trajs in trajs_groups:
        returns = [traj["reward"].sum() for traj in trajs]
        ep_lens = [len(traj["reward"]) for traj in trajs]
        logs[name + "_AverageReturn"] = np.mean(returns)
        logs[name + "_StdReturn"] = np.std(returns)
        logs[name + "_MaxReturn"] = np.max(returns)
        logs[name + "_MinReturn"] = np.min(returns)
        logs[name + "_AverageEpLen"] = np.mean(ep_lens)

    return logs


def convert_listofrollouts(trajs):
    """
    Take a list of rollout dictionaries and return separate arrays, where each array is a concatenation of that array
    from across the rollouts.
    """
    observations = np.concatenate([traj["observation"] for traj in trajs])
    actions = np.concatenate([traj["action"] for traj in trajs])
    next_observations = np.concatenate([traj["next_observation"] for traj in trajs])
    terminals = np.concatenate([traj["terminal"] for traj in trajs])
    concatenated_rewards = np.concatenate([traj["reward"] for traj in trajs])
    unconcatenated_rewards = [traj["reward"] for traj in trajs]
    return (
        observations,
        actions,
        next_observations,
        terminals,
        concatenated_rewards,
        unconcatenated_rewards,
    )


def get_traj_length(traj):
    return len(traj["reward"])
