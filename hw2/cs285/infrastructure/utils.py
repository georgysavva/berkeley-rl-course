from collections import OrderedDict
from typing import Dict, List, Tuple

import cv2
import gym
import gym.vector
import numpy as np

from cs285.infrastructure import pytorch_util as ptu
from cs285.networks.critics import ValueCritic
from cs285.networks.policies import MLPPolicy

IMAGE_SIZE = 250


def sample_trajectories_vectorized(
    env: gym.vector.VectorEnv,
    policy: MLPPolicy,
    critic: ValueCritic,
    steps: int,
    deterministic_predict: bool = False,
):
    obs = np.zeros(
        (steps, env.num_envs, *env.single_observation_space.shape), dtype=np.float32
    )
    acs = np.zeros(
        (steps, env.num_envs, *env.single_action_space.shape), dtype=np.float32
    )
    rewards = np.zeros((steps, env.num_envs), dtype=np.float32)
    value_bootstraps = np.zeros((steps, env.num_envs), dtype=np.float32)
    dones = np.zeros((steps, env.num_envs), dtype=np.float32)
    ob = env.reset()
    for step in range(steps):

        ac: np.ndarray = policy.get_action(ob, deterministic=deterministic_predict)

        next_ob, rew, done, info = env.step(ac)

        # record result of taking that action
        obs[step] = ob
        acs[step] = ac
        rewards[step] = rew
        dones[step] = done
        for i in range(env.num_envs):
            truncated = done[i] and info[i].get("TimeLimit.truncated", False)
            if step < steps - 1:
                if truncated:
                    value_bootstraps[step, i] = critic.predict(next_ob[i])
            else:
                if truncated or not done[i]:
                    value_bootstraps[step, i] = critic.predict(next_ob[i])

        ob = next_ob  # jump to next timestep

        # end the rollout if the rollout ended

    return {
        "observations": obs,
        "actions": acs,
        "rewards": rewards,
        "value_bootstraps": value_bootstraps,
        "dones": dones,
    }


def sample_n_trajectories_vectorized_for_eval(
    env: gym.vector.VectorEnv,
    policy: MLPPolicy,
    ntraj: int,
    deterministic_predict: bool = False,
):
    trajs = [
        [{"image_obs": [], "rewards": []} for _ in range(ntraj)]
        for _ in range(env.num_envs)
    ]
    env_counts = np.zeros(env.num_envs, dtype=np.int32)
    ob = env.reset()
    while np.any(env_counts < ntraj):
        # render an image
        if hasattr(env, "sim"):
            img = env.sim.render(camera_name="track", height=500, width=500)[::-1]
        else:
            img = env.render(mode="single_rgb_array")

        ac: np.ndarray = policy.get_action(ob, deterministic=deterministic_predict)

        next_ob, rew, done, _ = env.step(ac)

        for i in range(env.num_envs):
            if env_counts[i] < ntraj:
                trajs[i][env_counts[i]]["rewards"].append(rew[i])
                trajs[i][env_counts[i]]["image_obs"].append(
                    cv2.resize(
                        img,
                        dsize=(IMAGE_SIZE, IMAGE_SIZE),
                        interpolation=cv2.INTER_CUBIC,
                    )
                )
                env_counts[i] += done[i]

        ob = next_ob  # jump to next timestep

        # end the rollout if the rollout ended
    trajs = [traj for env_trajs in trajs for traj in env_trajs]
    return trajs


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


def compute_metrics(eval_trajs):
    """Compute metrics for logging."""

    # returns, for logging

    # decide what to log
    logs = OrderedDict()
    returns = [traj["reward"].sum() for traj in eval_trajs]
    ep_lens = [len(traj["reward"]) for traj in eval_trajs]
    name = "Eval"
    logs[name + "/AverageReturn"] = np.mean(returns)
    logs[name + "/StdReturn"] = np.std(returns)
    logs[name + "/MaxReturn"] = np.max(returns)
    logs[name + "/MinReturn"] = np.min(returns)
    logs[name + "/AverageEpLen"] = np.mean(ep_lens)

    return logs


def compute_average_return(trajs):
    """Compute average return."""
    returns = [traj["reward"].sum() for traj in trajs]
    return np.mean(returns)


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


class AttrDict(dict):
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"'AttrDict' object has no attribute '{item}'")


if __name__ == "__main__":
    env = gym.make("Humanoid-v4", render_mode=None)
    debug_sample_trajectory(env)
