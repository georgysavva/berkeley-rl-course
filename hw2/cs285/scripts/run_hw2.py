import functools
import time

import gym
import numpy as np
import optuna
import torch

import wandb
from cs285.agents.pg_agent import PGAgent
from cs285.infrastructure import pytorch_util as ptu
from cs285.infrastructure import utils
from cs285.infrastructure.logger import prepare_trajs_as_videos

MAX_NVIDEO = 2


def run_training_loop(config, training_callback):
    wandb_project = config.pop("wandb_project")
    wandb_group = config.pop("study_name", None)
    wandb.init(
        project=wandb_project,
        config=config,
        group=wandb_group,
        reinit=True,
    )
    # set random seeds
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    ptu.init_gpu(use_gpu=True, gpu_id=config.which_gpu)

    # make the gym environment
    env = gym.make(config.env_name, render_mode=None)
    discrete = isinstance(env.action_space, gym.spaces.Discrete)

    # # add action noise, if needed
    # if args.action_noise_std > 0:
    #     assert not discrete, f"Cannot use --action_noise_std for discrete environment {args.env_name}"
    #     env = ActionNoiseWrapper(env, args.seed, args.action_noise_std)

    max_ep_len = config.ep_len or env.spec.max_episode_steps
    assert max_ep_len is not None, "env must have max episode length"
    print(f"Will use a maximum of {max_ep_len} steps")
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.n if discrete else env.action_space.shape[0]

    # simulation timestep, will be used for video saving
    if hasattr(env, "model"):
        fps = 1 / env.model.opt.timestep
    else:
        fps = env.env.metadata["render_fps"]

    # initialize agent
    agent = PGAgent(
        ob_dim,
        ac_dim,
        discrete,
        n_layers=config.n_layers,
        layer_size=config.layer_size,
        gamma=config.discount,
        learning_rate=config.learning_rate,
        use_baseline=config.use_baseline,
        use_reward_to_go=config.use_reward_to_go,
        normalize_advantages=config.normalize_advantages,
        baseline_learning_rate=config.baseline_learning_rate,
        baseline_gradient_steps=config.baseline_gradient_steps,
        gae_lambda=config.gae_lambda,
    )

    total_envsteps = 0
    start_time = time.time()
    eval_reward = 0.0
    for itr in range(config.n_iter):
        print(f"\n********** Iteration {itr} ************")
        # sample `args.batch_size` transitions using utils.sample_trajectories
        # make sure to use `max_ep_len`
        trajs, envsteps_this_batch = utils.sample_trajectories(
            env, agent.actor, config.batch_size, max_ep_len
        )
        total_envsteps += envsteps_this_batch

        # trajs should be a list of dictionaries of NumPy arrays, where each dictionary corresponds to a trajectory.
        # this line converts this into a single dictionary of lists of NumPy arrays.
        trajs_dict = {k: [traj[k] for traj in trajs] for k in trajs[0]}

        # train the agent using the sampled trajectories and the agent's update function
        train_info: dict = agent.update(
            trajs_dict["observation"],
            trajs_dict["action"],
            trajs_dict["reward"],
            trajs_dict["terminal"],
        )

        if itr % config.scalar_log_freq == 0:
            # save eval metrics
            print("\nCollecting data for eval...")
            trajs_groups = [("Train", trajs)]
            eval_trajs, _ = utils.sample_trajectories(
                env,
                agent.actor,
                config.eval_batch_size,
                max_ep_len,
                deterministic_predict=config.deterministic_eval,
            )
            trajs_groups.append(("Eval", eval_trajs))
            eval_reward = utils.compute_average_return(eval_trajs)
            logs = utils.compute_metrics(trajs_groups)
            # compute additional metrics
            logs.update(train_info)
            logs["Train_EnvstepsSoFar"] = total_envsteps
            logs["TimeSinceStart"] = time.time() - start_time
            if itr == 0:
                logs["Initial_DataCollection_AverageReturn"] = logs[
                    "Train_AverageReturn"
                ]
            # perform the logging

            for key, value in logs.items():
                print("{} : {}".format(key, value))
            wandb.log(logs, step=itr)
            print("Done logging...\n\n")

        if config.video_log_freq != -1 and itr % config.video_log_freq == 0:
            print("\nCollecting video rollouts...")
            eval_video_trajs = utils.sample_n_trajectories(
                env,
                agent.actor,
                MAX_NVIDEO,
                max_ep_len,
                render=True,
                deterministic_predict=config.deterministic_eval,
            )
            videos = prepare_trajs_as_videos(eval_video_trajs, MAX_NVIDEO)
            wandb.log({"eval_rollouts": wandb.Video(videos, fps=fps)})
            training_callback(itr, eval_reward)

    return eval_reward


def objective(config, trial: optuna.Trial):
    hyper_params = get_hyper_parameters(trial)
    config = {
        **config,
        **hyper_params,
        **trial.study.user_attrs,
        "trial_number": trial.number,
    }
    config = utils.AttrDict(config)

    def training_callback(iteration, eval_return):
        trial.report(eval_return, step=iteration)
        if trial.should_prune():
            wandb.run.summary["state"] = "pruned"
            wandb.finish(quiet=True)
            raise optuna.TrialPruned()

    result = run_training_loop(config, training_callback)
    wandb.finish(quiet=True)
    return result


def get_hyper_parameters(trial: optuna.Trial):
    hyperparams = {
        "n_layers": trial.suggest_int("n_layers", 1, 3),
        "layer_size": trial.suggest_int("layer_size", 32, 256),
        "discount": trial.suggest_float("discount", 0.9, 0.99, step=0.01),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        "use_baseline": trial.suggest_categorical("use_baseline", [True]),
        "use_reward_to_go": trial.suggest_categorical("use_reward_to_go", [True]),
        "normalize_advantages": trial.suggest_categorical(
            "normalize_advantages", [True]
        ),
        "baseline_learning_rate": trial.suggest_float(
            "baseline_learning_rate", 1e-4, 1e-2, log=True
        ),
        "baseline_gradient_steps": trial.suggest_categorical(
            "baseline_gradient_steps", [1, 5, 10, 25, 50]
        ),
        "gae_lambda": trial.suggest_categorical(
            "gae_lambda", [0, 0.95, 0.97, 0.98, 0.99, 1]
        ),
        "batch_size": trial.suggest_categorical(
            "batch_size", [1000, 5000, 25000, 50000, 100000]
        ),
        "deterministic_eval": trial.suggest_categorical("deterministic", [False, True]),
    }
    return hyperparams


def main():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--which_gpu", "-gpu_id", default=0)
    parser.add_argument("--wandb_project", type=str, default="berkeleyrl-hw2")

    parser.add_argument("--study_name", type=str)
    parser.add_argument(
        "--study_storage", type=str, default="sqlite:///optuna_studies.db"
    )
    parser.add_argument("--study_sampler", type=str, default="RandomSampler")
    parser.add_argument("--study_pruner", type=str, default="NopPruner")
    parser.add_argument("--n_trials", type=int, default=100)

    args = parser.parse_args()
    study = optuna.load_study(
        study_name=args.study_name,
        sampler=getattr(optuna.samplers, args.study_sampler)(),
        pruner=getattr(optuna.pruners, args.study_pruner)(),
        storage=args.study_storage,
    )
    config = vars(args)
    config.pop("study_storage")
    study.optimize(functools.partial(objective, config), n_trials=args.n_trials)  # type: ignore


if __name__ == "__main__":
    main()
