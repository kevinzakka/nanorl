import time
from pathlib import Path
from typing import Any, Callable

import dm_env
import tqdm
import wandb
from nanorl import agent, replay, specs

from nanorl.infra import Experiment, utils


EnvFn = Callable[[], dm_env.Environment]
AgentFn = Callable[[dm_env.Environment], agent.Agent]
ReplayFn = Callable[[dm_env.Environment], replay.ReplayBuffer]
LoggerFn = Callable[[], Any]


def train_loop(
    experiment: Experiment,
    env_fn: EnvFn,
    agent_fn: AgentFn,
    replay_fn: ReplayFn,
    logger_fn: LoggerFn,
    max_steps: int,
    warmstart_steps: int,
    log_interval: int,
    checkpoint_interval: int,
    resets: bool,
    reset_interval: int,
    tqdm_bar: bool,
) -> None:
    env = env_fn()
    agent = agent_fn(env)
    replay_buffer = replay_fn(env)

    run = logger_fn()

    spec = specs.EnvironmentSpec.make(env)
    timestep = env.reset()
    replay_buffer.insert(timestep, None)

    # Log the observation and action dimensions.
    run.config.update(
        {
            "observation_dim": timestep.observation.shape[0],
            "action_dim": spec.action.shape[0],
        }
    )

    if hasattr(env, "random_state"):
        random_state = env.random_state
    else:
        random_state = env.task.random

    start_time = time.time()
    for i in tqdm.tqdm(range(1, max_steps + 1), disable=not tqdm_bar):
        if i < warmstart_steps:
            action = spec.sample_action(random_state=random_state)
        else:
            agent, action = agent.sample_actions(timestep.observation)

        timestep = env.step(action)
        replay_buffer.insert(timestep, action)

        if timestep.last():
            run.log(utils.prefix_dict("train", env.get_statistics()), step=i)
            timestep = env.reset()
            replay_buffer.insert(timestep, None)

        if i >= warmstart_steps:
            if replay_buffer.is_ready():
                transitions = replay_buffer.sample()
                agent, metrics = agent.update(transitions)
                if i % log_interval == 0:
                    run.log(utils.prefix_dict("train", metrics), step=i)

        if checkpoint_interval >= 0 and i % checkpoint_interval == 0:
            experiment.save_checkpoint(agent, step=i)

        if i % log_interval == 0:
            run.log({"train/fps": int(i / (time.time() - start_time))}, step=i)

        if resets and i % reset_interval == 0:
            agent = agent_fn(env)

    # Save final checkpoint and replay buffer.
    experiment.save_checkpoint(agent, step=max_steps, overwrite=True)
    utils.atomic_save(experiment.data_dir / "replay_buffer.pkl", replay_buffer.data)

    # Mark run as finished.
    run.finish()


def eval_loop(
    experiment: Experiment,
    env_fn: EnvFn,
    agent_fn: AgentFn,
    logger_fn: LoggerFn,
    num_episodes: int,
    video_dir: Path,
    max_steps: int,
) -> None:
    env = env_fn()
    agent = agent_fn(env)
    run = logger_fn()
    last_checkpoint = None
    while True:
        checkpoint = experiment.latest_checkpoint()
        if checkpoint == last_checkpoint or checkpoint is None:
            time.sleep(10.0)
        else:
            # Restore checkpoint.
            agent = experiment.restore_checkpoint(agent)
            i = int(Path(checkpoint).stem.split("_")[-1])
            print(f"Evaluating checkpoint at iteration {i}")

            # Eval!
            for _ in range(num_episodes):
                timestep = env.reset()
                while not timestep.last():
                    timestep = env.step(agent.eval_actions(timestep.observation))

            # Log statistics and video.
            log_dict = utils.prefix_dict("eval", env.get_statistics())
            run.log(log_dict, step=i)
            video_path = utils.get_latest_video(video_dir)
            if video_path is not None:
                video = wandb.Video(str(video_path), fps=20, format="mp4")
                run.log({"video": video}, step=i)
                video_path.unlink()  # Delete video after uploading.

            print(f"Done evaluating checkpoint {i}")
            last_checkpoint = checkpoint

            # Exit if we've evaluated the last checkpoint.
            if i >= max_steps:
                print(f"Last checkpoint (iteration {i}) evaluated, exiting")
                break

    # Mark run as finished.
    run.finish()
