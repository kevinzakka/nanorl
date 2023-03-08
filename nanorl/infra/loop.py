import time
from pathlib import Path
from typing import Any, Callable

import dm_env
import tqdm
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

    spec = specs.EnvironmentSpec.make(env)
    timestep = env.reset()
    replay_buffer.insert(timestep, None)

    start_time = time.time()
    for i in tqdm.tqdm(range(1, max_steps + 1), disable=not tqdm_bar):
        if i < warmstart_steps:
            action = spec.sample_action(random_state=env.random_state)
        else:
            agent, action = agent.sample_actions(timestep.observation)

        timestep = env.step(action)
        replay_buffer.insert(timestep, action)

        if timestep.last():
            experiment.log(utils.prefix_dict("train", env.get_statistics()), step=i)
            timestep = env.reset()
            replay_buffer.insert(timestep, None)

        if i >= warmstart_steps:
            if replay_buffer.is_ready():
                transitions = replay_buffer.sample()
                agent, metrics = agent.update(transitions)
                if i % log_interval == 0:
                    experiment.log(utils.prefix_dict("train", metrics), step=i)

        if checkpoint_interval >= 0 and i % checkpoint_interval == 0:
            experiment.save_checkpoint(agent, step=i)

        if i % log_interval == 0:
            experiment.log({"train/fps": int(i / (time.time() - start_time))}, step=i)

        if resets and i % reset_interval == 0:
            agent = agent_fn(env)

    # Save final checkpoint and replay buffer.
    experiment.save_checkpoint(agent, step=max_steps, overwrite=True)
    utils.atomic_save(experiment.data_dir / "replay_buffer.pkl", replay_buffer.data)


def eval_loop(
    experiment: Experiment,
    env_fn: EnvFn,
    agent_fn: AgentFn,
    num_episodes: int,
    max_steps: int,
) -> None:
    env = env_fn()
    agent = agent_fn(env)

    last_checkpoint = None
    while True:
        # Wait for new checkpoint.
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

            # Log statistics.
            log_dict = utils.prefix_dict("eval", env.get_statistics())
            experiment.log(log_dict, step=i)

            # Maybe log video.
            experiment.log_video(env.latest_filename, step=i)

            print(f"Done evaluating checkpoint {i}")
            last_checkpoint = checkpoint

            # Exit if we've evaluated the last checkpoint.
            if i >= max_steps:
                print(f"Last checkpoint (iteration {i}) evaluated, exiting")
                break
