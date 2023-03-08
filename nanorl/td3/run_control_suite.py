"""Train a TD3 agent on dm_control suite tasks."""

import time
from dataclasses import asdict, dataclass
from concurrent import futures
from pathlib import Path
from typing import Optional
import dm_env
import tyro
from dm_control import suite

from nanorl import replay, specs
from nanorl import TD3, TD3Config
from nanorl.infra import seed_rngs, Experiment, train_loop, eval_loop, wrap_env


@dataclass(frozen=True)
class Args:
    # Experiment configuration.
    root_dir: str = "/tmp/nanorl"
    """Where experiment directories are created."""
    seed: int = 42
    """RNG seed."""
    max_steps: int = 1_000_000
    """Total number of environment steps to train for."""
    warmstart_steps: int = 5_000
    """Number of steps in which to take random actions before starting training."""
    log_interval: int = 1_000
    """Number of steps between logging to wandb."""
    checkpoint_interval: int = -1
    """Number of steps between checkpoints and evaluations. Set to -1 to disable."""
    reset_interval: int = 200_000
    """Number of steps between resetting the policy."""
    eval_episodes: int = 10
    """Number of episodes to run at every evaluation."""
    batch_size: int = 256
    """Batch size for training."""
    discount: float = 0.99
    """Discount factor."""
    tqdm_bar: bool = False
    """Whether to use a tqdm progress bar in the training loop."""
    resets: bool = False
    """Whether to periodically reset the actor / critic layers."""
    init_from_checkpoint: Optional[str] = None
    """Path to a checkpoint to initialize the agent from."""

    # Replay buffer configuration.
    replay_capacity: int = 1_000_000
    """Replay buffer capacity."""
    offline_dataset: Optional[str] = None
    """Path to a pickle file containing a list of transitions."""
    offline_pct: float = 0.5
    """Percentage of offline data to use."""

    # W&B configuration.
    use_wandb: bool = False
    project: str = "nanorl"
    entity: str = ""
    name: str = ""
    tags: str = ""
    notes: str = ""
    mode: str = "online"

    # Task configuration.
    domain_name: str = "cartpole"
    """Which domain to use."""
    task_name: str = "swingup"
    """Which task to use."""

    # Environment wrapper configuration.
    frame_stack: int = 1
    """Number of frames to stack."""
    clip: bool = True
    """Whether to clip actions outside the canonical range."""
    record_dir: Optional[Path] = None
    """Where evaluation video renders are saved."""
    record_every: int = 1
    """How often to record videos."""
    camera_id: Optional[str | int] = 0
    """Camera to use for rendering."""
    action_reward_observation: bool = False
    """Whether to include the action and reward in the observation."""

    # TD3-specific configuration.
    agent_config: TD3Config = TD3Config()


def main(args: Args) -> None:
    if args.name:
        run_name = args.name
    else:
        run_name = f"TD3-{args.domain_name}-{args.task_name}-{args.seed}-{time.time()}"

    # Seed RNGs.
    seed_rngs(args.seed)

    # Setup the experiment for checkpoints, videos, metadata, etc.
    experiment = Experiment(Path(args.root_dir) / run_name).assert_new()
    experiment.write_metadata("config", args)

    if args.use_wandb:
        experiment.enable_wandb(
            project=args.project,
            entity=args.entity or None,
            tags=(args.tags.split(",") if args.tags else []),
            notes=args.notes or None,
            config=asdict(args),
            mode=args.mode,
            name=run_name,
            sync_tensorboard=True,
        )

    def agent_fn(env: dm_env.Environment) -> TD3:
        agent = TD3.initialize(
            spec=specs.EnvironmentSpec.make(env),
            config=args.agent_config,
            seed=args.seed,
            discount=args.discount,
        )

        if args.init_from_checkpoint is not None:
            ckpt_exp = Experiment(Path(args.init_from_checkpoint)).assert_exists()
            agent = ckpt_exp.restore_checkpoint(agent)

        return agent

    def replay_fn(env: dm_env.Environment) -> replay.ReplayBuffer:
        if args.offline_dataset is not None:
            offline_dataset = Path(args.offline_dataset)
            if not offline_dataset.exists():
                raise FileNotFoundError(f"Offline dataset {offline_dataset} not found.")
        else:
            offline_dataset = None

        utd_ratio = max(
            args.agent_config.critic_utd_ratio, args.agent_config.actor_utd_ratio
        )

        return replay.ReplayBuffer(
            capacity=args.replay_capacity,
            batch_size=args.batch_size * utd_ratio,
            spec=specs.EnvironmentSpec.make(env),
            offline_dataset=offline_dataset,
            offline_pct=args.offline_pct,
        )

    def env_fn(record_dir: Optional[Path] = None) -> dm_env.Environment:
        env = suite.load(
            domain_name=args.domain_name,
            task_name=args.task_name,
            task_kwargs=dict(random=args.seed),
        )

        return wrap_env(
            env=env,
            record_dir=record_dir,
            record_every=args.record_every,
            frame_stack=args.frame_stack,
            clip=args.clip,
            camera_id=args.camera_id,
            action_reward_observation=args.action_reward_observation,
        )

    pool = futures.ThreadPoolExecutor(1)

    # Run training in a background thread.
    pool.submit(
        train_loop,
        experiment=experiment,
        env_fn=env_fn,
        agent_fn=agent_fn,
        replay_fn=replay_fn,
        max_steps=args.max_steps,
        warmstart_steps=args.warmstart_steps,
        log_interval=args.log_interval,
        checkpoint_interval=args.checkpoint_interval,
        resets=args.resets,
        reset_interval=args.reset_interval,
        tqdm_bar=args.tqdm_bar,
    )

    # Continuously monitor for checkpoints and evaluate.
    eval_loop(
        experiment=experiment,
        env_fn=lambda: env_fn(record_dir=experiment.data_dir / "videos"),
        agent_fn=agent_fn,
        num_episodes=args.eval_episodes,
        max_steps=args.max_steps,
    )

    # Clean up.
    pool.shutdown()


if __name__ == "__main__":
    main(tyro.cli(Args, description=__doc__))
