import errno
import os
import pickle
import random
import shutil
import signal
import tempfile
import uuid
from pathlib import Path
from typing import Any, Optional
import dm_env
import dm_env_wrappers as wrappers

import numpy as np

PathOrStr = str | Path


def get_latest_video(video_dir: Path) -> Optional[Path]:
    """Returns the latest video in `video_dir`."""
    videos = list(video_dir.glob("*.mp4"))
    if not videos:
        return None
    videos.sort(key=lambda x: int(x.stem.split("_")[-1]))
    return videos[-1]


def prefix_dict(prefix: str, d: dict) -> dict:
    """Prefixes all keys in `d` with `prefix`."""
    return {f"{prefix}/{k}": v for k, v in d.items()}


def merge_dict(d1: dict, d2: dict) -> dict:
    """Merges two dictionaries."""
    return d1 | d2


def atomic_save(save_path: Path, obj: Any) -> None:
    # Ignore nanorl+c while saving.
    try:
        orig_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, lambda _sig, _frame: None)
    except ValueError:
        # Signal throws a ValueError if we're not in the main thread.
        orig_handler = None

    with tempfile.TemporaryDirectory() as tmp_dir:
        # First, save to a temporary file.
        tmp_path = Path(tmp_dir) / f"{uuid.uuid4()}.tmp.pkl"
        with open(tmp_path, "wb") as f:
            pickle.dump(obj, f)

        # Next, try an `os.rename`.
        try:
            os.rename(tmp_path, save_path)
        # If that fails, it means we're copying across a filesystem boundary. Fallback
        # to a copy.
        except OSError as e:
            if e.errno == errno.EXDEV:
                shutil.copy(tmp_path, save_path)
            else:
                raise

    # Restore SIGINT handler.
    if orig_handler is not None:
        signal.signal(signal.SIGINT, orig_handler)


def pickle_save(save_path: Path, obj: Any) -> None:
    with open(save_path, "wb") as f:
        pickle.dump(obj, f)


def seed_rngs(seed: int) -> None:
    """Seeds all random number generators."""
    random.seed(seed)
    np.random.seed(seed)


def wrap_env(
    env: dm_env.Environment,
    record_dir: Optional[PathOrStr] = None,
    record_every: int = 1,
    record_resolution: tuple[int, int] = (480, 640),
    camera_id: Optional[str | int] = 0,
    frame_stack: int = 1,
    clip: bool = True,
    action_reward_observation: bool = False,
) -> dm_env.Environment:
    if record_dir is not None:
        env = wrappers.DmControlVideoWrapper(
            environment=env,
            record_dir=record_dir,
            record_every=record_every,
            camera_id=camera_id,
            height=record_resolution[0],
            width=record_resolution[1],
        )
        env = wrappers.EpisodeStatisticsWrapper(
            environment=env, deque_size=record_every
        )
    else:
        env = wrappers.EpisodeStatisticsWrapper(environment=env, deque_size=1)

    if action_reward_observation:
        env = wrappers.ObservationActionRewardWrapper(env)

    env = wrappers.ConcatObservationWrapper(env)

    if frame_stack > 1:
        env = wrappers.FrameStackingWrapper(env, num_frames=frame_stack, flatten=True)

    env = wrappers.CanonicalSpecWrapper(env, clip=clip)
    env = wrappers.SinglePrecisionWrapper(env)

    return env
