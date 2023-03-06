import pathlib
from dataclasses import dataclass, is_dataclass
from typing import Any, Optional, Type, TypeVar

import flax.training.checkpoints
import tyro
import yaml
from typing_extensions import get_origin

T = TypeVar("T")


def _get_origin(cls: Type) -> Type:
    """Get origin type; helpful for unwrapping generics, etc."""
    origin = get_origin(cls)
    return cls if origin is None else origin


# Adapted from https://github.com/brentyi/fifteen.
@dataclass(frozen=True)
class Experiment:
    """A simple directory associated with a run of some training script."""

    data_dir: pathlib.Path

    # Checkpointing.

    def save_checkpoint(
        self,
        target,
        step: int,
        prefix: str = "checkpoint_",
        keep: int = 1,
        overwrite: bool = False,
        keep_every_n_steps: Optional[int] = None,
    ) -> str:
        self._maybe_mkdir()
        filename = flax.training.checkpoints.save_checkpoint(
            ckpt_dir=str(self.data_dir),
            target=target,
            step=step,
            prefix=prefix,
            keep=keep,
            overwrite=overwrite,
            keep_every_n_steps=keep_every_n_steps,
        )
        return filename

    def restore_checkpoint(
        self,
        target,
        step: Optional[int] = None,
        prefix: str = "checkpoint_",
    ):
        state_dict = flax.training.checkpoints.restore_checkpoint(
            ckpt_dir=str(self.data_dir),
            target=None,
            step=step,
            prefix=prefix,
        )
        if state_dict is None:
            raise FileNotFoundError(f"No checkpoint found in {self.data_dir}.")
        return flax.serialization.from_state_dict(target, state_dict)

    def latest_checkpoint(self) -> Optional[str]:
        return flax.training.checkpoints.latest_checkpoint(self.data_dir)

    # Metadata.

    def write_metadata(self, name: str, object: Any) -> None:
        self._maybe_mkdir()
        assert ".yml" not in name

        path = self.data_dir / f"{name}.yml"
        with open(path, "w") as f:
            f.write(tyro.to_yaml(object) if is_dataclass(object) else yaml.dump(object))

    def read_metadata(self, name: str, expected_type: Type[T]) -> T:
        assert ".yml" not in name

        path = self.data_dir / f"{name}.yml"
        with open(path, "r") as f:
            output = (
                tyro.from_yaml(expected_type, f.read())
                if is_dataclass(_get_origin(expected_type))
                else yaml.load(f.read(), Loader=yaml.Loader)
            )
        assert isinstance(output, expected_type)
        return output

    # Helpers.

    def assert_new(self) -> "Experiment":
        """Asserts that an experiment is new, otherwise returns self."""
        if self.data_dir.exists():
            raise ValueError(f"Experiment {self} already exists.")
        return self

    def assert_exists(self) -> "Experiment":
        """Asserts that an experiment exists, otherwise returns self."""
        if not self.data_dir.exists():
            raise ValueError(f"Experiment {self} does not exist.")
        return self

    # Misc.

    def _maybe_mkdir(self) -> None:
        if not self.data_dir.exists():
            self.data_dir.mkdir(parents=True)
