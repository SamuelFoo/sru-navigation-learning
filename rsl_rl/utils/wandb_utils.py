#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
from dataclasses import asdict
from torch.utils.tensorboard import SummaryWriter

try:
    import wandb
except ModuleNotFoundError:
    raise ModuleNotFoundError("Wandb is required to log to Weights and Biases.")


class WandbSummaryWriter(SummaryWriter):
    """Summary writer for Weights and Biases."""

    def __init__(self, log_dir: str, flush_secs: int, cfg):
        super().__init__(log_dir, flush_secs)

        try:
            project = cfg["wandb_project"]
        except KeyError:
            raise KeyError("Please specify wandb_project in the runner config, e.g. legged_gym.")

        # Try to get entity from environment, but make it optional
        entity = os.environ.get("WANDB_USERNAME", None)

        # A resume reuses the log directory, so the run id recorded there lets the
        # continuation append to the original W&B run instead of starting a new one.
        self._run_id_file = os.path.join(log_dir, "wandb_run_id.txt")
        resume_id = self._read_run_id()
        init_kwargs = {"project": project}
        if resume_id:
            print(f"Resuming W&B run {resume_id}")
            init_kwargs.update(id=resume_id, resume="allow")

        # Try to initialize with entity first, fall back to default if permission denied
        try:
            if entity:
                print(f"Logging to project {project} with entity {entity}")
                wandb.init(entity=entity, **init_kwargs)
            else:
                print(f"Logging to project {project} (using default entity from API key)")
                wandb.init(**init_kwargs)
        except wandb.errors.CommError as e:
            if "403" in str(e) or "permission denied" in str(e).lower():
                print(f"Permission denied for entity {entity}, trying without entity specification...")
                wandb.init(**init_kwargs)
            else:
                raise

        self.name_map = {
            "Train/mean_reward/time": "Train/mean_reward_time",
            "Train/mean_episode_length/time": "Train/mean_episode_length_time",
        }

        run_name = os.path.split(log_dir)[-1]

        # Set wandb run name to match the log directory name
        wandb.run.name = run_name

        if not resume_id:
            self._write_run_id(wandb.run.id)

        wandb.log({"log_dir": run_name})

    def _read_run_id(self) -> str | None:
        """Return the W&B run id previously recorded in the log directory."""
        if not os.path.isfile(self._run_id_file):
            return None
        with open(self._run_id_file) as handle:
            return handle.read().strip() or None

    def _write_run_id(self, run_id: str) -> None:
        """Record the W&B run id so a later resume can append to this run."""
        os.makedirs(os.path.dirname(self._run_id_file), exist_ok=True)
        with open(self._run_id_file, "w") as handle:
            handle.write(run_id)

    def store_config(self, env_cfg, runner_cfg, alg_cfg, policy_cfg):
        wandb.config.update({"runner_cfg": runner_cfg})
        wandb.config.update({"policy_cfg": policy_cfg})
        wandb.config.update({"alg_cfg": alg_cfg})
        wandb.config.update({"env_cfg": asdict(env_cfg)})

    def _map_path(self, path):
        if path in self.name_map:
            return self.name_map[path]
        else:
            return path

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None, new_style=False):
        super().add_scalar(
            tag,
            scalar_value,
            global_step=global_step,
            walltime=walltime,
            new_style=new_style,
        )
        wandb.log({self._map_path(tag): scalar_value}, step=global_step)

    def stop(self):
        wandb.finish()

    def log_config(self, env_cfg, runner_cfg, alg_cfg, policy_cfg):
        self.store_config(env_cfg, runner_cfg, alg_cfg, policy_cfg)

    def save_model(self, model_path, iter):
        wandb.save(model_path, base_path=os.path.dirname(model_path))

    def save_file(self, path, iter=None):
        wandb.save(path, base_path=os.path.dirname(path))

    def log_video(self, frames, step, fps=30, tag="Train/video"):
        """Log a video to Weights and Biases.

        Args:
            frames: List or numpy array of video frames with shape (T, H, W, C) or (T, C, H, W).
            step: The global step for logging.
            fps: Frames per second for the video. Defaults to 30.
            tag: The tag/name for the video in wandb. Defaults to "Train/video".

        Returns:
            True if video was logged successfully, False otherwise.
        """
        # Check if wandb run is active
        if wandb.run is None:
            print("[WARNING] WandB run not initialized, skipping video logging")
            return False

        if not frames or len(frames) == 0:
            return False

        import numpy as np

        # Convert to numpy if needed
        if not isinstance(frames, np.ndarray):
            frames = np.array(frames)

        # wandb.Video expects (T, C, H, W) format
        # Check if frames are in (T, H, W, C) format and transpose
        if frames.ndim == 4 and frames.shape[-1] in [1, 3, 4]:
            # (T, H, W, C) -> (T, C, H, W)
            frames = np.transpose(frames, (0, 3, 1, 2))

        # Create wandb video and log
        video = wandb.Video(frames, fps=fps, format="mp4")
        wandb.log({tag: video}, step=step)
        return True
