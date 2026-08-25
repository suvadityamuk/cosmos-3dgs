from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .config import GenerationConfig
from .telemetry import measure_stage, release_gpu_memory
from .trajectory import CameraTrajectory


@dataclass(frozen=True)
class CosmosGenerationResult:
    video_path: Path
    frames_dir: Path
    frame_paths: tuple[Path, ...]
    metadata_path: Path
    metrics: dict[str, float | str | bool]


def _normalise_video_frames(video: Any) -> list[Image.Image]:
    """Normalize common Diffusers video outputs to one list of RGB PIL frames."""

    if hasattr(video, "detach"):
        video = video.detach().float().cpu().numpy()
    if isinstance(video, np.ndarray):
        array = video
        if array.ndim == 5:
            array = array[0]
        if array.ndim != 4:
            raise ValueError(f"unexpected video array shape: {array.shape}")
        if array.shape[1] in (1, 3, 4):
            array = np.moveaxis(array, 1, -1)
        if np.issubdtype(array.dtype, np.floating):
            if array.min() < 0:
                array = (array + 1.0) / 2.0
            array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
        return [Image.fromarray(frame).convert("RGB") for frame in array]
    if isinstance(video, (list, tuple)):
        frames = list(video)
        if len(frames) == 1 and isinstance(frames[0], (list, tuple, np.ndarray)):
            return _normalise_video_frames(frames[0])
        normalized: list[Image.Image] = []
        for frame in frames:
            if isinstance(frame, Image.Image):
                normalized.append(frame.convert("RGB"))
            else:
                array = np.asarray(frame)
                if np.issubdtype(array.dtype, np.floating):
                    if array.min() < 0:
                        array = (array + 1.0) / 2.0
                    array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
                normalized.append(Image.fromarray(array).convert("RGB"))
        return normalized
    raise TypeError(f"unsupported Diffusers video type: {type(video)!r}")


class CosmosCameraGenerator:
    """Run Cosmos 3 camera-pose forward dynamics with lazy GPU imports."""

    def __init__(self, config: GenerationConfig) -> None:
        self.config = config

    def run(
        self,
        *,
        prompt: str,
        image: str | Path | Image.Image,
        trajectory: CameraTrajectory,
        output_dir: str | Path,
    ) -> CosmosGenerationResult:
        try:
            import torch
            from diffusers import Cosmos3OmniPipeline, CosmosActionCondition
            from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
            from diffusers.utils import export_to_video, load_image
        except ImportError as error:
            raise RuntimeError("Cosmos generation requires the package's 'gpu' optional dependencies") from error

        if not torch.cuda.is_available():
            raise RuntimeError("Cosmos 3 generation requires a CUDA GPU")
        root = Path(output_dir)
        frames_dir = root / "frames"
        root.mkdir(parents=True, exist_ok=True)
        frames_dir.mkdir(parents=True, exist_ok=True)
        reference = image.convert("RGB") if isinstance(image, Image.Image) else load_image(str(image)).convert("RGB")
        token = os.environ.get("HF_TOKEN") or None
        metrics: dict[str, float | str | bool] = {}
        pipe = None
        with measure_stage("cosmos_generation") as stage_metrics:
            try:
                pipe = Cosmos3OmniPipeline.from_pretrained(
                    self.config.model_id,
                    torch_dtype=torch.bfloat16,
                    enable_safety_checker=self.config.enable_guardrails,
                    token=token,
                )
                pipe.scheduler = UniPCMultistepScheduler.from_config(
                    pipe.scheduler.config,
                    flow_shift=self.config.flow_shift,
                    use_karras_sigmas=False,
                )
                if self.config.cpu_offload:
                    pipe.enable_model_cpu_offload()
                else:
                    pipe.to("cuda")

                actions = torch.as_tensor(trajectory.raw_actions, dtype=torch.float32)
                result = pipe(
                    prompt=prompt,
                    action=CosmosActionCondition(
                        mode="forward_dynamics",
                        chunk_size=len(actions),
                        domain_name="camera_pose",
                        resolution_tier=self.config.resolution_tier,
                        raw_actions=actions,
                        image=reference,
                        view_point="ego_view",
                    ),
                    fps=self.config.fps,
                    num_inference_steps=self.config.num_inference_steps,
                    guidance_scale=self.config.guidance_scale,
                    use_system_prompt=False,
                    enable_safety_check=self.config.enable_guardrails,
                    generator=torch.Generator(device="cuda").manual_seed(self.config.seed),
                )
                frames = _normalise_video_frames(result.video)
                expected_frames = len(trajectory.poses_c2w)
                if len(frames) != expected_frames:
                    raise RuntimeError(f"Cosmos returned {len(frames)} frames; expected {expected_frames}")
                frame_paths: list[Path] = []
                for index, frame in enumerate(frames):
                    frame_path = frames_dir / f"frame_{index:04d}.png"
                    frame.save(frame_path)
                    frame_paths.append(frame_path)
                video_path = root / "cosmos_orbit.mp4"
                export_to_video(frames, str(video_path), fps=self.config.fps, macro_block_size=1)
            finally:
                if pipe is not None and hasattr(pipe, "maybe_free_model_hooks"):
                    pipe.maybe_free_model_hooks()
                del pipe
                release_gpu_memory()
            metrics.update(stage_metrics)

        metadata_path = root / "generation.json"
        metadata_path.write_text(
            json.dumps(
                {
                    "model_id": self.config.model_id,
                    "prompt": prompt,
                    "fps": self.config.fps,
                    "num_frames": len(frame_paths),
                    "seed": self.config.seed,
                    "guardrails_enabled": self.config.enable_guardrails,
                    "cpu_offload": self.config.cpu_offload,
                    "metrics": metrics,
                },
                indent=2,
            )
            + "\n"
        )
        return CosmosGenerationResult(
            video_path=video_path,
            frames_dir=frames_dir,
            frame_paths=tuple(frame_paths),
            metadata_path=metadata_path,
            metrics=metrics,
        )
