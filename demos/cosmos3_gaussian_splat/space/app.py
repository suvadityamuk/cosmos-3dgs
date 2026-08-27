from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import spaces
import gradio as gr
import numpy as np
import torch
from diffusers import Cosmos3OmniPipeline, CosmosActionCondition
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import export_to_video
from huggingface_hub import snapshot_download
from PIL import Image

MODEL_ID = "nvidia/Cosmos3-Nano"
ACTION_STEPS = 60


def _normalize(vector: np.ndarray) -> np.ndarray:
    return vector / np.clip(np.linalg.norm(vector), 1e-8, None)


def _look_at(center: np.ndarray, target: np.ndarray) -> np.ndarray:
    forward = _normalize(target - center)
    right = _normalize(np.cross(forward, np.asarray([0.0, 0.0, 1.0])))
    down = _normalize(np.cross(forward, right))
    pose = np.eye(4)
    pose[:3, :3] = np.stack((right, down, forward), axis=1)
    pose[:3, 3] = center
    return pose


def _orbit_actions() -> np.ndarray:
    phase = np.linspace(0.0, 1.0, ACTION_STEPS + 1)
    azimuth = 2.0 * np.pi * phase
    elevation = np.deg2rad(12.0) * np.sin(2.0 * np.pi * phase)
    horizontal = np.cos(elevation)
    centers = np.stack(
        (horizontal * np.cos(azimuth), horizontal * np.sin(azimuth), np.sin(elevation)),
        axis=-1,
    )
    poses = np.stack([_look_at(center, np.zeros(3)) for center in centers])
    deltas = np.linalg.inv(poses[:-1]) @ poses[1:]
    rot6d = np.swapaxes(deltas[:, :3, :2], -1, -2).reshape(ACTION_STEPS, 6)
    return np.concatenate((deltas[:, :3, 3], rot6d), axis=-1).astype(np.float32)


def _cinematic_push_actions() -> np.ndarray:
    action = np.asarray(
        [-0.864, -0.113, 0.119, 0.999989, 0.0, -0.00461, 0.0, 1.0, 0.0],
        dtype=np.float32,
    )
    return np.repeat(action[None], ACTION_STEPS, axis=0)


def _materialize_guardrail_nltk_data() -> None:
    import nltk

    snapshot = Path(snapshot_download("nvidia/Cosmos-1.0-Guardrail", token=os.environ.get("HF_TOKEN")))
    source = snapshot / "blocklist/nltk_data"
    destination = Path("/tmp/cosmos-guardrail-nltk-data")
    if not destination.is_dir():
        temporary = Path(tempfile.mkdtemp(prefix="cosmos-guardrail-nltk-"))
        try:
            shutil.copytree(source, temporary / "nltk_data", symlinks=False)
            (temporary / "nltk_data").replace(destination)
        finally:
            shutil.rmtree(temporary, ignore_errors=True)
    nltk.data.path.insert(0, str(destination))


pipe = Cosmos3OmniPipeline.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,
    enable_safety_checker=True,
    token=os.environ.get("HF_TOKEN"),
)
_materialize_guardrail_nltk_data()
pipe.scheduler = UniPCMultistepScheduler.from_config(
    pipe.scheduler.config,
    flow_shift=10.0,
    use_karras_sigmas=False,
)
pipe.to("cuda")


@spaces.GPU(duration=120, size="xlarge")
def generate(reference: Image.Image, prompt: str, motion: str, seed: int):
    if reference is None:
        raise gr.Error("Upload a reference image.")
    if not prompt.strip():
        raise gr.Error("Enter a prompt describing a stationary subject and camera motion.")
    actions = _cinematic_push_actions() if motion.startswith("Cinematic") else _orbit_actions()
    result = pipe(
        prompt=prompt,
        action=CosmosActionCondition(
            mode="forward_dynamics",
            chunk_size=len(actions),
            domain_name="camera_pose",
            resolution_tier=480,
            raw_actions=torch.from_numpy(actions),
            image=reference.convert("RGB"),
            view_point="ego_view",
        ),
        fps=30,
        num_inference_steps=30,
        guidance_scale=1.0,
        use_system_prompt=False,
        enable_safety_check=True,
        generator=torch.Generator(device="cuda").manual_seed(int(seed)),
    )
    output_dir = Path(tempfile.mkdtemp(prefix="cosmos3-space-"))
    video_path = output_dir / "cosmos3-camera-generation.mp4"
    export_to_video(result.video, str(video_path), fps=30, macro_block_size=1)
    note = (
        "Cinematic push uses an action profile close to NVIDIA's public example."
        if motion.startswith("Cinematic")
        else "The 360° orbit is experimental and may collapse to wobble or morphing."
    )
    return str(video_path), note


example_root = Path(__file__).parent / "examples"
example_video = example_root / "generated-vs-splat.mp4"
example_ply = example_root / "gaussian_splat.ply"
example_compact = example_root / "gaussian_splat.splat"

with gr.Blocks(title="Cosmos 3 Gaussian Splat") as demo:
    gr.Markdown(
        """
# Cosmos 3 Gaussian Splat Research Demo

Generate a camera-conditioned Cosmos3-Nano video live on **ZeroGPU xlarge**.
The validated reconstruction below was produced by the full VGGT + gsplat A100 pipeline.

> The released camera model handles cinematic motion but did not produce a rigid near-field
> 360° chair orbit in our tests. The orbit option intentionally exposes that limitation.
"""
    )
    with gr.Tab("Live Cosmos generation"):
        with gr.Row():
            reference = gr.Image(type="pil", label="Reference image")
            with gr.Column():
                prompt = gr.Textbox(
                    value="A stationary chair in a neutral studio while the camera moves smoothly.",
                    lines=4,
                    label="Prompt",
                )
                motion = gr.Radio(
                    ["Cinematic push (recommended)", "Experimental 360° orbit"],
                    value="Cinematic push (recommended)",
                    label="Camera motion",
                )
                seed = gr.Number(value=0, precision=0, label="Seed")
                button = gr.Button("Generate video", variant="primary")
        live_video = gr.Video(label="Generated video")
        status = gr.Markdown()
        button.click(
            generate,
            inputs=[reference, prompt, motion, seed],
            outputs=[live_video, status],
            api_name="generate",
            concurrency_limit=1,
            concurrency_id="cosmos3-generation",
        )
    with gr.Tab("Precomputed Gaussian splat result"):
        gr.Markdown(
            "This example completed Cosmos generation, VGGT geometry recovery, and gsplat optimization on an A100."
        )
        gr.Video(
            value=str(example_video) if example_video.is_file() else None,
            label="Cosmos generated views vs. Gaussian splat",
        )
        with gr.Row():
            gr.File(value=str(example_ply) if example_ply.is_file() else None, label="Gaussian splat PLY")
            gr.File(
                value=str(example_compact) if example_compact.is_file() else None,
                label="Compact .splat",
            )

demo.queue(default_concurrency_limit=1).launch()
