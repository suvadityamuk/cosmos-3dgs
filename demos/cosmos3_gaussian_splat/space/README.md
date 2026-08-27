---
title: Cosmos 3 Gaussian Splat
emoji: 🪑
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 6.15.1
app_file: app.py
python_version: "3.12"
startup_duration_timeout: 1h
short_description: Cosmos camera generation and Gaussian splat research
---

# Cosmos 3 Gaussian Splat Research Demo

This Space runs camera-conditioned Cosmos3-Nano video generation live on ZeroGPU xlarge. It also
hosts a precomputed VGGT + gsplat reconstruction from the standalone research pipeline.

The released Cosmos `camera_pose` model supports general cinematic camera motion, but validation
did not produce a rigid near-field 360-degree chair orbit. The experimental orbit option is exposed
to make that model limitation reproducible; it should not be interpreted as calibrated rendering.

Full VGGT reconstruction and Gaussian optimization run through an A100 Hugging Face Job rather
than ZeroGPU because they require long-lived CUDA extensions and exceed interactive visitor quota.
