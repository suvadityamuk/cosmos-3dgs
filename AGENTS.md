# AGENTS.md

## Cursor Cloud specific instructions

### What this repo is

This is the **NVIDIA Cosmos 3** cookbook + evaluation repository. It is documentation-first:
its "products" are Jupyter notebooks and shell recipes under `cookbooks/` and `evaluation/`
that demonstrate Cosmos 3 Reasoner/Generator inference across several backends (Diffusers,
Transformers, vLLM, vLLM-Omni, TensorRT-LLM, SGLang, NIM, Cosmos Framework). There is **no
application source tree, no root test suite, no root lint config, and no CI** in this repo —
just markdown, notebooks (`*.ipynb`), helper `.py` files, and launch scripts.

Authoritative setup docs (do not duplicate them — read them):
- `README.md` (root) — per-backend Quickstart and troubleshooting.
- `cookbooks/cosmos3/README.md` — shared environment setup for every cookbook backend.
- Each subdir `README.md` (e.g. `cookbooks/cosmos3/generator/transfer/README.md`).

### Hard constraint: no GPU on the Cloud VM

Every real model-inference path here **requires an NVIDIA GPU** (Cosmos3-Edge 4B / Nano 16B /
Super 64B), CUDA 12.8/13, **gated Hugging Face model access** (and NGC for NIM), plus
tens-of-GiB weight downloads. The Cloud Agent VM has **no GPU** (`nvidia-smi` is absent), so
you **cannot run actual inference, servers, or the GPU notebooks end-to-end here.** Do not
burn time trying to `vllm serve` / run Diffusers pipelines / launch NIM containers — they will
fail on missing CUDA devices. Treat GPU inference as out of scope for this environment and
verify only the CPU-runnable, non-inference parts.

### Package manager: `uv`

`uv` is the package manager used by every backend and is preinstalled on `PATH` at
`/usr/local/cargo/bin/uv` by the environment update script. There is **no repo-wide lockfile
or `uv sync`** — the documented pattern is that each backend/notebook creates its **own
ephemeral venv** (`uv venv --python 3.13 --seed --managed-python`) and installs the
CUDA-matched deps for that backend. So "install dependencies" is per-notebook, not global.

### CPU-only development / validation you *can* do here

Create a throwaway venv and work with the notebooks and the repo's helper code (no GPU):

```bash
uv venv --python 3.13 --seed --managed-python /tmp/cosmos-dev
source /tmp/cosmos-dev/bin/activate
uv pip install jupyter nbconvert ipython imageio imageio-ffmpeg
```

Useful CPU-only checks:
- Validate/parse every notebook: `python -c "import glob,nbformat; [nbformat.validate(nbformat.read(p,as_version=4)) for p in glob.glob('**/*.ipynb',recursive=True)]"`.
- Exercise the transfer cookbook helpers against real repo assets (spec loading + an
  ffmpeg-backed preview encode — all CPU): import
  `cookbooks/cosmos3/generator/transfer/preview_helpers.py` and call
  `load_transfer_spec(...)`, `resolve_spec_path(...)`, `make_preview(...)`. System `ffmpeg`
  is available, and `imageio-ffmpeg` also bundles one.
- Media assets under `cookbooks/**/assets/` are committed as real files (not git-LFS
  pointers), so helper code that reads them works offline.

Notebook `make_preview` writes `*_preview.mp4` next to source assets — these are generated
byproducts; do not commit them.

### For real GPU runs (elsewhere, not this VM)

Authenticate before running: `uvx hf@latest auth login` or `export HF_TOKEN=...` (and request
access to the gated `nvidia/Cosmos3-*` and `nvidia/Cosmos-1.0-Guardrail` repos). NIM paths use
`NGC_API_KEY` instead. See `cookbooks/cosmos3/README.md` for the exact per-backend commands.
