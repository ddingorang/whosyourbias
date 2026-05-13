# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SAMURAI (Adapting Segment Anything Model for Zero-Shot Visual Tracking with Motion-Aware Memory) extends Meta's SAM 2 for zero-shot visual object tracking (VOT) in videos, adding Kalman filter-based motion estimation and motion-aware memory selection — without requiring additional training.

## Setup & Installation

```bash
# Install the sam2 package (run from repo root)
cd sam2
pip install -e .
pip install -e ".[notebooks]"
cd ..

# Install additional dependencies
pip install matplotlib==3.7 tikzplotlib jpeg4py opencv-python lmdb pandas scipy loguru

# Download SAM 2.1 checkpoints
cd sam2/checkpoints && bash download_ckpts.sh && cd ../..
```

**Requirements:** Python >= 3.10, PyTorch >= 2.3.1 with CUDA, TorchVision >= 0.18.1.

## Web Demo (Interactive UI)

An interactive web app lives in `app/`. It provides video upload → bbox drawing → tracking → result download.

**Setup (first time):**
```bash
pip install fastapi "uvicorn[standard]" python-multipart opencv-python
```

**Run:**
```bash
# From repo root:
bash run_app.sh
# Or directly:
python -m uvicorn app.server:app --host 0.0.0.0 --port 8000 --reload
```

Open `http://localhost:8000` in a browser.

**Structure:**
- `app/server.py` — FastAPI backend (upload, inference, serve results)
- `app/static/index.html` — single-page frontend (no build step)
- `app/uploads/` / `app/outputs/` — temp files (gitignored)

API endpoints: `POST /api/upload`, `POST /api/track`, `GET /api/status/{job_id}`, `GET /api/result/{job_id}`, `GET /api/health`

## Running Inference

**Custom video demo (most common):**
```bash
python scripts/demo.py \
  --video_path <video.mp4 or frame_dir/> \
  --txt_path <first_frame_bbox.txt> \
  --model_path sam2/checkpoints/sam2.1_hiera_base_plus.pt \
  --video_output_path demo.mp4
```

**LaSOT benchmark (single GPU):**
```bash
python scripts/main_inference.py
```

**Multi-GPU parallel (8 GPUs via chunking):**
```bash
bash scripts/inference.sh
# Or directly:
python scripts/main_inference_chunk.py \
  --dataset_path data/LaSOT-ext --tracker_name samurai \
  --model_name large --chunk_idx 0 --num_chunks 8 \
  --exp_name test --root_result_folder results
```

**VOS benchmarks (DAVIS/MOSE/SA-V):**
```bash
python sam2/tools/vos_inference.py \
  --sam2_cfg configs/samurai/sam2.1_hiera_l.yaml \
  --sam2_checkpoint sam2/checkpoints/sam2.1_hiera_large.pt \
  --base_video_dir /path/to/JPEGImages \
  --input_mask_dir /path/to/Annotations \
  --video_list_file /path/to/val.txt \
  --output_mask_dir ./outputs/
```

## Architecture

```
Input Video
    ↓
SAM2VideoPredictor (sam2/sam2/sam2_video_predictor.py)
    ├── init_state()           — load frames, init tracking state
    ├── add_new_points_or_box() — prompt with bbox on first frame
    └── propagate_in_video()   — track through all frames
         ↓
SAM2Base (sam2/sam2/modeling/sam2_base.py)
    ├── ImageEncoder (Hiera backbone + FPN neck)
    ├── MemoryEncoder    — compress mask+features into 64D memory
    ├── MemoryAttention  — RoPE cross-attention over stored memory frames
    └── MaskDecoder      — predict segmentation masks
         ↓
SAMURAI Extensions (in sam2_base.py + kalman_filter.py)
    ├── KalmanFilter     — track bbox state across frames
    ├── kf_score_weight  — blend SAM confidence with KF estimate
    └── Stability gating — discard unstable frames from memory bank
```

**Key files:**
- `sam2/sam2/sam2_video_predictor.py` — public API for video tracking
- `sam2/sam2/modeling/sam2_base.py` — core model + all SAMURAI-specific logic (1061 lines)
- `sam2/sam2/utils/kalman_filter.py` — Kalman filter for motion estimation
- `sam2/configs/samurai/sam2.1_hiera_b+.yaml` — default SAMURAI config
- `scripts/demo.py` — entry point for custom video inference

## SAMURAI-Specific Configuration

SAMURAI configs live in `sam2/configs/samurai/` and add these parameters (not present in base SAM 2 configs):

```yaml
samurai_mode: true
stable_frames_threshold: 15    # frames needed before relying on KF
stable_ious_threshold: 0.3     # IoU floor for frame stability
kf_score_weight: 0.25          # weight of Kalman score in memory selection
memory_bank_iou_threshold: 0.7 # IoU threshold for adding frames to memory bank
```

Model sizes: `sam2.1_hiera_t.yaml` (tiny), `sam2.1_hiera_s.yaml` (small), `sam2.1_hiera_b+.yaml` (base plus, default), `sam2.1_hiera_l.yaml` (large).

## Evaluation Framework

The `lib/` directory contains the VOT evaluation harness:
- `lib/test/evaluation/` — dataset loaders for LaSOT, GOT-10k, OTB, TrackingNet, NFS, UAV123
- `lib/test/analysis/` — result extraction and plotting utilities
- Dataset paths are configured inside the dataset loader files; data is expected under `data/`

## CUDA Extension

`sam2/sam2/csrc/connected_components.cu` provides a GPU-accelerated connected-components algorithm. It is compiled during `pip install -e .`. To skip CUDA compilation: `export SAM2_BUILD_CUDA=0` before installing.
