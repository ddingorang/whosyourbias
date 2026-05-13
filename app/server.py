"""
SAMURAI Web Demo — FastAPI backend
Run from repo root:
    uvicorn app.server:app --host 0.0.0.0 --port 8000
"""

import contextlib
import gc
import math
import shutil
import sys
import uuid
import base64
import threading
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "sam2"))

UPLOADS_DIR = Path(__file__).parent / "uploads"
OUTPUTS_DIR = Path(__file__).parent / "outputs"
UPLOADS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

# ── Global state ───────────────────────────────────────────────────────────────
jobs: dict = {}
predictor = None
device_str = "cuda" if torch.cuda.is_available() else "cpu"
inference_lock = threading.Lock()

MODEL_PATH = ROOT / "sam2" / "checkpoints" / "sam2.1_hiera_base_plus.pt"
MODEL_CFG  = "configs/samurai/sam2.1_hiera_b+.yaml"

# 객체별 색상 (BGR) — 최대 8개 객체
COLORS_BGR = [
    (50,  220,  50),   # 초록
    (230, 180,   0),   # 하늘
    (  0, 140, 255),   # 주황
    (200,  50, 200),   # 보라
    ( 50, 220, 220),   # 노랑
    (255,  80,  80),   # 파랑
    ( 80, 255, 180),   # 민트
    (  0,  80, 255),   # 빨강
]

# 청크 처리 파라미터
MAX_SIDE     = 1920  # 긴 변 최대 픽셀 (2K→1080p 다운스케일, SAM2 내부 1024×1024 처리로 품질 동일)
CHUNK_FRAMES = 300   # 청크당 최대 프레임 (~10초 @30fps)
                     # 2K 다운스케일(1920×1080) 프레임당 ~6MB VRAM
                     # 300 × ~6MB ≈ 1.8GB → 모델·어텐션 버퍼 포함 16GB 내 안전

CROP_EMA_ALPHA = 0.4  # 크롭 중심 EMA 평활 계수 (클수록 최신 bbox에 민감)


# ── Startup ────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    if MODEL_PATH.exists():
        try:
            from sam2.build_sam import build_sam2_video_predictor
            predictor = build_sam2_video_predictor(
                MODEL_CFG, str(MODEL_PATH), device=device_str
            )
            print(f"[SAMURAI] Model loaded on {device_str.upper()}")
        except Exception as e:
            print(f"[SAMURAI] Model load failed: {e}")
    else:
        print(f"[SAMURAI] Checkpoint not found: {MODEL_PATH}")
        print("[SAMURAI] Run: cd sam2/checkpoints && bash download_ckpts.sh")
    yield


app = FastAPI(lifespan=lifespan)


# ── Helpers ────────────────────────────────────────────────────────────────────
def _scale_params(src_w: int, src_h: int):
    """축소 비율과 출력 해상도 반환. MAX_SIDE=0이면 원본 유지."""
    if MAX_SIDE > 0:
        scale = min(1.0, MAX_SIDE / max(src_w, src_h))
    else:
        scale = 1.0
    out_w = int(src_w * scale) & ~1
    out_h = int(src_h * scale) & ~1
    return scale, out_w, out_h


def _mask_to_bbox(mask: np.ndarray):
    """마스크 → (x1,y1,x2,y2) 바운딩박스."""
    nz = np.argwhere(mask)
    if len(nz) == 0:
        return None
    ymin, xmin = nz.min(axis=0)
    ymax, xmax = nz.max(axis=0)
    return int(xmin), int(ymin), int(xmax), int(ymax)


def _crop_9x16(frame: np.ndarray, cx: float, cy: float, crop_w: int, crop_h: int) -> np.ndarray:
    """EMA 중심 기준으로 9:16 고정 크기 크롭. 화면 경계에서 clamp."""
    h, w = frame.shape[:2]
    x1 = int(cx - crop_w / 2)
    y1 = int(cy - crop_h / 2)
    x1 = max(0, min(x1, max(0, w - crop_w)))
    y1 = max(0, min(y1, max(0, h - crop_h)))
    crop = frame[y1:y1 + crop_h, x1:x1 + crop_w]
    # 슬라이싱 결과가 선언 크기와 다를 경우 강제 리사이즈 (VideoWriter 크기 불일치 방지)
    if crop.shape[1] != crop_w or crop.shape[0] != crop_h:
        crop = cv2.resize(crop, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
    return crop


def _write_chunk_jpegs(frames: list, chunk_dir: Path):
    """프레임 리스트를 임시 JPEG 디렉터리에 저장."""
    chunk_dir.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames):
        cv2.imwrite(
            str(chunk_dir / f"{i:05d}.jpg"),
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 95],
        )


# ── Post-inference assembly (main + crop) ──────────────────────────────────────
def _assemble_videos(
    video_path: str,
    bbox_log: list,   # bbox_log[f][oid] = (x1,y1,x2,y2) or None  — inference-space coords
    crop_log: list,   # crop_log[f][oid] = (cx, cy)                — inference-space coords
    fps: float, job_id: str, n_objs: int,
    scale: float, src_w: int, src_h: int,
) -> tuple[str, list[str]]:
    """추론 완료 후 원본 영상을 1회 재독해 메인·크롭 영상을 원본 해상도로 생성한다."""
    import imageio

    crop_h = src_h
    crop_w = int(src_h * 9 / 16) & ~1

    main_filename  = f"{job_id}.mp4"
    crop_filenames = [f"{job_id}_obj{i}.mp4" for i in range(n_objs)]

    main_writer = imageio.get_writer(
        str(OUTPUTS_DIR / main_filename),
        fps=fps, codec="libx264", pixelformat="yuv420p",
        output_params=["-crf", "23", "-movflags", "+faststart"],
    )
    crop_writers = [
        imageio.get_writer(
            str(OUTPUTS_DIR / crop_filenames[i]),
            fps=fps, codec="libx264", pixelformat="yuv420p",
            output_params=["-crf", "23", "-movflags", "+faststart"],
        )
        for i in range(n_objs)
    ]

    cap = cv2.VideoCapture(video_path)
    for bboxes, centers in zip(bbox_log, crop_log):
        ret, frame = cap.read()
        if not ret:
            break

        # 크롭 영상: 오버레이 전 원본 프레임에서 크롭 (bbox 없음)
        for oid, (cx_inf, cy_inf) in enumerate(centers):
            cx = cx_inf / scale
            cy = cy_inf / scale
            crop = _crop_9x16(frame, cx, cy, crop_w, crop_h)
            crop_writers[oid].append_data(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

        # 메인 영상: bbox 좌표를 원본 해상도로 역스케일해 오버레이
        for oid, bb in enumerate(bboxes):
            if bb is None:
                continue
            x1 = int(bb[0] / scale)
            y1 = int(bb[1] / scale)
            x2 = int(bb[2] / scale)
            y2 = int(bb[3] / scale)
            color = COLORS_BGR[oid % len(COLORS_BGR)]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"obj{oid}", (x1, max(y1 - 6, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
        main_writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()
    main_writer.close()
    for w in crop_writers:
        w.close()
    return main_filename, crop_filenames


# ── Inference worker ───────────────────────────────────────────────────────────
def run_inference(job_id: str, video_path: str, bboxes: list[list[int]]):
    """청크 분할 방식으로 다중 객체를 추적."""
    chunk_dir = None
    try:
        jobs[job_id]["status"] = "running"

        if predictor is None:
            raise RuntimeError("모델이 로드되지 않았습니다. 체크포인트를 다운로드하세요.")

        # 영상 정보
        cap = cv2.VideoCapture(video_path)
        fps          = cap.get(cv2.CAP_PROP_FPS) or 30
        src_w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # 앞 1분만 처리
        max_frames = int(fps * 60)
        total_frames = min(total_frames, max_frames)

        scale, out_w, out_h = _scale_params(src_w, src_h)
        n_chunks = math.ceil(total_frames / CHUNK_FRAMES)

        # 입력 bbox 목록을 출력 해상도로 변환
        current_bboxes = [
            (int(b[0]*scale), int(b[1]*scale), int(b[2]*scale), int(b[3]*scale))
            for b in bboxes
        ]

        n_objs = len(bboxes)

        # ── 추론 중에는 좌표만 기록 (영상 인코딩은 추론 후 별도 패스) ────────
        bbox_log: list = []  # bbox_log[f][oid] = (x1,y1,x2,y2) or None — inference-space
        crop_log: list = []  # crop_log[f][oid] = (cx, cy)               — inference-space
        ema_cx = {i: (b[0] + b[2]) / 2.0 for i, b in enumerate(current_bboxes)}
        ema_cy = {i: (b[1] + b[3]) / 2.0 for i, b in enumerate(current_bboxes)}

        ctx = (
            torch.autocast("cuda", dtype=torch.float16)
            if device_str == "cuda"
            else contextlib.nullcontext()
        )

        for chunk_idx in range(n_chunks):
            chunk_start = chunk_idx * CHUNK_FRAMES
            chunk_end   = min(chunk_start + CHUNK_FRAMES, total_frames)
            n_frames    = chunk_end - chunk_start

            jobs[job_id]["chunk"] = f"{chunk_idx + 1}/{n_chunks}"

            # ─ 프레임 로드 및 JPEG 저장 ─────────────────────────────────────
            frames = []
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, chunk_start)
            for _ in range(n_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                if scale < 1.0:
                    frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
                frames.append(frame)
            cap.release()

            if not frames:
                break

            chunk_dir = UPLOADS_DIR / f"{job_id}_chunk{chunk_idx}"
            _write_chunk_jpegs(frames, chunk_dir)

            # ─ SAM2 다중 객체 추적 ──────────────────────────────────────────
            last_masks: dict[int, np.ndarray] = {}
            with inference_lock:
                with torch.inference_mode(), ctx:
                    state = predictor.init_state(
                        str(chunk_dir), offload_video_to_cpu=False
                    )
                    for obj_id, (x1, y1, x2, y2) in enumerate(current_bboxes):
                        predictor.add_new_points_or_box(
                            state, box=(x1, y1, x2, y2), frame_idx=0, obj_id=obj_id
                        )

                    for fid, obj_ids, masks in predictor.propagate_in_video(state):
                        frame_bboxes: list = [None] * n_objs

                        for obj_id, mask in zip(obj_ids, masks):
                            m = mask[0].cpu().numpy() > 0.0
                            last_masks[obj_id] = m

                            bb = _mask_to_bbox(m)
                            frame_bboxes[obj_id] = bb
                            if bb:
                                xmin, ymin, xmax, ymax = bb
                                raw_cx = (xmin + xmax) / 2.0
                                raw_cy = (ymin + ymax) / 2.0
                                ema_cx[obj_id] = CROP_EMA_ALPHA * raw_cx + (1 - CROP_EMA_ALPHA) * ema_cx[obj_id]
                                ema_cy[obj_id] = CROP_EMA_ALPHA * raw_cy + (1 - CROP_EMA_ALPHA) * ema_cy[obj_id]

                        # 좌표만 기록 (인코딩은 추론 완료 후 원본 해상도로 수행)
                        bbox_log.append(frame_bboxes)
                        crop_log.append([(ema_cx[oid], ema_cy[oid]) for oid in range(n_objs)])
                        jobs[job_id]["progress"] = int((chunk_start + fid + 1) / total_frames * 100)

            # ─ 다음 청크 초기 bbox = 이전 청크 마지막 마스크 ────────────────
            new_bboxes = []
            for obj_id, prev in enumerate(current_bboxes):
                bb = _mask_to_bbox(last_masks[obj_id]) if obj_id in last_masks else None
                new_bboxes.append(bb if bb else prev)
            current_bboxes = new_bboxes

            # ─ 임시 파일 정리 ────────────────────────────────────────────────
            del frames
            shutil.rmtree(str(chunk_dir), ignore_errors=True)
            chunk_dir = None
            gc.collect()
            if device_str == "cuda":
                torch.cuda.empty_cache()

        # ── 추론 완료 후 원본 해상도로 영상 조립 ────────────────────────────
        jobs[job_id]["status"] = "cropping"
        main_filename, crop_filenames = _assemble_videos(
            video_path, bbox_log, crop_log,
            fps, job_id, n_objs, scale, src_w, src_h,
        )

        jobs[job_id].update({
            "status":   "done",
            "progress": 100,
            "result":   main_filename,
            "crops":    crop_filenames,
        })

    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)
        traceback.print_exc()
    finally:
        if chunk_dir is not None:
            shutil.rmtree(str(chunk_dir), ignore_errors=True)
        gc.collect()
        if device_str == "cuda":
            torch.cuda.empty_cache()


# ── API routes ─────────────────────────────────────────────────────────────────
@app.get("/api/health")
async def health():
    return {
        "model_loaded": predictor is not None,
        "device": device_str,
        "checkpoint_exists": MODEL_PATH.exists(),
        "busy": inference_lock.locked(),
    }


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower()
    if ext not in {".mp4", ".avi", ".mov", ".mkv"}:
        raise HTTPException(400, "지원하지 않는 파일 형식입니다 (MP4, AVI, MOV, MKV).")

    session_id = str(uuid.uuid4())
    video_path = UPLOADS_DIR / f"{session_id}{ext}"
    video_path.write_bytes(await file.read())

    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    if not ret:
        video_path.unlink(missing_ok=True)
        raise HTTPException(400, "비디오 파일을 읽을 수 없습니다.")

    h, w = frame.shape[:2]
    _, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    frame_b64 = base64.b64encode(jpg.tobytes()).decode()

    return {
        "video_path": str(video_path),
        "width": w,
        "height": h,
        "total_frames": total_frames,
        "fps": round(fps, 2),
        "first_frame": f"data:image/jpeg;base64,{frame_b64}",
    }


class TrackRequest(BaseModel):
    video_path: str
    bboxes: list[list[int]]   # [[x1,y1,x2,y2], ...] — 원본 해상도 기준


@app.post("/api/track")
async def start_tracking(req: TrackRequest):
    if not req.bboxes or any(len(b) != 4 for b in req.bboxes):
        raise HTTPException(400, "bboxes는 [[x1,y1,x2,y2], ...] 형식이어야 합니다.")
    if len(req.bboxes) > 8:
        raise HTTPException(400, "최대 8개 객체까지 지원합니다.")
    if predictor is None:
        raise HTTPException(503, "모델이 로드되지 않았습니다.")
    if inference_lock.locked():
        raise HTTPException(429, "다른 추적 작업이 진행 중입니다. 잠시 후 시도하세요.")

    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "pending", "progress": 0, "chunk": "0/0"}

    threading.Thread(
        target=run_inference,
        args=(job_id, req.video_path, req.bboxes),
        daemon=True,
    ).start()

    return {"job_id": job_id}


@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "작업을 찾을 수 없습니다.")
    return jobs[job_id]


@app.get("/api/crops/{job_id}/{obj_id}")
async def get_crop(job_id: str, obj_id: int):
    if job_id not in jobs or jobs[job_id]["status"] != "done":
        raise HTTPException(400, "결과가 아직 준비되지 않았습니다.")
    crops = jobs[job_id].get("crops", [])
    if obj_id >= len(crops):
        raise HTTPException(404, "해당 객체를 찾을 수 없습니다.")
    fname = crops[obj_id]
    return FileResponse(
        str(OUTPUTS_DIR / fname),
        media_type="video/mp4",
        filename=f"samurai_obj{obj_id}.mp4",
    )


@app.get("/api/result/{job_id}")
async def get_result(job_id: str):
    if job_id not in jobs or jobs[job_id]["status"] != "done":
        raise HTTPException(400, "결과가 아직 준비되지 않았습니다.")
    return FileResponse(
        str(OUTPUTS_DIR / jobs[job_id]["result"]),
        media_type="video/mp4",
        filename="samurai_result.mp4",
    )


# ── Static files (must be last) ────────────────────────────────────────────────
app.mount(
    "/",
    StaticFiles(directory=str(Path(__file__).parent / "static"), html=True),
    name="static",
)
