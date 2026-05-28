"""
SAMURAI Web Demo — FastAPI backend
Run from repo root:
    uvicorn app.server:app --host 0.0.0.0 --port 8000
"""

import contextlib
import gc
import math
import shutil
import subprocess
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

# 미리보기용 VideoCapture 캐시 — 요청마다 파일을 다시 열지 않음
_preview_caps: dict[str, cv2.VideoCapture] = {}
_preview_lock = threading.Lock()

MODEL_PATH = ROOT / "sam2" / "checkpoints" / "sam2.1_hiera_small.pt"
MODEL_CFG  = "configs/samurai/sam2.1_hiera_s.yaml"

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
FRAME_SKIP     = 2    # SAM2에 넘길 프레임 샘플링 간격 (1=전체, 2=절반, 3=1/3 …)


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


# ── FFmpeg pipe writer ─────────────────────────────────────────────────────────
def _open_ffmpeg_writer(out_path, fps: float, w: int, h: int) -> subprocess.Popen:
    """bgr24 raw 프레임을 stdin으로 받아 libx264(ultrafast) MP4를 출력하는 ffmpeg 프로세스."""
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
        ff = get_ffmpeg_exe()
    except Exception:
        ff = "ffmpeg"
    return subprocess.Popen(
        [
            ff, "-y",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{w}x{h}", "-pix_fmt", "bgr24", "-r", str(int(fps)),
            "-i", "pipe:0",
            "-c:v", "libx264", "-preset", "ultrafast",
            "-crf", "23", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(out_path),
        ],
        stdin=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )


# ── Post-inference assembly (main + crop) ──────────────────────────────────────
def _assemble_videos(
    video_path: str,
    bbox_log: list,   # bbox_log[f][oid] = (x1,y1,x2,y2) or None  — inference-space coords
    crop_log: list,   # crop_log[f][oid] = (cx, cy)                — inference-space coords
    fps: float, job_id: str, n_objs: int,
    scale: float, src_w: int, src_h: int,
    start_frame: int = 0,
) -> tuple[str, list[str]]:
    """추론 완료 후 원본 영상을 1회 재독해 메인·크롭 영상을 원본 해상도로 생성한다."""
    crop_h = src_h
    crop_w = int(src_h * 9 / 16) & ~1

    main_filename  = f"{job_id}.mp4"
    crop_filenames = [f"{job_id}_obj{i}.mp4" for i in range(n_objs)]

    main_proc  = _open_ffmpeg_writer(OUTPUTS_DIR / main_filename,  fps, src_w, src_h)
    crop_procs = [
        _open_ffmpeg_writer(OUTPUTS_DIR / crop_filenames[i], fps, crop_w, crop_h)
        for i in range(n_objs)
    ]

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for bboxes, centers in zip(bbox_log, crop_log):
        ret, frame = cap.read()
        if not ret:
            break

        # 크롭 영상: BGR raw 그대로 pipe (색 변환 불필요)
        for oid, (cx_inf, cy_inf) in enumerate(centers):
            crop = _crop_9x16(frame, cx_inf / scale, cy_inf / scale, crop_w, crop_h)
            crop_procs[oid].stdin.write(crop.tobytes())

        # 메인 영상: bbox 오버레이 후 BGR raw pipe
        for oid, bb in enumerate(bboxes):
            if bb is None:
                continue
            x1 = int(bb[0] / scale); y1 = int(bb[1] / scale)
            x2 = int(bb[2] / scale); y2 = int(bb[3] / scale)
            color = COLORS_BGR[oid % len(COLORS_BGR)]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"obj{oid}", (x1, max(y1 - 6, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
        main_proc.stdin.write(frame.tobytes())

    cap.release()
    main_proc.stdin.close();  main_proc.wait()
    for p in crop_procs:
        p.stdin.close();  p.wait()

    return main_filename, crop_filenames


# ── Inference worker ───────────────────────────────────────────────────────────
def run_inference(job_id: str, video_path: str, bboxes: list[list[int]], start_frame: int = 0):
    """양방향 청크 추적: start_frame 이전은 역방향, 이후는 순방향으로 처리 후 시간순 결합."""
    chunk_dir = None
    try:
        jobs[job_id]["status"] = "running"

        if predictor is None:
            raise RuntimeError("모델이 로드되지 않았습니다. 체크포인트를 다운로드하세요.")

        # ── 영상 메타 ──────────────────────────────────────────────────────────
        cap = cv2.VideoCapture(video_path)
        fps          = cap.get(cv2.CAP_PROP_FPS) or 30
        src_w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        scale, out_w, out_h = _scale_params(src_w, src_h)
        n_objs = len(bboxes)

        init_bboxes = [
            (int(b[0]*scale), int(b[1]*scale), int(b[2]*scale), int(b[3]*scale))
            for b in bboxes
        ]

        ctx = (
            torch.autocast("cuda", dtype=torch.float16)
            if device_str == "cuda"
            else contextlib.nullcontext()
        )

        # ── 공통 헬퍼: 프레임 리스트 → SAM2 추적 + 보간 ──────────────────────
        def _track_and_interpolate(frames_list, anchor_bboxes, tag):
            """frames_list[0]을 앵커로 SAM2 추적 후 전체 프레임 bbox 리스트 반환."""
            nonlocal chunk_dir
            chunk_dir = UPLOADS_DIR / f"{job_id}_{tag}"
            _write_chunk_jpegs(frames_list[::FRAME_SKIP], chunk_dir)

            last_masks: dict[int, np.ndarray] = {}
            sampled: list = []

            with inference_lock:
                with torch.inference_mode(), ctx:
                    state = predictor.init_state(str(chunk_dir), offload_video_to_cpu=False)
                    for obj_id, (x1, y1, x2, y2) in enumerate(anchor_bboxes):
                        predictor.add_new_points_or_box(
                            state, box=(x1, y1, x2, y2), frame_idx=0, obj_id=obj_id
                        )
                    for _, obj_ids, masks in predictor.propagate_in_video(state):
                        fb = [None] * n_objs
                        for obj_id, mask in zip(obj_ids, masks):
                            m = mask[0].cpu().numpy() > 0.0
                            last_masks[obj_id] = m
                            fb[obj_id] = _mask_to_bbox(m)
                        sampled.append(fb)

            shutil.rmtree(str(chunk_dir), ignore_errors=True)
            chunk_dir = None

            # 선형 보간: 샘플 → 전체 프레임
            n_s = len(sampled)
            result = []
            for f in range(len(frames_list)):
                s_lo = min(f // FRAME_SKIP, n_s - 1)
                s_hi = min(s_lo + 1, n_s - 1)
                alpha = (f % FRAME_SKIP) / FRAME_SKIP if s_lo < s_hi else 0.0
                fb = [None] * n_objs
                for oid in range(n_objs):
                    bl, bh = sampled[s_lo][oid], sampled[s_hi][oid]
                    if bl is None:                bb = bh
                    elif bh is None or alpha == 0.0: bb = bl
                    else: bb = tuple(int(bl[i]*(1-alpha) + bh[i]*alpha) for i in range(4))
                    fb[oid] = bb
                result.append(fb)

            return result, last_masks

        def _advance_bboxes(current, last_masks):
            """다음 청크 앵커 bbox: 마지막 마스크 → bbox, 실패 시 이전 값 유지."""
            nxt = []
            for oid in range(n_objs):
                bb = _mask_to_bbox(last_masks[oid]) if oid in last_masks else None
                nxt.append(bb if bb is not None else current[oid])
            return nxt

        # ════════════════════════════════════════════════════════════════
        # Phase 1  역방향: start_frame-1 → 0
        # ════════════════════════════════════════════════════════════════
        back_bbox_log: list = []   # 시간순으로 채워짐 (frame 0 ~ start_frame-1)

        if start_frame > 0:
            n_back = math.ceil(start_frame / CHUNK_FRAMES)
            back_bboxes = list(init_bboxes)

            for bi in range(n_back):
                t_end   = start_frame - bi * CHUNK_FRAMES      # 시간상 끝 (exclusive)
                t_start = max(0, t_end - CHUNK_FRAMES)         # 시간상 시작 (inclusive)

                # 순차 로드 후 역순으로 뒤집어 SAM2에 넘김
                # → SAM2 입장에서 index 0 = 시간상 최신 프레임 (앵커에 가장 가까운 프레임)
                cap = cv2.VideoCapture(video_path)
                cap.set(cv2.CAP_PROP_POS_FRAMES, t_start)
                frames_fwd = []
                for _ in range(t_end - t_start):
                    ret, frm = cap.read()
                    if not ret: break
                    if scale < 1.0:
                        frm = cv2.resize(frm, (out_w, out_h), interpolation=cv2.INTER_AREA)
                    frames_fwd.append(frm)
                cap.release()

                if not frames_fwd: break

                frames_rev = list(reversed(frames_fwd))

                chunk_bboxes_rev, last_masks = _track_and_interpolate(
                    frames_rev, back_bboxes, f"back{bi}"
                )

                # chunk_bboxes_rev[0] = t_end-1, [1] = t_end-2, …
                # 뒤집으면 시간순 [t_start, …, t_end-1] → 맨 앞에 삽입
                back_bbox_log = list(reversed(chunk_bboxes_rev)) + back_bbox_log

                jobs[job_id]["progress"] = int((bi + 1) * CHUNK_FRAMES / total_frames * 50)
                jobs[job_id]["chunk"] = f"역방향 {bi + 1}/{n_back}"

                back_bboxes = _advance_bboxes(back_bboxes, last_masks)

                del frames_fwd, frames_rev
                gc.collect()
                if device_str == "cuda":
                    torch.cuda.empty_cache()

        # ════════════════════════════════════════════════════════════════
        # Phase 2  순방향: start_frame → total_frames-1
        # ════════════════════════════════════════════════════════════════
        fwd_frames   = total_frames - start_frame
        n_fwd        = math.ceil(fwd_frames / CHUNK_FRAMES)
        cur_bboxes   = list(init_bboxes)
        fwd_bbox_log: list = []

        for ci in range(n_fwd):
            c_start = start_frame + ci * CHUNK_FRAMES
            c_end   = min(c_start + CHUNK_FRAMES, total_frames)

            jobs[job_id]["chunk"] = f"순방향 {ci + 1}/{n_fwd}"

            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, c_start)
            frames = []
            for _ in range(c_end - c_start):
                ret, frm = cap.read()
                if not ret: break
                if scale < 1.0:
                    frm = cv2.resize(frm, (out_w, out_h), interpolation=cv2.INTER_AREA)
                frames.append(frm)
            cap.release()

            if not frames: break

            chunk_bboxes, last_masks = _track_and_interpolate(frames, cur_bboxes, f"fwd{ci}")
            fwd_bbox_log.extend(chunk_bboxes)

            base = 50 if start_frame > 0 else 0
            span = 50 if start_frame > 0 else 100
            jobs[job_id]["progress"] = base + int(
                (ci * CHUNK_FRAMES + len(frames)) / fwd_frames * span
            )

            cur_bboxes = _advance_bboxes(cur_bboxes, last_masks)

            del frames
            gc.collect()
            if device_str == "cuda":
                torch.cuda.empty_cache()

        # ════════════════════════════════════════════════════════════════
        # 결합 · EMA(시간순 1회 통과) · 영상 조립
        # ════════════════════════════════════════════════════════════════
        bbox_log = back_bbox_log + fwd_bbox_log

        ema_cx = {i: (b[0]+b[2])/2.0 for i, b in enumerate(init_bboxes)}
        ema_cy = {i: (b[1]+b[3])/2.0 for i, b in enumerate(init_bboxes)}
        crop_log = []
        for fb in bbox_log:
            row = []
            for oid in range(n_objs):
                bb = fb[oid]
                if bb:
                    cx = (bb[0]+bb[2])/2.0
                    cy = (bb[1]+bb[3])/2.0
                    ema_cx[oid] = CROP_EMA_ALPHA * cx + (1 - CROP_EMA_ALPHA) * ema_cx[oid]
                    ema_cy[oid] = CROP_EMA_ALPHA * cy + (1 - CROP_EMA_ALPHA) * ema_cy[oid]
                row.append((ema_cx[oid], ema_cy[oid]))
            crop_log.append(row)

        jobs[job_id]["status"] = "cropping"
        main_filename, crop_filenames = _assemble_videos(
            video_path, bbox_log, crop_log,
            fps, job_id, n_objs, scale, src_w, src_h,
            start_frame=0,   # 역방향 포함 → 항상 프레임 0부터 조립
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


@app.get("/api/frame")
async def get_frame(video_path: str, frame_idx: int = 0):
    with _preview_lock:
        if video_path not in _preview_caps or not _preview_caps[video_path].isOpened():
            _preview_caps.get(video_path, None) and _preview_caps[video_path].release()
            _preview_caps[video_path] = cv2.VideoCapture(video_path)
        cap = _preview_caps[video_path]
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

    if not ret:
        raise HTTPException(400, "해당 프레임을 읽을 수 없습니다.")

    # 미리보기용 해상도 축소 (전송량 감소)
    h, w = frame.shape[:2]
    if w > 1280:
        scale = 1280 / w
        frame = cv2.resize(frame, (int(w * scale) & ~1, int(h * scale) & ~1), interpolation=cv2.INTER_AREA)

    _, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return {"frame": f"data:image/jpeg;base64,{base64.b64encode(jpg.tobytes()).decode()}"}


class TrackRequest(BaseModel):
    video_path: str
    bboxes: list[list[int]]   # [[x1,y1,x2,y2], ...] — 원본 해상도 기준
    start_frame: int = 0      # bbox를 그린 프레임 인덱스


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
        args=(job_id, req.video_path, req.bboxes, req.start_frame),
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
