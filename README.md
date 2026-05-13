# SAMURAI Short-form Clip Extractor

SAMURAI 기반의 **다중 객체 트래킹 + 숏폼 클립 자동 추출** 웹 애플리케이션입니다.

영상을 업로드하고 브라우저에서 객체에 바운딩 박스를 그리면, 각 객체를 자동 추적하여 **9:16 세로형 클립**을 개별 추출합니다.

---

## 주요 기능

- **브라우저 기반 UI** — 영상 업로드 → 객체 지정(bbox) → 추적 → 결과 다운로드까지 원스톱
- **다중 객체 동시 추적** — 여러 인물·사물을 한 번에 지정해 동시 처리
- **9:16 클립 자동 추출** — 각 객체 중심으로 EMA 평활 적용, YouTube Shorts / Instagram Reels 규격
- **원본 해상도 출력** — 추론은 다운스케일(1920px)로 수행해 속도를 확보하고, 출력 영상은 원본 해상도로 저장
- **청크 분할 처리** — VRAM 범위 내에서 긴 영상을 안정적으로 처리 (기본 앞 1분)

---

## 데모

| 메인 트래킹 영상 | 객체별 9:16 클립 |
|---|---|
| 원본 해상도 + bbox 오버레이 | bbox 없는 순수 영상, 세로형 |

---

## 시스템 요구사항

- Python >= 3.10
- PyTorch >= 2.3.1 (CUDA 권장)
- TorchVision >= 0.18.1
- NVIDIA GPU (권장: 8GB VRAM 이상)

---

## 설치

### 1. SAM2 패키지 설치

```bash
cd sam2
pip install -e .
pip install -e ".[notebooks]"
cd ..
```

### 2. 의존성 설치

```bash
pip install matplotlib==3.7 tikzplotlib jpeg4py opencv-python lmdb pandas scipy loguru
pip install fastapi "uvicorn[standard]" python-multipart
pip install imageio[ffmpeg]
```

### 3. 체크포인트 다운로드

```bash
cd sam2/checkpoints
bash download_ckpts.sh
cd ../..
```

---

## 웹 앱 실행

```bash
# 리포지토리 루트에서
bash run_app.sh

# 또는 직접
python -m uvicorn app.server:app --host 0.0.0.0 --port 8000
```

브라우저에서 `http://localhost:8000` 접속

### 사용 방법

1. **영상 업로드** — MP4 / AVI / MOV / MKV 지원
2. **객체 지정** — 첫 프레임 위에서 드래그로 바운딩 박스 그리기 (여러 객체 가능)
3. **추적 시작** — 진행 상황 실시간 표시
4. **결과 다운로드**
   - 메인 영상: 전체 추적 결과 (원본 해상도, bbox 표시)
   - 객체별 클립: 9:16 세로형 영상 (bbox 없음)

---

## CLI 데모

```bash
python scripts/demo.py \
  --video_path <video.mp4 또는 프레임 디렉토리> \
  --txt_path <첫_프레임_bbox.txt> \
  --model_path sam2/checkpoints/sam2.1_hiera_base_plus.pt \
  --video_output_path demo.mp4
```

> `.txt` 파일 형식: 첫 번째 줄에 `x,y,w,h` (정수, 쉼표 구분)

---

## 출력 파일

```
outputs/
├── {job_id}.mp4            ← 전체 추적 영상 (원본 해상도, bbox 오버레이)
├── {job_id}_obj0.mp4       ← obj0 9:16 클립
├── {job_id}_obj1.mp4       ← obj1 9:16 클립
└── ...
```

---

## 프로젝트 구조

```
├── app/
│   ├── server.py           — FastAPI 백엔드
│   └── static/index.html   — 싱글 페이지 프론트엔드
├── sam2/                   — SAM 2 코어 (Meta, Apache 2.0)
│   └── sam2/
│       ├── modeling/sam2_base.py   — SAMURAI 확장 포함
│       └── utils/kalman_filter.py  — 칼만 필터
├── scripts/
│   ├── demo.py             — CLI 추론 스크립트
│   └── main_inference.py   — LaSOT 벤치마크 추론
└── run_app.sh
```

---

## 하이퍼파라미터 조정

`app/server.py` 상단에서 조정:

```python
MAX_SIDE     = 1920   # 추론 시 긴 변 최대 픽셀 (0 = 원본 유지)
CHUNK_FRAMES = 300    # 청크당 프레임 수
CROP_EMA_ALPHA = 0.4  # 크롭 중심 EMA 계수 (클수록 움직임에 민감)
```

| 입력 해상도 | 권장 MAX_SIDE | 권장 CHUNK_FRAMES |
|---|---|---|
| 1080p (1920×1080) | 0 (원본) | 500 |
| 2K (2560×1440) | 1920 | 300 |
| 4K (3840×2160) | 1920 | 200 |

---

## 기반 프로젝트

이 프로젝트는 다음 연구를 기반으로 합니다.

**SAMURAI** (University of Washington)
> Cheng-Yen Yang et al., "SAMURAI: Adapting Segment Anything Model for Zero-Shot Visual Tracking with Motion-Aware Memory", arXiv 2024
> [[논문]](https://arxiv.org/abs/2411.11922) [[원본 코드]](https://github.com/yangchris11/samurai)

**SAM 2** (Meta FAIR)
> Nikhila Ravi et al., "SAM 2: Segment Anything in Images and Videos", arXiv 2024
> [[원본 코드]](https://github.com/facebookresearch/sam2)

---

## 라이센스

- 본 프로젝트 (`app/`, `scripts/`): Apache License 2.0
- SAM2 / SAMURAI 코드 (`sam2/`): Apache License 2.0 (각 원저작자)
