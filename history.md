# 변경 이력

---

## 2026-05-19

### 1. 모델 교체 및 추론 속도 최적화

**파일:** `sam2/sam2/configs/samurai/sam2.1_hiera_s.yaml`
**체크포인트:** `sam2/checkpoints/`

**배경:** 정밀한 세그멘테이션 마스크가 불필요하고(bbox 추출만 사용), 추론 속도 단축이 목표.

**모델 교체:**

| 항목 | 변경 전 | 변경 후 |
|---|---|---|
| 체크포인트 | `sam2.1_hiera_base_plus.pt` (309MB) | `sam2.1_hiera_small.pt` (176MB) |
| config | `sam2.1_hiera_b+.yaml` | `sam2.1_hiera_s.yaml` |

**하이퍼파라미터 변경 (`sam2.1_hiera_s.yaml`):**

| 파라미터 | 변경 전 | 변경 후 | 이유 |
|---|---|---|---|
| `image_size` | 1024 | 512 | bbox 추출에 고해상도 불필요, 메모리 어텐션 연산 4배 감소 |
| `use_high_res_features_in_sam` | true | false | 마스크 경계 정밀도 전용, bbox 추출에 불필요 (FPN 피처 레벨 3→1) |
| `multimask_output_in_sam` | true | false | 3개 마스크 후보 생성 불필요 |
| `multimask_output_for_tracking` | true | false | 동일 |
| `use_multimask_token_for_obj_ptr` | true | false | multimask 비활성화에 따른 연동 설정 |
| `num_maskmem` | 7 | 4 | 메모리 뱅크 크기 축소, 메모리 어텐션 연산 감소 |
| `max_obj_ptrs_in_encoder` | 16 (기본값) | 8 | object pointer 어텐션 토큰 수 감소 |
| `compile_image_encoder` | False | True | torch.compile으로 이미지 인코더 20~40% 속도 향상 |

**유지한 설정 (추적 잠금 핵심):**
- `kf_score_weight: 0.25` — 칼만 필터 기반 모션 예측
- `stable_frames_threshold: 15` — KF 안정화 임계값
- `memory_bank_iou_threshold: 0.5` — 메모리 뱅크 품질 필터
- `pred_obj_scores: true` — 객체 소실 감지

**실행 명령:**
```bash
python scripts/demo.py \
  --video_path <video.mp4> \
  --txt_path <bbox.txt> \
  --model_path sam2/checkpoints/sam2.1_hiera_small.pt \
  --video_output_path demo.mp4
```

---

## 2026-05-12

### 1. 입력 영상 처리 범위 제한 — 앞 1분만 처리

**파일:** `app/server.py`, `scripts/demo.py`

**배경:** 전체 영상을 처리할 경우 소요 시간이 과도하게 길어지는 문제.

**변경 내용:**

- `app/server.py` — `total_frames`를 `fps × 60`으로 cap
  ```python
  max_frames = int(fps * 60)
  total_frames = min(total_frames, max_frames)
  ```
- `scripts/demo.py` — 프레임 로드 루프에 `max_frames` 상한 추가
  ```python
  max_frames = int(frame_rate * 60)
  while len(loaded_frames) < max_frames:
      ...
  ```

---

### 2. 객체별 9:16 크롭 영상 추출

**파일:** `app/server.py`, `app/static/index.html`

**배경:** 추적 결과에서 각 객체를 중심으로 한 세로형(숏폼) 영상을 별도 추출하고 싶다는 요구.

**설계 결정:**
- 크롭 중심: bbox 중심에 EMA(지수 이동 평균, α=0.4) 적용 → 빠른 움직임 추종 + 노이즈 억제
- 크롭 크기: 고정 9:16 (`crop_h = out_h`, `crop_w = out_h × 9/16`)
- 경계 처리: 화면 밖으로 나가는 경우 clamp
- 미감지 프레임: 마지막 EMA 중심 유지 (튐 없음)

**변경 내용 (`app/server.py`):**

| 항목 | 내용 |
|---|---|
| `CROP_EMA_ALPHA = 0.4` | 전역 상수 추가 |
| `_crop_9x16()` | EMA 중심 기반 9:16 크롭 헬퍼 함수 추가 |
| `crop_writers` 초기화 | `run_inference` 내 객체 수만큼 `VideoWriter` 생성 |
| EMA 갱신 | 프레임 루프 내 `if bb:` 블록에서 `ema_cx/cy` 업데이트 |
| 크롭 쓰기 | `out_writer.write(img)` 직후 각 객체별 크롭 프레임 기록 |
| 완료 처리 | `crops` 파일명 목록을 job 결과에 포함 |
| `finally` 정리 | `crop_writers` 해제 추가 |
| `GET /api/crops/{job_id}/{obj_id}` | 객체별 크롭 영상 다운로드 엔드포인트 추가 |

**변경 내용 (`app/static/index.html`):**

- 결과 화면 하단에 **"객체별 클립 (9:16)"** 다운로드 섹션 추가
- `showResult()` — 완료 후 `/api/status/{job_id}`에서 `crops` 목록을 읽어 객체별 색상 점 + 다운로드 버튼 자동 렌더링

**출력 파일 규칙:**
```
{job_id}.mp4            ← 전체 추적 결과 (기존)
{job_id}_obj0.mp4       ← obj0 9:16 크롭
{job_id}_obj1.mp4       ← obj1 9:16 크롭
...
```

**크롭 해상도 예시 (1920×1080 입력 기준):** 606×1080

---

### 3. 크롭 영상 인코딩 방식 개선 — 추론·인코딩 분리 + portrait 정상 출력

**파일:** `app/server.py`

**배경:** 추론 루프 내 VideoWriter 호출 시 두 가지 문제 발생.
1. Windows OpenCV VideoWriter가 portrait(세로) 해상도에서 `Unknown C++ exception from OpenCV code` 발생
2. 추론 중 MJPG 인코딩이 CPU를 점유해 전체 소요 시간 증가

**해결 과정:**

| 시도 | 내용 | 결과 |
|---|---|---|
| `mp4v` portrait | 기본 코덱으로 세로 VideoWriter | C++ 예외 |
| `XVID` `.avi` | 폴백 코덱 | C++ 예외 동일 |
| `MJPG` `.avi` | 가장 범용적인 코덱 | C++ 예외 동일 |
| `np.ascontiguousarray` | 메모리 연속성 보장 | 해결 안 됨 |
| 90° 회전 저장 (landscape) | portrait → landscape 변환 후 `mp4v` 저장 | 인코딩 성공, 단 영상 누워서 출력 |
| **`imageio[ffmpeg]` 도입** | 내장 ffmpeg로 portrait 직접 인코딩 | **정상 해결** |

**최종 구조 (`_assemble_crop_videos`):**

- 추론 중: EMA 중심 좌표만 `crop_log`에 기록 (float 2개/프레임/객체, GPU 연산 방해 없음)
- 추론 완료 후: 원본 영상 재독 → 크롭 → `imageio` + `libx264`로 portrait `.mp4` 인코딩
- 프론트엔드: `cropping` 상태 표시 추가 ("크롭 영상 조립 중...")

**설치 추가:**
```bash
pip install imageio[ffmpeg]
```

**인코딩 파라미터:**
```python
imageio.get_writer(path, fps=fps, codec="libx264", pixelformat="yuv420p",
                   output_params=["-crf", "23"])
```
