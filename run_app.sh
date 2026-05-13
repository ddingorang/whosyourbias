#!/bin/bash
# SAMURAI Web Demo 실행 스크립트
# 사용법 (repo root에서): bash run_app.sh

set -e
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

echo "========================================"
echo "  SAMURAI Web Demo"
echo "========================================"

# 체크포인트 확인
CKPT="sam2/checkpoints/sam2.1_hiera_base_plus.pt"
if [ ! -f "$CKPT" ]; then
    echo ""
    echo "[경고] 체크포인트 파일이 없습니다."
    echo "  다음 명령으로 다운로드하세요 (~315 MB):"
    echo "    cd sam2/checkpoints && bash download_ckpts.sh && cd ../.."
    echo ""
fi

# 가상환경 활성화
if [ -f ".venv/Scripts/activate" ]; then
    source .venv/Scripts/activate
    echo "[INFO] 가상환경 활성화: .venv"
elif [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "[INFO] 가상환경 활성화: .venv"
else
    echo "[경고] .venv 가상환경을 찾을 수 없습니다."
fi

echo "[INFO] 서버 시작 중..."
echo "[INFO] 브라우저: http://localhost:8000"
echo "[INFO] 종료: Ctrl+C"
echo ""

python -m uvicorn app.server:app \
    --host 0.0.0.0 \
    --port 8000 \
    --reload \
    --reload-dir app
