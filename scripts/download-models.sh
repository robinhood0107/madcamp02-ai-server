#!/bin/bash
set -e

MODELS_DIR="${1:-./models}"

echo "모델 다운로드 디렉토리: $MODELS_DIR"
mkdir -p "$MODELS_DIR"

# HuggingFace CLI 설치 확인
if ! command -v huggingface-cli &> /dev/null; then
    echo "huggingface-cli가 설치되어 있지 않습니다."
    echo "다음 명령으로 설치하세요: pip install huggingface_hub[cli]"
    exit 1
fi

# Dolphin 2.9.4 Llama 3.1 8B (기본 모델, 페르소나 지원)
echo "=========================================="
echo "Downloading Dolphin 2.9.4 Llama 3.1 8B..."
echo "=========================================="
huggingface-cli download cognitivecomputations/dolphin-2.9.4-llama3.1-8b \
    --local-dir "$MODELS_DIR/dolphin2.9.4-llama3.1-8b" \
    --local-dir-use-symlinks False || {
    echo "⚠️ Dolphin 2.9.4 Llama 3.1 8B 다운로드 실패"
}

# Tiny Llama 1.1B (CPU Fallback) - GGUF 형식 필요
echo "=========================================="
echo "Downloading Tiny Llama 1.1B (GGUF)..."
echo "=========================================="
# GGUF 파일은 별도로 다운로드 필요 (예: HuggingFace에서 GGUF 변환된 모델)
# 또는 llama.cpp로 변환 필요
huggingface-cli download TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF \
    --local-dir "$MODELS_DIR/tiny-llama-1.1b" \
    --local-dir-use-symlinks False \
    --include "*.gguf" || {
    echo "⚠️ Tiny Llama 1.1B GGUF 다운로드 실패"
    echo "대안: https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF 에서 수동 다운로드"
}

# GPT-20B (선택, 큰 용량 주의)
# echo "=========================================="
# echo "Downloading GPT-OSS-20B..."
# echo "=========================================="
# huggingface-cli download <model-path> \
#     --local-dir "$MODELS_DIR/gpt-oss-20b" \
#     --local-dir-use-symlinks False

echo "=========================================="
echo "모델 다운로드 완료!"
echo "=========================================="
echo "다운로드된 모델 위치: $MODELS_DIR"
ls -lh "$MODELS_DIR"
