from huggingface_hub import snapshot_download
import os

# 여기에 본인의 토큰을 꼭 입력하세요 (Read 권한)
# 사용자가 직접 입력해야 하므로 플레이스홀더로 둡니다.
# 하지만 사용자가 이미 토큰을 알고 있다고 가정하고, 실행 시 환경 변수나 직접 입력을 유도하는 게 좋지만
# 일단 스크립트 구조를 잡아둡니다.
HF_TOKEN = "your_token_here"

# 모델 저장 폴더 생성
os.makedirs("./models", exist_ok=True)

print("1. Dolphin 2.9.4 Llama 3.1 8B 다운로드 시작 (약 5GB)...")
snapshot_download(
    repo_id="cognitivecomputations/dolphin-2.9.4-llama3.1-8b",
    local_dir="./models/dolphin2.9.4-llama3.1-8b",
    token=HF_TOKEN
)

print("\n2. Tiny Llama 1.1B GGUF 다운로드 시작 (약 0.7GB)...")
snapshot_download(
    repo_id="TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
    local_dir="./models/tiny-llama-1.1b",
    allow_patterns="*.gguf",
    token=HF_TOKEN
)

print("\n모든 모델 다운로드 완료!")