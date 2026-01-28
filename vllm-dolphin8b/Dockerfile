FROM nvcr.io/nvidia/pytorch:24.01-py3

WORKDIR /app

# vLLM 설치
RUN pip install --no-cache-dir vllm

# curl 설치 (헬스체크용)
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 모델 경로는 볼륨 마운트로 제공
ENV MODEL_PATH=/models/dolphin2.9.4-llama3.1-8b
ENV LORA_ADAPTERS_DIR=/adapters

EXPOSE 8002

# vLLM OpenAI API 서버 시작 (LoRA 어댑터 지원)
# LoRA 어댑터는 런타임에 동적으로 로드됨
# 어댑터는 --lora-modules로 미리 등록하거나, API 요청 시 lora_id로 지정
# 환경 변수 치환을 위해 shell 형식 사용
CMD python -m vllm.entrypoints.openai.api_server \
    --model ${MODEL_PATH} \
    --host 0.0.0.0 \
    --port 8002 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.85 \
    --enable-lora \
    --lora-modules "sage=${LORA_ADAPTERS_DIR}/persona-sage-lora,analyst=${LORA_ADAPTERS_DIR}/persona-analyst-lora,friend=${LORA_ADAPTERS_DIR}/persona-friend-lora"
