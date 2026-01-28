FROM nvcr.io/nvidia/pytorch:24.01-py3

WORKDIR /app

# vLLM 설치
RUN pip install --no-cache-dir vllm

# curl 설치 (헬스체크용)
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 모델 경로는 볼륨 마운트로 제공
ENV MODEL_PATH=/models/gpt-oss-20b

EXPOSE 8004

# vLLM OpenAI API 서버 시작 (4bit quantization 권장)
CMD ["python", "-m", "vllm.entrypoints.openai.api_server", \
     "--model", "${MODEL_PATH}", \
     "--host", "0.0.0.0", \
     "--port", "8004", \
     "--tensor-parallel-size", "1", \
     "--gpu-memory-utilization", "0.90", \
     "--quantization", "awq"]
