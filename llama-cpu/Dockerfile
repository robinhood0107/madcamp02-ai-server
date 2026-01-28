FROM python:3.11-slim

WORKDIR /app

# 시스템 의존성 (llama.cpp 빌드용)
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    curl \
    && rm -rf /var/lib/apt/lists/*

# llama-cpp-python 설치
RUN pip install --no-cache-dir llama-cpp-python

# 서버 스크립트
COPY server.py .

EXPOSE 8003

CMD ["python", "server.py", "--host", "0.0.0.0", "--port", "8003"]
