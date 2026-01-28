"""
llama.cpp 기반 CPU Fallback 서버
"""
import os
import argparse
from llama_cpp import Llama
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# 환경 변수
MODEL_PATH = os.getenv("MODEL_PATH", "/models/tiny-llama-1.1b")
N_THREADS = int(os.getenv("N_THREADS", "8"))

# 모델 로딩
llm = None

def load_model():
    """모델 로딩"""
    global llm
    try:
        # GGUF 파일 찾기
        model_file = None
        if os.path.isdir(MODEL_PATH):
            for file in os.listdir(MODEL_PATH):
                if file.endswith(".gguf"):
                    model_file = os.path.join(MODEL_PATH, file)
                    break
        elif os.path.isfile(MODEL_PATH) and MODEL_PATH.endswith(".gguf"):
            model_file = MODEL_PATH
        else:
            logger.error(f"GGUF 모델 파일을 찾을 수 없습니다: {MODEL_PATH}")
            return False
        
        logger.info(f"모델 로딩 중: {model_file}")
        llm = Llama(
            model_path=model_file,
            n_ctx=2048,
            n_threads=N_THREADS,
            verbose=False
        )
        logger.info("모델 로딩 완료")
        return True
    except Exception as e:
        logger.error(f"모델 로딩 실패: {e}")
        return False


@app.on_event("startup")
async def startup():
    """서버 시작 시 모델 로딩"""
    if not load_model():
        logger.error("모델 로딩 실패로 서버를 시작할 수 없습니다.")


@app.get("/health")
async def health_check():
    """헬스체크"""
    if llm is None:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "reason": "Model not loaded"}
        )
    return {"status": "healthy", "model": MODEL_PATH}


@app.post("/v1/chat/completions")
async def chat_completions(request: dict):
    """OpenAI 호환 채팅 완성 API"""
    if llm is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Model not loaded"}
        )
    
    try:
        messages = request.get("messages", [])
        user_message = messages[-1]["content"] if messages else ""
        
        # 프롬프트 구성
        prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
        
        # 생성
        response = llm(
            prompt,
            max_tokens=request.get("max_tokens", 512),
            temperature=request.get("temperature", 0.7),
            stop=["\n\n"],
            echo=False
        )
        
        content = response["choices"][0]["text"]
        
        return {
            "id": "chatcmpl-cpu",
            "object": "chat.completion",
            "created": int(os.path.getmtime(MODEL_PATH)) if os.path.exists(MODEL_PATH) else 0,
            "model": MODEL_PATH,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(content.split()),
                "total_tokens": len(prompt.split()) + len(content.split())
            }
        }
    except Exception as e:
        logger.error(f"생성 실패: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


if __name__ == "__main__":
    import uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8003)
    args = parser.parse_args()
    
    uvicorn.run(app, host=args.host, port=args.port)
