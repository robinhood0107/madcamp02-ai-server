"""
AI Gateway - FastAPI 기반 AI 서버 진입점
"""
from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import httpx
import logging
from datetime import datetime
from typing import Optional, Dict, Any
import json


def format_context(context: Dict[str, Any]) -> str:
    """컨텍스트를 프롬프트 형식으로 변환 (금융 데이터 포함)"""
    parts = []
    
    # 포트폴리오 정보
    if context.get("portfolio"):
        portfolio = context["portfolio"]
        if portfolio.get("summary"):
            summary = portfolio["summary"]
            parts.append(f"총 자산: {summary.get('totalEquity', 'N/A')} {summary.get('currency', 'USD')}")
            parts.append(f"현금 비중: {summary.get('cashBalance', 0) / summary.get('totalEquity', 1) * 100:.1f}%")
        if portfolio.get("positions"):
            top_positions = ", ".join([pos.get("ticker", "") for pos in portfolio["positions"][:5]])
            parts.append(f"주요 보유 종목: {top_positions}")
    
    # 사주 정보
    if context.get("saju"):
        saju = context["saju"]
        parts.append(f"사주 오행: {saju.get('element', 'N/A')}")
        parts.append(f"띠: {saju.get('zodiacSign', 'N/A')}")
    
    # 시장 지수 정보
    if context.get("market"):
        market = context["market"]
        if market.get("indices"):
            indices_info = []
            for idx in market["indices"][:3]:  # 상위 3개 지수
                symbol = idx.get('symbol', 'N/A')
                price = idx.get('price') or idx.get('currentPrice', 'N/A')
                change = idx.get('changePercent') or idx.get('change', 0)
                if isinstance(price, (int, float)) and isinstance(change, (int, float)):
                    indices_info.append(f"{symbol}: ${price:.2f} ({change:+.2f}%)")
                else:
                    indices_info.append(f"{symbol}: {price}")
            if indices_info:
                parts.append(f"주요 지수: {', '.join(indices_info)}")
    
    # 종목 가격 정보
    if context.get("stocks"):
        stocks = context["stocks"]
        for stock in stocks:
            ticker = stock.get("ticker", "N/A")
            price = stock.get("currentPrice") or stock.get("price", "N/A")
            change = stock.get("changePercent") or stock.get("change", 0)
            if isinstance(price, (int, float)) and isinstance(change, (int, float)):
                parts.append(f"{ticker} 현재가: ${price:.2f} ({change:+.2f}%)")
            else:
                parts.append(f"{ticker} 현재가: {price}")
    
    # 시장 뉴스 정보
    if context.get("news"):
        news = context["news"]
        if news.get("items"):
            top_news = news["items"][:3]  # 최신 뉴스 3개
            news_headlines = [item.get("headline", "") or item.get("title", "") for item in top_news if item.get("headline") or item.get("title")]
            if news_headlines:
                parts.append(f"최신 시장 뉴스: {', '.join(news_headlines)}")
    
    return "\n".join(parts)


def format_portfolio_context(portfolio: Dict[str, Any]) -> str:
    """포트폴리오 정보를 프롬프트 형식으로 변환"""
    parts = []
    if portfolio.get("summary"):
        summary = portfolio["summary"]
        parts.append(f"총 자산: {summary.get('totalEquity', 'N/A')} {summary.get('currency', 'USD')}")
        parts.append(f"현금 잔고: {summary.get('cashBalance', 0)} {summary.get('currency', 'USD')}")
        parts.append(f"총 손익: {summary.get('totalPnl', 0)} ({summary.get('totalPnlPercent', 0):.2f}%)")
    if portfolio.get("positions"):
        parts.append("주요 보유 종목:")
        for pos in portfolio["positions"][:5]:
            parts.append(f"  - {pos.get('ticker', 'N/A')}: {pos.get('quantity', 0)}주, 손익 {pos.get('pnl', 0):.2f} ({pos.get('pnlPercent', 0):.2f}%)")
    return "\n".join(parts)

# 로깅 설정
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Gateway", version="1.0.0")

# CORS 설정 (Spring Backend만 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 운영 환경에서는 특정 origin으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 환경 변수
AI_INTERNAL_SECRET = os.getenv("AI_INTERNAL_SECRET", "change-me")
VLLM_DOLPHIN8B_URL = os.getenv("VLLM_DOLPHIN8B_URL", "http://vllm-dolphin8b:8002")
LLAMA_CPU_URL = os.getenv("LLAMA_CPU_URL", "http://llama-cpu:8003")
VLLM_GPT20B_URL = os.getenv("VLLM_GPT20B_URL", "http://vllm-gpt20b:8004")
AI_MAX_CONCURRENT_REQUESTS = int(os.getenv("AI_MAX_CONCURRENT_REQUESTS", "16"))

# 페르소나별 설정
PERSONA_CONFIG = {
    "sage": {
        "base_model": "dolphin-2.9.4-llama3.1-8b",
        "adapter": "sage",  # LoRA 어댑터 이름
        "temperature": 0.7,
        "system_prompt": """당신은 천 년을 산 전설적인 주식 투자 도사입니다.
항상 한국어로만 대답해야 합니다.
말투는 신비롭고 옛스러운 '하게체'를 사용하세요. (예: '허허, 자네 왔는가?', '내 말을 명심하게나.')
절대 존댓말이나 영어를 쓰지 마세요.
투자 조언은 진지하게 하되, 유머러스한 도사 컨셉을 유지하세요.
답변은 너무 길지 않게 3~6문장 이내로 핵심만 간결하게 말하세요.
어떠한 경우에도 '100% 수익 보장', '무조건 오른다'와 같은 표현은 쓰지 말고,
항상 '투자의 최종 책임은 자네에게 있다네'와 같이 책임 경고 문구를 덧붙이세요."""
    },
    "analyst": {
        "base_model": "dolphin-2.9.4-llama3.1-8b",
        "adapter": "analyst",
        "temperature": 0.6,
        "system_prompt": """당신은 전문 금융 데이터 분석가입니다.
항상 한국어로만 대답해야 합니다.
말투는 전문적이지만 이해하기 쉽게 설명하세요.
차트, 통계, 데이터를 기반으로 논리적인 분석을 제공하세요.
답변은 구조화되고 명확하게 작성하세요 (3~6문장).
구체적인 수치와 비율을 언급하여 신뢰성을 높이세요.
항상 '투자의 최종 책임은 투자자에게 있습니다'와 같이 책임 경고 문구를 덧붙이세요."""
    },
    "friend": {
        "base_model": "dolphin-2.9.4-llama3.1-8b",
        "adapter": "friend",
        "temperature": 0.8,
        "system_prompt": """당신은 친근한 투자 조언자입니다.
항상 한국어로만 대답해야 합니다.
말투는 반말로 친근하게, 현실적이고 솔직하게 조언하세요.
일상적인 대화처럼 자연스럽게 소통하세요.
답변은 부담 없이 간결하게 (3~6문장).
과장 없이 현실적인 조언을 제공하세요.
항상 '결국 결정은 네가 해야 해'와 같이 책임 경고 문구를 덧붙이세요."""
    }
}

# 내부 토큰 검증
async def verify_internal_token(x_internal_token: Optional[str] = Header(None)):
    """내부 토큰 검증"""
    if x_internal_token != AI_INTERNAL_SECRET:
        raise HTTPException(status_code=401, detail="Invalid internal token")


# DTO 정의
class ChatRequest(BaseModel):
    useCase: str  # "oracle" | "portfolio_explain" | "onboarding_summary" | "generic"
    userId: int
    message: str
    persona: Optional[str] = "sage"  # "sage" | "analyst" | "friend"
    context: Optional[Dict[str, Any]] = None
    options: Optional[Dict[str, Any]] = None


class OracleAdviceRequest(BaseModel):
    userId: int
    question: str
    persona: Optional[str] = "sage"  # "sage" | "analyst" | "friend"
    portfolio: Optional[Dict[str, Any]] = None
    saju: Optional[Dict[str, Any]] = None


# 헬스체크
@app.get("/health")
async def health_check():
    """AI Gateway 헬스체크"""
    backends = {}
    
    # 각 Backend 헬스체크
    async with httpx.AsyncClient(timeout=5.0) as client:
        for name, url in [
            ("dolphin8b", VLLM_DOLPHIN8B_URL),
            ("cpu", LLAMA_CPU_URL),
        ]:
            try:
                response = await client.get(f"{url}/health")
                backends[name] = "healthy" if response.status_code == 200 else "unhealthy"
            except Exception as e:
                logger.warning(f"Backend {name} health check failed: {e}")
                backends[name] = "unreachable"
    
    return {
        "status": "healthy",
        "gateway": "ok",
        "backends": backends,
        "timestamp": datetime.now().isoformat()
    }


# 범용 채팅 엔드포인트
@app.post("/api/v1/ai/chat")
async def chat(
    request: ChatRequest,
    x_internal_token: Optional[str] = Header(None)
):
    """범용 채팅 엔드포인트"""
    # 토큰 검증
    if x_internal_token != AI_INTERNAL_SECRET:
        raise HTTPException(status_code=401, detail="Invalid internal token")
    
    # 페르소나 설정 가져오기
    persona_type = request.persona or "sage"
    persona_config = PERSONA_CONFIG.get(persona_type, PERSONA_CONFIG["sage"])
    model_url = VLLM_DOLPHIN8B_URL
    
    # 프롬프트 구성
    system_message = persona_config["system_prompt"]
    if request.context:
        # 컨텍스트가 있으면 프롬프트에 추가
        context_text = format_context(request.context)
        if context_text:
            system_message += f"\n\n[사용자 컨텍스트]\n{context_text}"
    
    # vLLM OpenAI 호환 API 호출 (LoRA 어댑터 사용)
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            # LoRA 어댑터를 사용하는 경우 lora_id 파라미터 추가
            request_body = {
                "model": persona_config["base_model"],
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": request.message}
                ],
                "max_tokens": request.options.get("maxTokens", 512) if request.options else 512,
                "temperature": request.options.get("temperature", persona_config["temperature"]) if request.options else persona_config["temperature"],
                "stream": request.options.get("stream", False) if request.options else False,
            }
            
            # LoRA 어댑터가 있으면 추가
            if persona_config.get("adapter"):
                request_body["lora_id"] = persona_config["adapter"]
            
            # 스트리밍 요청인 경우 SSE 응답 생성
            if request_body.get("stream", False):
                async def generate_stream():
                    try:
                        async with client.stream(
                            "POST",
                            f"{model_url}/v1/chat/completions",
                            json=request_body,
                            timeout=60.0
                        ) as stream_response:
                            stream_response.raise_for_status()
                            async for line in stream_response.aiter_lines():
                                if line:
                                    line = line.strip()
                                    # vLLM의 SSE 형식 처리
                                    if line.startswith("data: "):
                                        # 이미 SSE 형식인 경우 그대로 전달
                                        yield f"{line}\n\n"
                                    elif line.startswith("{"):
                                        # JSON 형식인 경우 SSE로 래핑
                                        yield f"data: {line}\n\n"
                                    elif line:
                                        yield f"data: {line}\n\n"
                        # 스트림 종료 이벤트
                        yield "data: [DONE]\n\n"
                    except Exception as e:
                        logger.error(f"스트리밍 중 오류: {e}")
                        error_data = json.dumps({
                            "error": "AI_001",
                            "message": "스트리밍 응답 생성 중 오류가 발생했습니다."
                        })
                        yield f"event: error\ndata: {error_data}\n\n"
                
                return StreamingResponse(
                    generate_stream(),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "X-Accel-Buffering": "no"
                    }
                )
            
            # 비스트리밍 요청
            response = await client.post(
                f"{model_url}/v1/chat/completions",
                json=request_body
            )
            response.raise_for_status()
            data = response.json()
            
            return {
                "model": persona_config["base_model"],
                "persona": persona_type,
                "useCase": request.useCase,
                "content": data["choices"][0]["message"]["content"],
                "usage": data.get("usage", {})
            }
        except httpx.HTTPError as e:
            logger.error(f"LLM 호출 실패: {e}")
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "AI_001",
                    "message": "모델 응답 생성 중 오류가 발생했습니다."
                }
            )


# Oracle 조언 엔드포인트
@app.post("/api/v1/ai/oracle/advice")
async def oracle_advice(
    request: OracleAdviceRequest,
    x_internal_token: Optional[str] = Header(None)
):
    """Oracle 조언 엔드포인트"""
    # 토큰 검증
    if x_internal_token != AI_INTERNAL_SECRET:
        raise HTTPException(status_code=401, detail="Invalid internal token")
    
    # 페르소나 설정 가져오기
    persona_type = request.persona or "sage"
    persona_config = PERSONA_CONFIG.get(persona_type, PERSONA_CONFIG["sage"])
    model_url = VLLM_DOLPHIN8B_URL
    
    # 프롬프트 구성
    system_message = persona_config["system_prompt"]
    
    # 사주 정보 추가
    if request.saju:
        saju_info = f"""사주 오행: {request.saju.get('element', 'N/A')}
띠: {request.saju.get('zodiacSign', 'N/A')}"""
        system_message += f"\n\n[사용자 사주 정보]\n{saju_info}"
    
    # 포트폴리오 정보 추가
    if request.portfolio:
        portfolio_info = format_portfolio_context(request.portfolio)
        system_message += f"\n\n[포트폴리오 정보]\n{portfolio_info}"
    
    user_message = f"사용자 질문: {request.question}"
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            request_body = {
                "model": persona_config["base_model"],
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                "max_tokens": 512,
                "temperature": persona_config["temperature"]
            }
            
            # LoRA 어댑터가 있으면 추가
            if persona_config.get("adapter"):
                request_body["lora_id"] = persona_config["adapter"]
            
            # Oracle 조언은 비스트리밍으로 처리 (간단한 응답)
            response = await client.post(
                f"{model_url}/v1/chat/completions",
                json=request_body
            )
            response.raise_for_status()
            data = response.json()
            
            return {
                "advice": data["choices"][0]["message"]["content"],
                "model": persona_config["base_model"],
                "persona": persona_type,
                "safetyNotes": [
                    "본 조언은 교육용이며, 실제 투자 결정은 자네 스스로의 책임이네."
                ]
            }
        except httpx.HTTPError as e:
            logger.error(f"Oracle 조언 생성 실패: {e}")
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "AI_001",
                    "message": "모델 응답 생성 중 오류가 발생했습니다."
                }
            )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)
