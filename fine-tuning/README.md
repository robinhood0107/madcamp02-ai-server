# 페르소나 Fine-tuning 가이드

이 디렉토리는 페르소나별 LoRA Fine-tuning을 위한 스크립트와 데이터셋 구조를 포함합니다.

## 📁 디렉토리 구조

```
fine-tuning/
├── scripts/
│   ├── prepare_dataset.py      # 데이터셋 전처리
│   ├── train_lora.py           # LoRA 학습
│   ├── evaluate_persona.py     # 페르소나 품질 평가
│   └── export_adapter.py       # 어댑터 내보내기
├── data/
│   ├── persona-sage/
│   │   ├── train.jsonl
│   │   └── val.jsonl
│   ├── persona-analyst/
│   │   ├── train.jsonl
│   │   └── val.jsonl
│   └── persona-friend/
│       ├── train.jsonl
│       └── val.jsonl
├── adapters/                   # 학습된 어댑터 저장
│   ├── persona-sage-lora/
│   ├── persona-analyst-lora/
│   └── persona-friend-lora/
└── requirements.txt
```

## 🚀 사용 방법

### 1. 환경 설정

```bash
cd ai-server/fine-tuning
pip install -r requirements.txt
```

### 2. 데이터셋 준비

원본 데이터를 `data/` 디렉토리에 배치합니다.

**데이터 형식 예시** (`data/sage_raw.jsonl`):

```json
{"question": "불(火) 오행인데 어떤 종목이 좋을까?", "answer": "허허, 자네의 사주는 불 기운이 왕성하니..."}
{"question": "오늘 투자 운세는?", "answer": "오늘은 금(金) 기운이 강하여..."}
```

### 3. 데이터셋 전처리

```bash
# 투자 도사 (Sage) 데이터 전처리
python scripts/prepare_dataset.py \
    --persona sage \
    --data-dir ./data \
    --output-dir ./data \
    --test-size 0.2

# 데이터 분석가 (Analyst) 데이터 전처리
python scripts/prepare_dataset.py \
    --persona analyst \
    --data-dir ./data \
    --output-dir ./data

# 친구 조언자 (Friend) 데이터 전처리
python scripts/prepare_dataset.py \
    --persona friend \
    --data-dir ./data \
    --output-dir ./data
```

### 4. LoRA 학습

```bash
# 투자 도사 (Sage) 학습
python scripts/train_lora.py \
    --base-model cognitivecomputations/dolphin-2.9.4-llama3.1-8b \
    --persona sage \
    --train-data ./data/persona-sage/train.jsonl \
    --val-data ./data/persona-sage/val.jsonl \
    --output-dir ./adapters/persona-sage-lora \
    --lora-rank 16 \
    --lora-alpha 32 \
    --learning-rate 2e-4 \
    --batch-size 4 \
    --num-epochs 3

# 데이터 분석가 (Analyst) 학습
python scripts/train_lora.py \
    --base-model cognitivecomputations/dolphin-2.9.4-llama3.1-8b \
    --persona analyst \
    --train-data ./data/persona-analyst/train.jsonl \
    --val-data ./data/persona-analyst/val.jsonl \
    --output-dir ./adapters/persona-analyst-lora

# 친구 조언자 (Friend) 학습
python scripts/train_lora.py \
    --base-model cognitivecomputations/dolphin-2.9.4-llama3.1-8b \
    --persona friend \
    --train-data ./data/persona-friend/train.jsonl \
    --val-data ./data/persona-friend/val.jsonl \
    --output-dir ./adapters/persona-friend-lora
```

### 5. 품질 평가

```bash
# 투자 도사 (Sage) 평가
python scripts/evaluate_persona.py \
    --base-model cognitivecomputations/dolphin-2.9.4-llama3.1-8b \
    --adapter ./adapters/persona-sage-lora \
    --test-data ./data/persona-sage/val.jsonl

# 다른 페르소나도 동일하게 평가
```

### 6. 어댑터 배포

```bash
# 어댑터를 프로덕션 디렉토리로 내보내기
python scripts/export_adapter.py \
    --adapter ./adapters/persona-sage-lora \
    --output ../adapters/persona-sage-lora \
    --persona sage \
    --base-model dolphin-2.9.4-llama3.1-8b

# Docker 볼륨에 복사
cp -r ../adapters/persona-*-lora /mnt/ai-server/adapters/

# vLLM 서비스 재시작
cd ..
docker compose restart vllm-dolphin8b
```

## 📊 데이터 수집 가이드

### 실제 데이터 수집 (권장)

Spring Backend의 `ChatHistory`에서 실제 사용자 대화 데이터를 수집합니다.

**필요한 API**: `GET /api/v1/admin/chat-history` (관리자 권한 필요)

**스크립트 사용법**:

```bash
# 특정 페르소나 데이터 수집
python scripts/collect_chat_history.py \
    --base-url http://localhost:8080 \
    --token YOUR_ADMIN_TOKEN \
    --persona sage \
    --output ./data/sage_raw.jsonl \
    --limit 1000

# 모든 페르소나 데이터 수집
python scripts/collect_chat_history.py \
    --base-url http://localhost:8080 \
    --token YOUR_ADMIN_TOKEN \
    --all \
    --output-dir ./data \
    --start-date 2026-01-01 \
    --end-date 2026-01-31
```

**API 구현 필요**: 백엔드에서 `ChatHistoryAdminController` 구현 필요

- 상세 설계: `docs/BACKEND_CHAT_HISTORY_API.md` 참조
- 엔드포인트: `GET /api/v1/admin/chat-history`
- 권한: 관리자 권한 (`ROLE_ADMIN`) 필요

**직접 API 호출**:

```bash
# cURL로 데이터 수집
curl -H "Authorization: Bearer YOUR_ADMIN_TOKEN" \
     "http://localhost:8080/api/v1/admin/chat-history?personaType=sage&limit=1000" \
     > sage_raw.jsonl

# JSONL 형식으로 직접 다운로드
curl -H "Authorization: Bearer YOUR_ADMIN_TOKEN" \
     "http://localhost:8080/api/v1/admin/chat-history/export?personaType=sage" \
     -o sage_raw.jsonl
```

### 금융 데이터 기반 대화 생성 (권장)

실제 금융 API 데이터를 사용하여 Fine-tuning용 대화를 자동 생성합니다.

**스크립트 사용법**:

```bash
# 실제 금융 데이터를 사용하여 대화 생성
python scripts/generate_financial_conversations.py \
    --backend-url http://localhost:8080 \
    --token YOUR_TOKEN \
    --persona analyst \
    --tickers AAPL TSLA MSFT GOOGL NVDA \
    --count 200 \
    --output ./data/analyst_financial_raw.jsonl \
    --include-context
```

**장점**:

- 실제 금융 데이터를 사용하여 정확한 정보 제공
- 다양한 종목과 시장 상황을 반영
- 대량의 데이터를 빠르게 생성 가능

**상세 가이드**: `docs/AI_FINANCIAL_DATA_INTEGRATION.md` 참조

### 초기 데이터셋 구축 (500-1000개/페르소나)

각 페르소나별로 다음과 같은 데이터를 수집합니다:

#### 투자 도사 (Sage)

- 사주 기반 투자 조언 대화
- 운세 기반 종목 추천
- 하게체 말투 유지

#### 데이터 분석가 (Analyst)

- 차트/통계 기반 분석
- 포트폴리오 리스크 평가
- 전문적이지만 이해하기 쉬운 설명

#### 친구 조언자 (Friend)

- 친근한 톤의 투자 대화
- 현실적인 조언
- 반말 사용

### 데이터 형식

**입력 형식** (JSONL):

```json
{ "question": "사용자 질문", "answer": "페르소나별 답변" }
```

또는 JSON:

```json
[
  { "question": "질문1", "answer": "답변1" },
  { "question": "질문2", "answer": "답변2" }
]
```

## ⚙️ 하이퍼파라미터 튜닝

기본 파라미터:

- LoRA rank: 16-32
- LoRA alpha: 32-64
- Learning rate: 2e-4
- Batch size: 4-8
- Epochs: 3-5

품질이 낮으면:

- LoRA rank 증가 (16 → 32)
- Learning rate 감소 (2e-4 → 1e-4)
- Epochs 증가 (3 → 5)

과적합 발생 시:

- LoRA dropout 증가 (0.1 → 0.2)
- Early stopping 사용
- 데이터 증강

## 📈 평가 메트릭

- **ROUGE-1, ROUGE-2, ROUGE-L**: 답변 품질 평가
- **사람 평가**: 페르소나 말투 일치도 (1-5점)
- **A/B 테스트**: 실제 사용자 피드백

## 🔄 지속적 개선

1. **사용자 대화 수집**: Spring Backend의 `ChatHistory`에서 페르소나별 대화 수집
2. **증분 학습**: 월별 또는 분기별로 새로운 데이터로 재학습
3. **품질 모니터링**: 페르소나별 사용 통계 및 피드백 분석

## 참고 자료

- [PEFT (Parameter-Efficient Fine-Tuning)](https://github.com/huggingface/peft)
- [LoRA 논문](https://arxiv.org/abs/2106.09685)
- [vLLM LoRA 지원](https://docs.vllm.ai/en/latest/serving/lora.html)

## 관련 문서

- **금융 데이터 통합**: `docs/AI_FINANCIAL_DATA_INTEGRATION.md` - 실제 금융 API 데이터를 활용한 대화 및 Fine-tuning
- **ChatHistory 데이터 수집**: `docs/BACKEND_CHAT_HISTORY_API.md` - Fine-tuning용 실제 대화 데이터 수집 API
- **페르소나 시스템 설계**: `docs/BACKEND_PERSONA_DESIGN.md` - 백엔드 페르소나 시스템 상세 설계
- **AI 서버 명세**: `docs/AI_SERVER_SPEC.md` - AI 서버 전체 아키텍처 및 API 명세
- **백엔드 개발 계획**: `docs/BACKEND_DEVELOPMENT_PLAN.md` - 백엔드 개발 계획 및 AI 연동 상세
- **프론트엔드 개발 계획**: `docs/FRONTEND_DEVELOPMENT_PLAN.md` - 프론트엔드 개발 계획 및 `/oracle` 페이지 연동
