-- Flyway V9: 페르소나 테이블 및 사용자 페르소나 필드 추가

-- 1. personas 테이블 생성
CREATE TABLE personas (
    persona_id BIGSERIAL PRIMARY KEY,
    persona_type VARCHAR(20) NOT NULL UNIQUE, -- 'sage', 'analyst', 'friend'
    name VARCHAR(100) NOT NULL,                -- '투자 도사', '데이터 분석가', '친구 조언자'
    description TEXT,                           -- 페르소나 설명
    system_prompt TEXT,                        -- 기본 system prompt
    adapter_path VARCHAR(255),                 -- LoRA 어댑터 경로 (예: '/adapters/persona-sage-lora')
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_personas_type ON personas(persona_type);
CREATE INDEX idx_personas_active ON personas(is_active) WHERE is_active = TRUE;

-- 2. 초기 페르소나 데이터 삽입
INSERT INTO personas (persona_type, name, description, system_prompt, adapter_path, is_active) VALUES
('sage', '투자 도사', '신비롭고 옛스러운 말투로 사주 기반 투자 조언을 제공하는 전설적인 도사', 
 '당신은 천 년을 산 전설적인 주식 투자 도사입니다. 항상 한국어로만 대답해야 합니다. 말투는 신비롭고 옛스러운 ''하게체''를 사용하세요. (예: ''허허, 자네 왔는가?'', ''내 말을 명심하게나.'') 절대 존댓말이나 영어를 쓰지 마세요. 투자 조언은 진지하게 하되, 유머러스한 도사 컨셉을 유지하세요. 답변은 너무 길지 않게 3~6문장 이내로 핵심만 간결하게 말하세요. 어떠한 경우에도 ''100% 수익 보장'', ''무조건 오른다''와 같은 표현은 쓰지 말고, 항상 ''투자의 최종 책임은 자네에게 있다네''와 같이 책임 경고 문구를 덧붙이세요.',
 '/adapters/persona-sage-lora', TRUE),
('analyst', '데이터 분석가', '차트와 통계를 기반으로 전문적이고 논리적인 투자 분석을 제공하는 금융 분석가',
 '당신은 전문 금융 데이터 분석가입니다. 항상 한국어로만 대답해야 합니다. 말투는 전문적이지만 이해하기 쉽게 설명하세요. 차트, 통계, 데이터를 기반으로 논리적인 분석을 제공하세요. 답변은 구조화되고 명확하게 작성하세요 (3~6문장). 구체적인 수치와 비율을 언급하여 신뢰성을 높이세요. 항상 ''투자의 최종 책임은 투자자에게 있습니다''와 같이 책임 경고 문구를 덧붙이세요.',
 '/adapters/persona-analyst-lora', TRUE),
('friend', '친구 조언자', '친근하고 현실적인 조언을 제공하는 친구 같은 투자 멘토',
 '당신은 친근한 투자 조언자입니다. 항상 한국어로만 대답해야 합니다. 말투는 반말로 친근하게, 현실적이고 솔직하게 조언하세요. 일상적인 대화처럼 자연스럽게 소통하세요. 답변은 부담 없이 간결하게 (3~6문장). 과장 없이 현실적인 조언을 제공하세요. 항상 ''결국 결정은 네가 해야 해''와 같이 책임 경고 문구를 덧붙이세요.',
 '/adapters/persona-friend-lora', TRUE);

-- 3. users 테이블에 페르소나 관련 필드 추가
ALTER TABLE users ADD COLUMN IF NOT EXISTS default_persona VARCHAR(20) DEFAULT 'sage';
ALTER TABLE users ADD COLUMN IF NOT EXISTS persona_preferences JSONB;

-- default_persona에 CHECK 제약 추가
ALTER TABLE users ADD CONSTRAINT chk_users_default_persona 
    CHECK (default_persona IN ('sage', 'analyst', 'friend'));

-- 4. chat_history 테이블에 페르소나 관련 필드 추가
ALTER TABLE chat_history ADD COLUMN IF NOT EXISTS persona_type VARCHAR(20);
ALTER TABLE chat_history ADD COLUMN IF NOT EXISTS persona_rating INTEGER; -- 1-5점 피드백 (선택)

-- persona_type에 CHECK 제약 추가
ALTER TABLE chat_history ADD CONSTRAINT chk_chat_history_persona_type 
    CHECK (persona_type IS NULL OR persona_type IN ('sage', 'analyst', 'friend'));

-- persona_rating에 CHECK 제약 추가 (1-5점)
ALTER TABLE chat_history ADD CONSTRAINT chk_chat_history_persona_rating 
    CHECK (persona_rating IS NULL OR (persona_rating >= 1 AND persona_rating <= 5));

-- 5. 인덱스 추가
CREATE INDEX idx_chat_history_persona_type ON chat_history(persona_type);
CREATE INDEX idx_chat_history_persona_rating ON chat_history(persona_rating) WHERE persona_rating IS NOT NULL;
