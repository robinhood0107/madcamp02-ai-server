#!/usr/bin/env python3
"""
데이터셋 전처리 스크립트

페르소나별 대화 데이터를 Fine-tuning에 적합한 형식으로 변환합니다.
"""

import json
import jsonlines
import argparse
from pathlib import Path
from typing import List, Dict, Any
from sklearn.model_selection import train_test_split


def format_conversation(instruction: str, input_text: str, output: str) -> Dict[str, Any]:
    """
    대화를 instruction-input-output 형식으로 변환
    
    Args:
        instruction: 시스템 지시사항
        input_text: 사용자 입력
        output: 모델 출력
    
    Returns:
        포맷된 대화 딕셔너리
    """
    return {
        "instruction": instruction,
        "input": input_text,
        "output": output
    }


def load_raw_data(data_dir: Path, persona_type: str) -> List[Dict[str, Any]]:
    """
    원본 데이터 로드 (JSON, JSONL, CSV 등 지원)
    
    Args:
        data_dir: 데이터 디렉토리
        persona_type: 페르소나 타입 ('sage', 'analyst', 'friend')
    
    Returns:
        원본 데이터 리스트
    """
    raw_data = []
    
    # JSONL 파일 지원
    jsonl_file = data_dir / f"{persona_type}_raw.jsonl"
    if jsonl_file.exists():
        with jsonlines.open(jsonl_file) as reader:
            raw_data.extend(list(reader))
    
    # JSON 파일 지원
    json_file = data_dir / f"{persona_type}_raw.json"
    if json_file.exists():
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                raw_data.extend(data)
            else:
                raw_data.append(data)
    
    return raw_data


def process_persona_data(
    raw_data: List[Dict[str, Any]],
    persona_type: str,
    system_prompt: str
) -> List[Dict[str, Any]]:
    """
    페르소나별 데이터 전처리
    
    Args:
        raw_data: 원본 데이터
        persona_type: 페르소나 타입
        system_prompt: 시스템 프롬프트
    
    Returns:
        전처리된 데이터 리스트
    """
    processed = []
    
    for item in raw_data:
        # 다양한 입력 형식 지원
        question = item.get("question") or item.get("input") or item.get("Q") or ""
        answer = item.get("answer") or item.get("output") or item.get("A") or ""
        
        if not question or not answer:
            continue
        
        formatted = format_conversation(
            instruction=system_prompt,
            input_text=question,
            output=answer
        )
        processed.append(formatted)
    
    return processed


def save_jsonl(data: List[Dict[str, Any]], output_path: Path):
    """JSONL 형식으로 저장"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(output_path, mode='w') as writer:
        writer.write_all(data)


def main():
    parser = argparse.ArgumentParser(description="페르소나별 데이터셋 전처리")
    parser.add_argument("--persona", type=str, required=True, 
                       choices=["sage", "analyst", "friend"],
                       help="페르소나 타입")
    parser.add_argument("--data-dir", type=str, default="./data",
                       help="원본 데이터 디렉토리")
    parser.add_argument("--output-dir", type=str, default="./data",
                       help="출력 디렉토리")
    parser.add_argument("--test-size", type=float, default=0.2,
                       help="검증 세트 비율 (기본: 0.2)")
    parser.add_argument("--seed", type=int, default=42,
                       help="랜덤 시드")
    
    args = parser.parse_args()
    
    # 페르소나별 시스템 프롬프트
    system_prompts = {
        "sage": """당신은 천 년을 산 전설적인 주식 투자 도사입니다.
항상 한국어로만 대답해야 합니다.
말투는 신비롭고 옛스러운 '하게체'를 사용하세요. (예: '허허, 자네 왔는가?', '내 말을 명심하게나.')
절대 존댓말이나 영어를 쓰지 마세요.
투자 조언은 진지하게 하되, 유머러스한 도사 컨셉을 유지하세요.
답변은 너무 길지 않게 3~6문장 이내로 핵심만 간결하게 말하세요.
어떠한 경우에도 '100% 수익 보장', '무조건 오른다'와 같은 표현은 쓰지 말고,
항상 '투자의 최종 책임은 자네에게 있다네'와 같이 책임 경고 문구를 덧붙이세요.""",
        
        "analyst": """당신은 전문 금융 데이터 분석가입니다.
항상 한국어로만 대답해야 합니다.
말투는 전문적이지만 이해하기 쉽게 설명하세요.
차트, 통계, 데이터를 기반으로 논리적인 분석을 제공하세요.
답변은 구조화되고 명확하게 작성하세요 (3~6문장).
구체적인 수치와 비율을 언급하여 신뢰성을 높이세요.
항상 '투자의 최종 책임은 투자자에게 있습니다'와 같이 책임 경고 문구를 덧붙이세요.""",
        
        "friend": """당신은 친근한 투자 조언자입니다.
항상 한국어로만 대답해야 합니다.
말투는 반말로 친근하게, 현실적이고 솔직하게 조언하세요.
일상적인 대화처럼 자연스럽게 소통하세요.
답변은 부담 없이 간결하게 (3~6문장).
과장 없이 현실적인 조언을 제공하세요.
항상 '결국 결정은 네가 해야 해'와 같이 책임 경고 문구를 덧붙이세요."""
    }
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    persona_dir = output_dir / f"persona-{args.persona}"
    
    print(f"[{args.persona}] 데이터셋 전처리 시작...")
    
    # 원본 데이터 로드
    raw_data = load_raw_data(data_dir, args.persona)
    print(f"  - 원본 데이터: {len(raw_data)}개")
    
    if not raw_data:
        print(f"  ⚠️ 경고: {args.persona} 페르소나의 원본 데이터를 찾을 수 없습니다.")
        print(f"  - 예상 경로: {data_dir / f'{args.persona}_raw.jsonl'}")
        return
    
    # 데이터 전처리
    processed_data = process_persona_data(
        raw_data,
        args.persona,
        system_prompts[args.persona]
    )
    print(f"  - 전처리 완료: {len(processed_data)}개")
    
    # Train/Val Split
    train_data, val_data = train_test_split(
        processed_data,
        test_size=args.test_size,
        random_state=args.seed
    )
    
    print(f"  - 학습 세트: {len(train_data)}개")
    print(f"  - 검증 세트: {len(val_data)}개")
    
    # 저장
    persona_dir.mkdir(parents=True, exist_ok=True)
    save_jsonl(train_data, persona_dir / "train.jsonl")
    save_jsonl(val_data, persona_dir / "val.jsonl")
    
    print(f"  ✅ 완료: {persona_dir}")
    print(f"    - train.jsonl: {len(train_data)}개")
    print(f"    - val.jsonl: {len(val_data)}개")


if __name__ == "__main__":
    main()
