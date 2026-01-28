#!/usr/bin/env python3
"""
페르소나 품질 평가 스크립트

학습된 LoRA 어댑터의 품질을 평가합니다.
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from rouge_score import rouge_scorer


def load_test_data(test_data_path: Path) -> List[Dict[str, Any]]:
    """테스트 데이터 로드"""
    data = []
    with open(test_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def generate_response(
    model,
    tokenizer,
    instruction: str,
    input_text: str,
    max_length: int = 512,
    temperature: float = 0.7
) -> str:
    """모델 응답 생성"""
    prompt = f"{instruction}\n\n사용자: {input_text}\n\n조언자: "
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 프롬프트 부분 제거
    response = generated_text.split("조언자: ")[-1].strip()
    
    return response


def evaluate_persona(
    base_model: str,
    adapter_path: Path,
    test_data_path: Path,
    max_length: int = 512,
    temperature: float = 0.7
):
    """
    페르소나 품질 평가
    
    Args:
        base_model: 기본 모델 경로
        adapter_path: LoRA 어댑터 경로
        test_data_path: 테스트 데이터 경로
        max_length: 최대 생성 길이
        temperature: 생성 온도
    """
    print(f"페르소나 품질 평가 시작...")
    print(f"  - 기본 모델: {base_model}")
    print(f"  - 어댑터: {adapter_path}")
    print(f"  - 테스트 데이터: {test_data_path}")
    
    # 모델 및 토크나이저 로드
    print("  - 모델 로드 중...")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    base_model_obj = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    
    # LoRA 어댑터 로드
    model = PeftModel.from_pretrained(base_model_obj, str(adapter_path))
    model.eval()
    
    # 테스트 데이터 로드
    test_data = load_test_data(test_data_path)
    print(f"  - 테스트 샘플: {len(test_data)}개")
    
    # ROUGE 스코어 계산
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    results = []
    rouge_scores = []
    
    print("  - 평가 진행 중...")
    for i, example in enumerate(test_data):
        instruction = example.get("instruction", "")
        input_text = example.get("input", "")
        expected_output = example.get("output", "")
        
        # 모델 응답 생성
        generated_output = generate_response(
            model,
            tokenizer,
            instruction,
            input_text,
            max_length,
            temperature
        )
        
        # ROUGE 점수 계산
        scores = scorer.score(expected_output, generated_output)
        rouge_scores.append({
            "rouge1": scores["rouge1"].fmeasure,
            "rouge2": scores["rouge2"].fmeasure,
            "rougeL": scores["rougeL"].fmeasure
        })
        
        results.append({
            "input": input_text,
            "expected": expected_output,
            "generated": generated_output,
            "rouge1": scores["rouge1"].fmeasure,
            "rouge2": scores["rouge2"].fmeasure,
            "rougeL": scores["rougeL"].fmeasure
        })
        
        if (i + 1) % 10 == 0:
            print(f"    - 진행: {i + 1}/{len(test_data)}")
    
    # 평균 점수 계산
    avg_rouge1 = sum(s["rouge1"] for s in rouge_scores) / len(rouge_scores)
    avg_rouge2 = sum(s["rouge2"] for s in rouge_scores) / len(rouge_scores)
    avg_rougeL = sum(s["rougeL"] for s in rouge_scores) / len(rouge_scores)
    
    # 결과 출력
    print("\n  📊 평가 결과:")
    print(f"    - ROUGE-1: {avg_rouge1:.4f}")
    print(f"    - ROUGE-2: {avg_rouge2:.4f}")
    print(f"    - ROUGE-L: {avg_rougeL:.4f}")
    
    # 결과 저장
    output_file = adapter_path.parent / "evaluation_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "summary": {
                "avg_rouge1": avg_rouge1,
                "avg_rouge2": avg_rouge2,
                "avg_rougeL": avg_rougeL,
                "num_samples": len(test_data)
            },
            "detailed_results": results
        }, f, ensure_ascii=False, indent=2)
    
    print(f"  ✅ 결과 저장: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="페르소나 품질 평가")
    parser.add_argument("--base-model", type=str, required=True,
                       help="기본 모델 경로")
    parser.add_argument("--adapter", type=str, required=True,
                       help="LoRA 어댑터 경로")
    parser.add_argument("--test-data", type=str, required=True,
                       help="테스트 데이터 경로 (JSONL)")
    parser.add_argument("--max-length", type=int, default=512,
                       help="최대 생성 길이")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="생성 온도")
    
    args = parser.parse_args()
    
    evaluate_persona(
        base_model=args.base_model,
        adapter_path=Path(args.adapter),
        test_data_path=Path(args.test_data),
        max_length=args.max_length,
        temperature=args.temperature
    )


if __name__ == "__main__":
    main()
