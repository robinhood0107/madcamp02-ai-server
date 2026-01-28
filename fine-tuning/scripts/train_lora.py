#!/usr/bin/env python3
"""
LoRA Fine-tuning 스크립트

페르소나별 LoRA 어댑터를 학습합니다.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
import os


def load_jsonl_dataset(file_path: Path):
    """JSONL 파일을 데이터셋으로 로드"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def format_prompt(example: Dict[str, Any]) -> str:
    """
    instruction-input-output을 프롬프트 형식으로 변환
    
    Args:
        example: 데이터 예시
    
    Returns:
        포맷된 프롬프트 문자열
    """
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output = example.get("output", "")
    
    if input_text:
        prompt = f"{instruction}\n\n사용자: {input_text}\n\n조언자: {output}"
    else:
        prompt = f"{instruction}\n\n{output}"
    
    return prompt


def tokenize_function(examples, tokenizer, max_length: int = 512):
    """토큰화 함수"""
    prompts = [format_prompt(ex) for ex in examples]
    
    tokenized = tokenizer(
        prompts,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt"
    )
    
    # labels는 input_ids와 동일 (언어 모델링)
    tokenized["labels"] = tokenized["input_ids"].clone()
    
    return tokenized


def train_persona_lora(
    base_model: str,
    persona_name: str,
    train_data_path: Path,
    val_data_path: Path,
    output_dir: Path,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    learning_rate: float = 2e-4,
    batch_size: int = 4,
    num_epochs: int = 3,
    max_length: int = 512,
    use_gpu: bool = True
):
    """
    페르소나별 LoRA 학습
    
    Args:
        base_model: 기본 모델 경로 (HuggingFace 모델 ID 또는 로컬 경로)
        persona_name: 페르소나 이름
        train_data_path: 학습 데이터 경로
        val_data_path: 검증 데이터 경로
        output_dir: 출력 디렉토리
        lora_rank: LoRA rank
        lora_alpha: LoRA alpha
        learning_rate: 학습률
        batch_size: 배치 크기
        num_epochs: 에폭 수
        max_length: 최대 시퀀스 길이
        use_gpu: GPU 사용 여부
    """
    print(f"[{persona_name}] LoRA Fine-tuning 시작...")
    print(f"  - 기본 모델: {base_model}")
    print(f"  - 출력 디렉토리: {output_dir}")
    
    # 디바이스 설정
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    print(f"  - 디바이스: {device}")
    
    # 토크나이저 로드
    print("  - 토크나이저 로드 중...")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 모델 로드
    print("  - 모델 로드 중...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None
    )
    
    # LoRA 설정
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Llama 계열
        bias="none"
    )
    
    # PEFT 모델로 변환
    print("  - LoRA 어댑터 적용 중...")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 데이터셋 로드
    print("  - 데이터셋 로드 중...")
    train_data = load_jsonl_dataset(train_data_path)
    val_data = load_jsonl_dataset(val_data_path)
    
    print(f"    - 학습 데이터: {len(train_data)}개")
    print(f"    - 검증 데이터: {len(val_data)}개")
    
    # 데이터셋 토큰화
    def tokenize_dataset(examples):
        return tokenize_function(examples, tokenizer, max_length)
    
    # 간단한 데이터셋 클래스 (실제로는 datasets 라이브러리 사용 권장)
    class SimpleDataset:
        def __init__(self, data, tokenizer, max_length):
            self.data = data
            self.tokenizer = tokenizer
            self.max_length = max_length
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            prompt = format_prompt(self.data[idx])
            tokenized = self.tokenizer(
                prompt,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt"
            )
            return {
                "input_ids": tokenized["input_ids"].squeeze(),
                "attention_mask": tokenized["attention_mask"].squeeze(),
                "labels": tokenized["input_ids"].squeeze()
            }
    
    train_dataset = SimpleDataset(train_data, tokenizer, max_length)
    val_dataset = SimpleDataset(val_data, tokenizer, max_length)
    
    # Data Collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Training Arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        logging_dir=str(output_dir / "logs"),
        logging_steps=10,
        save_steps=100,
        eval_steps=100,
        evaluation_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        fp16=use_gpu and torch.cuda.is_available(),
        report_to="tensorboard",
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    # 학습 시작
    print("  - 학습 시작...")
    trainer.train()
    
    # 모델 저장
    print("  - 모델 저장 중...")
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    
    print(f"  ✅ 완료: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="페르소나별 LoRA Fine-tuning")
    parser.add_argument("--base-model", type=str, required=True,
                       help="기본 모델 경로 (HuggingFace ID 또는 로컬 경로)")
    parser.add_argument("--persona", type=str, required=True,
                       choices=["sage", "analyst", "friend"],
                       help="페르소나 타입")
    parser.add_argument("--train-data", type=str, required=True,
                       help="학습 데이터 경로 (JSONL)")
    parser.add_argument("--val-data", type=str, required=True,
                       help="검증 데이터 경로 (JSONL)")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="출력 디렉토리")
    parser.add_argument("--lora-rank", type=int, default=16,
                       help="LoRA rank (기본: 16)")
    parser.add_argument("--lora-alpha", type=int, default=32,
                       help="LoRA alpha (기본: 32)")
    parser.add_argument("--learning-rate", type=float, default=2e-4,
                       help="학습률 (기본: 2e-4)")
    parser.add_argument("--batch-size", type=int, default=4,
                       help="배치 크기 (기본: 4)")
    parser.add_argument("--num-epochs", type=int, default=3,
                       help="에폭 수 (기본: 3)")
    parser.add_argument("--max-length", type=int, default=512,
                       help="최대 시퀀스 길이 (기본: 512)")
    parser.add_argument("--no-gpu", action="store_true",
                       help="GPU 사용 안 함")
    
    args = parser.parse_args()
    
    train_persona_lora(
        base_model=args.base_model,
        persona_name=args.persona,
        train_data_path=Path(args.train_data),
        val_data_path=Path(args.val_data),
        output_dir=Path(args.output_dir),
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        max_length=args.max_length,
        use_gpu=not args.no_gpu
    )


if __name__ == "__main__":
    main()
