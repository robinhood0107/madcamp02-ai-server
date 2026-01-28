#!/usr/bin/env python3
"""
어댑터 내보내기 스크립트

학습된 LoRA 어댑터를 프로덕션 배포 형식으로 내보냅니다.
"""

import argparse
import shutil
from pathlib import Path
import json


def export_adapter(
    adapter_path: Path,
    output_path: Path,
    persona_type: str,
    base_model: str
):
    """
    어댑터를 배포 형식으로 내보내기
    
    Args:
        adapter_path: 학습된 어댑터 경로
        output_path: 출력 경로
        persona_type: 페르소나 타입
        base_model: 기본 모델 정보
    """
    print(f"[{persona_type}] 어댑터 내보내기...")
    print(f"  - 소스: {adapter_path}")
    print(f"  - 대상: {output_path}")
    
    # 출력 디렉토리 생성
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 어댑터 파일 복사
    if adapter_path.is_dir():
        # 디렉토리 전체 복사
        for file in adapter_path.iterdir():
            if file.is_file():
                shutil.copy2(file, output_path / file.name)
        print(f"  - 파일 복사 완료: {len(list(adapter_path.iterdir()))}개")
    else:
        print(f"  ⚠️ 경고: 어댑터 경로가 디렉토리가 아닙니다: {adapter_path}")
        return
    
    # 메타데이터 생성
    metadata = {
        "persona_type": persona_type,
        "base_model": base_model,
        "adapter_format": "peft",
        "version": "1.0.0"
    }
    
    metadata_file = output_path / "metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(f"  ✅ 완료: {output_path}")
    print(f"    - 메타데이터: {metadata_file}")


def main():
    parser = argparse.ArgumentParser(description="LoRA 어댑터 내보내기")
    parser.add_argument("--adapter", type=str, required=True,
                       help="학습된 어댑터 경로")
    parser.add_argument("--output", type=str, required=True,
                       help="출력 경로")
    parser.add_argument("--persona", type=str, required=True,
                       choices=["sage", "analyst", "friend"],
                       help="페르소나 타입")
    parser.add_argument("--base-model", type=str, required=True,
                       help="기본 모델 정보")
    
    args = parser.parse_args()
    
    export_adapter(
        adapter_path=Path(args.adapter),
        output_path=Path(args.output),
        persona_type=args.persona,
        base_model=args.base_model
    )


if __name__ == "__main__":
    main()
