#!/usr/bin/env python3
"""
ChatHistory에서 실제 대화 데이터를 수집하여 Fine-tuning용 JSONL 파일로 변환하는 스크립트

사용법:
    python collect_chat_history.py --persona sage --output ./data/sage_raw.jsonl
    python collect_chat_history.py --persona analyst --output ./data/analyst_raw.jsonl
    python collect_chat_history.py --persona friend --output ./data/friend_raw.jsonl
    python collect_chat_history.py --all --output-dir ./data
"""

import argparse
import json
import jsonlines
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta


class ChatHistoryCollector:
    """Spring Backend API에서 ChatHistory를 수집하는 클래스"""
    
    def __init__(self, base_url: str, token: Optional[str] = None):
        """
        Args:
            base_url: Spring Backend Base URL (예: http://localhost:8080)
            token: 인증 토큰 (관리자 권한 필요)
        """
        self.base_url = base_url.rstrip('/')
        self.token = token
        self.session = requests.Session()
        if token:
            self.session.headers.update({
                'Authorization': f'Bearer {token}',
                'Content-Type': 'application/json'
            })
    
    def fetch_chat_history(
        self,
        persona_type: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 1000,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        ChatHistory API에서 대화 데이터를 가져옵니다.
        
        Args:
            persona_type: 페르소나 타입 ('sage', 'analyst', 'friend')
            start_date: 시작 날짜
            end_date: 종료 날짜
            limit: 최대 개수
            offset: 오프셋
        
        Returns:
            ChatHistory 리스트
        """
        # API 엔드포인트 (백엔드 구현 필요)
        # 예상 엔드포인트: GET /api/v1/admin/chat-history
        url = f"{self.base_url}/api/v1/admin/chat-history"
        
        params = {
            'limit': limit,
            'offset': offset
        }
        
        if persona_type:
            params['personaType'] = persona_type
        
        if start_date:
            params['startDate'] = start_date.isoformat()
        
        if end_date:
            params['endDate'] = end_date.isoformat()
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # 응답 형식에 따라 조정 필요
            # 예상 형식: { "items": [...], "total": 100 }
            if isinstance(data, dict) and 'items' in data:
                return data['items']
            elif isinstance(data, list):
                return data
            else:
                print(f"⚠️  예상치 못한 응답 형식: {type(data)}")
                return []
        except requests.exceptions.RequestException as e:
            print(f"❌ API 호출 실패: {e}")
            return []
    
    def convert_to_training_format(
        self,
        chat_history: List[Dict[str, Any]],
        persona_type: str,
        include_context: bool = False
    ) -> List[Dict[str, Any]]:
        """
        ChatHistory를 Fine-tuning용 형식으로 변환합니다.
        
        Args:
            chat_history: ChatHistory 리스트
            persona_type: 페르소나 타입
            include_context: 금융 데이터 컨텍스트 포함 여부
        
        Returns:
            {"question": "...", "answer": "...", "context": {...}} 형식의 리스트
        """
        training_data = []
        
        for history in chat_history:
            # ChatHistory 필드명에 따라 조정 필요
            question = history.get('question') or history.get('message') or ''
            answer = history.get('response') or history.get('answer') or ''
            
            # 빈 데이터는 제외
            if not question.strip() or not answer.strip():
                continue
            
            # PII 제거 (선택적)
            # 개인정보가 포함된 경우 필터링 로직 추가
            
            data_item = {
                "question": question.strip(),
                "answer": answer.strip()
            }
            
            # 컨텍스트가 있으면 포함 (금융 데이터 등)
            if include_context and history.get('context'):
                data_item["context"] = history.get('context')
            
            training_data.append(data_item)
        
        return training_data
    
    def save_to_jsonl(
        self,
        data: List[Dict[str, str]],
        output_path: Path
    ):
        """데이터를 JSONL 파일로 저장합니다."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with jsonlines.open(output_path, mode='w') as writer:
            for item in data:
                writer.write(item)
        
        print(f"✅ {len(data)}개 데이터를 {output_path}에 저장했습니다.")


def main():
    parser = argparse.ArgumentParser(
        description='ChatHistory에서 Fine-tuning용 데이터 수집'
    )
    parser.add_argument(
        '--base-url',
        type=str,
        default='http://localhost:8080',
        help='Spring Backend Base URL'
    )
    parser.add_argument(
        '--token',
        type=str,
        help='인증 토큰 (관리자 권한 필요)'
    )
    parser.add_argument(
        '--persona',
        type=str,
        choices=['sage', 'analyst', 'friend'],
        help='페르소나 타입 (하나만 선택)'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='모든 페르소나 데이터 수집'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='출력 파일 경로 (--persona 사용 시)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./data',
        help='출력 디렉토리 (--all 사용 시)'
    )
    parser.add_argument(
        '--start-date',
        type=str,
        help='시작 날짜 (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        help='종료 날짜 (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=1000,
        help='최대 수집 개수'
    )
    
    args = parser.parse_args()
    
    # 날짜 파싱
    start_date = None
    end_date = None
    if args.start_date:
        start_date = datetime.fromisoformat(args.start_date)
    if args.end_date:
        end_date = datetime.fromisoformat(args.end_date)
    
    collector = ChatHistoryCollector(args.base_url, args.token)
    
    if args.all:
        # 모든 페르소나 데이터 수집
        output_dir = Path(args.output_dir)
        personas = ['sage', 'analyst', 'friend']
        
        for persona in personas:
            print(f"\n📊 {persona} 페르소나 데이터 수집 중...")
            chat_history = collector.fetch_chat_history(
                persona_type=persona,
                start_date=start_date,
                end_date=end_date,
                limit=args.limit
            )
            
            if not chat_history:
                print(f"⚠️  {persona} 페르소나 데이터가 없습니다.")
                continue
            
            training_data = collector.convert_to_training_format(
                chat_history,
                persona
            )
            
            output_path = output_dir / f"{persona}_raw.jsonl"
            collector.save_to_jsonl(training_data, output_path)
    
    elif args.persona:
        # 특정 페르소나만 수집
        if not args.output:
            args.output = f"./data/{args.persona}_raw.jsonl"
        
        print(f"📊 {args.persona} 페르소나 데이터 수집 중...")
        chat_history = collector.fetch_chat_history(
            persona_type=args.persona,
            start_date=start_date,
            end_date=end_date,
            limit=args.limit
        )
        
        if not chat_history:
            print(f"⚠️  {args.persona} 페르소나 데이터가 없습니다.")
            return
        
        training_data = collector.convert_to_training_format(
            chat_history,
            args.persona
        )
        
        output_path = Path(args.output)
        collector.save_to_jsonl(training_data, output_path)
    
    else:
        parser.print_help()
        print("\n❌ --persona 또는 --all 옵션을 지정해주세요.")


if __name__ == '__main__':
    main()
