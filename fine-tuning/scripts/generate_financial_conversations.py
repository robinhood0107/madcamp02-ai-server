#!/usr/bin/env python3
"""
실제 금융 데이터를 사용하여 Fine-tuning용 대화를 생성하는 스크립트

사용법:
    python generate_financial_conversations.py \
        --backend-url http://localhost:8080 \
        --token YOUR_TOKEN \
        --persona analyst \
        --tickers AAPL TSLA MSFT \
        --count 100 \
        --output ./data/analyst_financial_raw.jsonl
"""

import argparse
import json
import jsonlines
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime


class FinancialDataGenerator:
    """금융 데이터 기반 대화 생성기"""
    
    def __init__(self, backend_url: str, token: Optional[str] = None):
        """
        Args:
            backend_url: Spring Backend Base URL (예: http://localhost:8080)
            token: 인증 토큰 (선택적)
        """
        self.backend_url = backend_url.rstrip('/')
        self.token = token
        self.session = requests.Session()
        if token:
            self.session.headers.update({
                'Authorization': f'Bearer {token}',
                'Content-Type': 'application/json'
            })
    
    def get_market_indices(self) -> List[Dict[str, Any]]:
        """시장 지수 데이터 조회"""
        try:
            response = self.session.get(f"{self.backend_url}/api/v1/market/indices")
            if response.status_code == 200:
                data = response.json()
                return data.get('items', []) if isinstance(data, dict) else data
        except Exception as e:
            print(f"⚠️  시장 지수 조회 실패: {e}")
        return []
    
    def get_stock_quote(self, ticker: str) -> Optional[Dict[str, Any]]:
        """종목 현재가 조회"""
        try:
            response = self.session.get(f"{self.backend_url}/api/v1/stock/quote/{ticker}")
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"⚠️  {ticker} 조회 실패: {e}")
        return None
    
    def get_market_news(self, limit: int = 5) -> List[Dict[str, Any]]:
        """시장 뉴스 조회"""
        try:
            response = self.session.get(f"{self.backend_url}/api/v1/market/news")
            if response.status_code == 200:
                data = response.json()
                items = data.get('items', []) if isinstance(data, dict) else data
                return items[:limit]
        except Exception as e:
            print(f"⚠️  시장 뉴스 조회 실패: {e}")
        return []
    
    def generate_sage_conversation(self, stock_data: Dict[str, Any]) -> Dict[str, str]:
        """투자 도사 스타일 대화 생성"""
        ticker = stock_data.get("ticker", "N/A")
        price = stock_data.get("currentPrice") or stock_data.get("price", 0)
        change = stock_data.get("changePercent") or stock_data.get("change", 0)
        
        # 다양한 질문 패턴
        questions = [
            f"{ticker} 주가가 어때?",
            f"{ticker}는 지금 투자해도 될까?",
            f"{ticker} 종목에 대해 알려줘",
            f"{ticker}의 운세는?",
        ]
        
        import random
        question = random.choice(questions)
        
        if change > 0:
            answer = f"허허, {ticker}는 현재 ${price:.2f}로 전일 대비 {change:.2f}% 상승했네. " \
                    f"상승 기운이 강하니 단기적으로는 긍정적이지만, 변동성을 고려하여 신중하게 접근하도록 하게. " \
                    f"투자의 최종 책임은 자네에게 있다네."
        elif change < 0:
            answer = f"{ticker}는 현재 ${price:.2f}로 전일 대비 {abs(change):.2f}% 하락했네. " \
                    f"하락 기운이 있으니 비중 조절을 명심하게나. 다만 장기 관점에서 보면 기회일 수도 있으니 " \
                    f"차분하게 분석하는 것이 중요하네. 투자의 최종 책임은 자네에게 있다네."
        else:
            answer = f"{ticker}는 현재 ${price:.2f}로 전일과 비슷한 수준이네. " \
                    f"변동이 크지 않으니 안정적이라고 볼 수 있지만, 큰 수익 기대는 어려울 수 있네. " \
                    f"투자의 최종 책임은 자네에게 있다네."
        
        return {"question": question, "answer": answer}
    
    def generate_analyst_conversation(self, stock_data: Dict[str, Any]) -> Dict[str, str]:
        """데이터 분석가 스타일 대화 생성"""
        ticker = stock_data.get("ticker", "N/A")
        price = stock_data.get("currentPrice") or stock_data.get("price", 0)
        change = stock_data.get("changePercent") or stock_data.get("change", 0)
        volume = stock_data.get("volume", 0)
        
        questions = [
            f"{ticker}의 현재 상황을 분석해줘",
            f"{ticker} 주가 분석 부탁해",
            f"{ticker}는 어떤가요?",
            f"{ticker} 투자 의견은?",
        ]
        
        import random
        question = random.choice(questions)
        
        answer = f"{ticker}의 현재가는 ${price:.2f}이며, 전일 대비 {change:+.2f}% 변동했습니다. "
        
        if volume > 0:
            answer += f"거래량은 {volume:,}주로 기록되었습니다. "
        
        if abs(change) > 3:
            answer += f"변동성이 크므로 리스크 관리가 중요합니다. "
        elif abs(change) > 1:
            answer += f"적정한 변동 범위 내에 있습니다. "
        else:
            answer += f"변동성이 낮아 안정적인 수준입니다. "
        
        answer += f"포트폴리오에 포함할 경우 적절한 비중 조절을 권장합니다. " \
                 f"투자의 최종 책임은 투자자에게 있습니다."
        
        return {"question": question, "answer": answer}
    
    def generate_friend_conversation(self, stock_data: Dict[str, Any]) -> Dict[str, str]:
        """친구 조언자 스타일 대화 생성"""
        ticker = stock_data.get("ticker", "N/A")
        price = stock_data.get("currentPrice") or stock_data.get("price", 0)
        change = stock_data.get("changePercent") or stock_data.get("change", 0)
        
        questions = [
            f"{ticker} 지금 사도 될까?",
            f"{ticker} 어때?",
            f"{ticker} 투자해도 돼?",
            f"{ticker} 괜찮아?",
        ]
        
        import random
        question = random.choice(questions)
        
        if abs(change) > 3:
            answer = f"야, {ticker}는 지금 ${price:.2f}인데 전일 대비 {change:+.2f}%나 변동했어. " \
                    f"변동성이 크니까 급하게 결정하지 말고 좀 더 지켜보는 게 나을 것 같아. " \
                    f"차트도 보고, 뉴스도 확인해보고 나서 결정하는 게 좋겠어. " \
                    f"결국 결정은 네가 해야 해."
        elif change > 0:
            answer = f"{ticker}는 현재 ${price:.2f}로 전일 대비 {change:.2f}% 올랐어. " \
                    f"상승 추세인 것 같은데, 이미 많이 올랐을 수도 있으니까 조심해야 해. " \
                    f"조정이 올 때 기다렸다가 사는 것도 방법이야. " \
                    f"결국 결정은 네가 해야 해."
        elif change < 0:
            answer = f"{ticker}는 현재 ${price:.2f}로 전일 대비 {abs(change):.2f}% 내렸어. " \
                    f"하락 중이니까 급하게 사지 말고 좀 더 기다려보는 게 나을 것 같아. " \
                    f"더 내릴 수도 있으니까 인내심을 갖는 게 중요해. " \
                    f"결국 결정은 네가 해야 해."
        else:
            answer = f"{ticker}는 현재 ${price:.2f}로 비교적 안정적인 수준이야. " \
                    f"큰 변동은 없는데, 이 정도면 큰 기대는 어려울 수도 있어. " \
                    f"다른 기회를 노리는 것도 방법이야. " \
                    f"결국 결정은 네가 해야 해."
        
        return {"question": question, "answer": answer}
    
    def generate_conversations(
        self,
        tickers: List[str],
        persona: str,
        count: int = 100,
        include_context: bool = False
    ) -> List[Dict[str, Any]]:
        """금융 데이터 기반 대화 생성"""
        conversations = []
        
        # 각 티커별로 데이터 조회
        stock_data_list = []
        for ticker in tickers:
            stock_data = self.get_stock_quote(ticker)
            if stock_data:
                stock_data["ticker"] = ticker
                stock_data_list.append(stock_data)
        
        if not stock_data_list:
            print("⚠️  조회된 종목 데이터가 없습니다.")
            return []
        
        # 대화 생성
        per_ticker = max(1, count // len(stock_data_list))
        
        for stock_data in stock_data_list:
            for _ in range(per_ticker):
                if persona == "sage":
                    conv = self.generate_sage_conversation(stock_data)
                elif persona == "analyst":
                    conv = self.generate_analyst_conversation(stock_data)
                else:  # friend
                    conv = self.generate_friend_conversation(stock_data)
                
                # 컨텍스트 포함 (선택적)
                if include_context:
                    conv["context"] = {
                        "stocks": [{
                            "ticker": stock_data.get("ticker"),
                            "currentPrice": stock_data.get("currentPrice") or stock_data.get("price"),
                            "changePercent": stock_data.get("changePercent") or stock_data.get("change"),
                            "timestamp": datetime.now().isoformat()
                        }]
                    }
                
                conversations.append(conv)
        
        return conversations


def main():
    parser = argparse.ArgumentParser(description="금융 데이터 기반 대화 생성")
    parser.add_argument(
        '--backend-url',
        type=str,
        default='http://localhost:8080',
        help='Spring Backend Base URL'
    )
    parser.add_argument(
        '--token',
        type=str,
        help='인증 토큰 (선택적)'
    )
    parser.add_argument(
        '--persona',
        type=str,
        choices=['sage', 'analyst', 'friend'],
        required=True,
        help='페르소나 타입'
    )
    parser.add_argument(
        '--tickers',
        type=str,
        nargs='+',
        default=['AAPL', 'TSLA', 'MSFT', 'GOOGL', 'NVDA'],
        help='종목 티커 리스트'
    )
    parser.add_argument(
        '--count',
        type=int,
        default=100,
        help='생성할 대화 개수'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='출력 파일 경로'
    )
    parser.add_argument(
        '--include-context',
        action='store_true',
        help='금융 데이터 컨텍스트 포함'
    )
    
    args = parser.parse_args()
    
    generator = FinancialDataGenerator(args.backend_url, args.token)
    
    print(f"📊 {args.persona} 페르소나 대화 생성 중...")
    print(f"   종목: {', '.join(args.tickers)}")
    print(f"   목표 개수: {args.count}개")
    
    conversations = generator.generate_conversations(
        args.tickers,
        args.persona,
        args.count,
        args.include_context
    )
    
    if not conversations:
        print("❌ 생성된 대화가 없습니다.")
        return
    
    # JSONL로 저장
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with jsonlines.open(output_path, mode='w') as writer:
        for conv in conversations:
            writer.write(conv)
    
    print(f"✅ {len(conversations)}개 대화를 {output_path}에 저장했습니다.")


if __name__ == '__main__':
    main()
