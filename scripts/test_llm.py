#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM 서비스 작동 여부 확인 스크립트

사용법:
  python scripts/test_llm.py
  python scripts/test_llm.py --gateway http://localhost:9000

필수: 프로젝트 루트에 .env 또는 env 파일에 AI_INTERNAL_SECRET 정의
"""

import argparse
import json
import os
import sys
from pathlib import Path

# 프로젝트 루트 (스크립트 기준 상위)
ROOT = Path(__file__).resolve().parent.parent


def load_secret() -> str:
    """ .env 또는 env 파일에서 AI_INTERNAL_SECRET 로드 """
    for name in (".env", "env"):
        p = ROOT / name
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                if k.strip() == "AI_INTERNAL_SECRET":
                    return v.strip().strip('"').strip("'")
    return os.getenv("AI_INTERNAL_SECRET", "")


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM 서비스 작동 테스트")
    parser.add_argument(
        "--gateway",
        default="http://localhost:9000",
        help="AI Gateway URL (기본: http://localhost:9000)",
    )
    parser.add_argument(
        "--skip-chat",
        action="store_true",
        help="헬스체크만 수행하고 채팅 요청 생략",
    )
    args = parser.parse_args()
    base = args.gateway.rstrip("/")

    from urllib.request import Request, urlopen
    from urllib.error import HTTPError, URLError

    print("=== 1. AI Gateway 헬스체크 ===")
    try:
        with urlopen(f"{base}/health", timeout=10) as r:
            data = json.loads(r.read().decode("utf-8"))
            print(json.dumps(data, indent=2, ensure_ascii=False))
            backends = data.get("backends", {})
            if data.get("status") == "healthy" and backends.get("dolphin8b") == "healthy":
                print("-> dolphin8b(LLM) 정상\n")
            elif data.get("status") == "healthy":
                print("-> Gateway는 정상, Backend 일부 비정상:", backends, "\n")
            else:
                print("-> Gateway/Backend 상태 확인 필요\n")
    except (URLError, HTTPError, OSError) as e:
        print(f"실패: {e}")
        print("  docker compose up -d 후 다시 시도하세요.\n")
        sys.exit(1)

    if args.skip_chat:
        print("--skip-chat: 채팅 테스트 생략")
        return

    secret = load_secret()
    if not secret:
        print("=== 2. 채팅 API 테스트 (건너뜀) ===")
        print("  AI_INTERNAL_SECRET을 .env 또는 env에 설정한 뒤 다시 실행하세요.")
        return

    print("=== 2. 채팅 API 테스트 (/api/v1/ai/chat) ===")
    body = {
        "useCase": "generic",
        "userId": 1,
        "message": "안녕, 오늘 투자 조언 한 줄만 해줘.",
        "persona": "sage",
    }
    req = Request(
        f"{base}/api/v1/ai/chat",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "X-Internal-Token": secret,
        },
        method="POST",
    )
    try:
        with urlopen(req, timeout=90) as r:
            data = json.loads(r.read().decode("utf-8"))
            print("응답 모델:", data.get("model", "-"))
            print("응답 내용:", data.get("content", "-")[:500])
            if data.get("content"):
                print("\n-> LLM 채팅 정상 작동\n")
    except HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        print(f"HTTP {e.code}: {err_body[:500]}")
        if e.code == 401:
            print("  X-Internal-Token(AI_INTERNAL_SECRET) 값을 확인하세요.")
        sys.exit(1)
    except (URLError, OSError) as e:
        print(f"요청 실패: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
