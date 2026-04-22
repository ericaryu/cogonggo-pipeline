# 이 파일이 존재하는 이유:
# pipeline.py가 의존하는 진입점을 유지하면서 실제 로직을 위임한다.
# 직접 구현 없음 — pass1_runner / pass2_runner 의 얇은 파사드.

import asyncio
import logging

from analyzers import pass1_runner, pass2_runner

log = logging.getLogger(__name__)


def run_pass1(category: str | None = None, force: bool = False) -> None:
    """Pass1: 공고 1건씩 구조화 추출 (gpt-4o-mini, 병렬).
    category=None 이면 data/compressed/ 전체 카테고리 실행.
    """
    asyncio.run(pass1_runner.run(category=category, force=force))


def run_pass2(category: str | None = None, force: bool = False) -> None:
    """Pass2: Pass1 캐시 집계 → 외주 전략 JSON 생성 (gpt-4.1, 직렬).
    category=None 이면 cache/pass1_*.jsonl 전체 카테고리 실행.
    """
    pass2_runner.run(category=category, force=force)


def run_all(category: str | None = None, force: bool = False) -> None:
    """Pass1 → Pass2 순차 실행."""
    run_pass1(category=category, force=force)
    run_pass2(category=category, force=force)


# 구 버전 호환: pipeline.py의 --step analyze가 analyze_all()을 호출하던 경우 대비
def analyze_all() -> None:
    """하위 호환용. run_all()과 동일."""
    run_all()
