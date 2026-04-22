"""
cogonggo.co/recruit 채용공고 분석 파이프라인

Usage:
    python pipeline.py                          # 전체 실행 (scrape → classify → analyze)
    python pipeline.py --step scrape            # 1~2단계: 공고 수집
    python pipeline.py --step classify          # 3단계: 카테고리 분류
    python pipeline.py --step analyze           # 4단계: Pass1 + Pass2 전체
    python pipeline.py --step analyze-pass1     # 4a단계: 공고 구조화 추출만
    python pipeline.py --step analyze-pass2     # 4b단계: 전략 분석만 (Pass1 캐시 필요)

    python pipeline.py --step analyze --category 마케팅        # 특정 카테고리만
    python pipeline.py --step analyze-pass1 --force            # 캐시 무시 재실행
"""
import argparse
import asyncio
import json
from pathlib import Path

RAW_PATH = Path("data/raw/jobs_raw.json")


def run(step: str = "all", category: str | None = None, force: bool = False) -> None:
    if step in ("scrape", "all"):
        print("=" * 50)
        print("STEP 1-2: 공고 수집 (API 탐지 → 전체 수집)")
        print("=" * 50)
        from scraper import scrape
        jobs = asyncio.run(scrape())
        print(f"\n수집 완료: {len(jobs)}건\n")

    if step in ("classify", "all"):
        if not RAW_PATH.exists():
            print(f"[pipeline] {RAW_PATH} 없음. --step scrape 먼저 실행하세요.")
            return
        print("=" * 50)
        print("STEP 3: 카테고리 분류 + 압축")
        print("=" * 50)
        from classifier import classify_and_compress
        jobs = json.loads(RAW_PATH.read_text())
        categorized = classify_and_compress(jobs)
        total = sum(len(v) for v in categorized.values())
        print(f"\n분류 완료: {total}건 → {len(categorized)}개 카테고리\n")

    if step in ("analyze", "all"):
        print("=" * 50)
        print("STEP 4a: Pass1 공고 구조화 추출 (gpt-4o-mini, 병렬)")
        print("=" * 50)
        from analyzer import run_pass1
        run_pass1(category=category, force=force)

        print("=" * 50)
        print("STEP 4b: Pass2 외주 전략 분석 (gpt-4.1, 직렬)")
        print("=" * 50)
        from analyzer import run_pass2
        run_pass2(category=category, force=force)
        print("\n분석 완료. 출력: output/\n")

    if step == "analyze-pass1":
        print("=" * 50)
        print("STEP 4a: Pass1 공고 구조화 추출 (gpt-4o-mini, 병렬)")
        print("=" * 50)
        from analyzer import run_pass1
        run_pass1(category=category, force=force)

    if step == "analyze-pass2":
        print("=" * 50)
        print("STEP 4b: Pass2 외주 전략 분석 (gpt-4.1, 직렬)")
        print("=" * 50)
        from analyzer import run_pass2
        run_pass2(category=category, force=force)
        print("\n분석 완료. 출력: output/\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="cogonggo.co 채용공고 분석 파이프라인")
    parser.add_argument(
        "--step",
        choices=["scrape", "classify", "analyze", "analyze-pass1", "analyze-pass2", "all"],
        default="all",
        help="실행할 단계 (기본: all)",
    )
    parser.add_argument(
        "--category",
        default=None,
        metavar="CATEGORY",
        help="특정 카테고리만 실행 (예: --category 마케팅). 생략 시 전체 카테고리.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="캐시/출력 파일 무시하고 재실행",
    )
    args = parser.parse_args()
    run(step=args.step, category=args.category, force=args.force)
