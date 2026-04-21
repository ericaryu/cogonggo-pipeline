"""
cogonggo.co/recruit 채용공고 분석 파이프라인

Usage:
    python pipeline.py                   # 전체 실행 (scrape → classify → analyze)
    python pipeline.py --step scrape     # 1~2단계만
    python pipeline.py --step classify   # 3단계만
    python pipeline.py --step analyze    # 4단계만
"""
import argparse
import asyncio
import json
from pathlib import Path

RAW_PATH = Path("data/raw/jobs_raw.json")


def run(step: str = "all") -> None:
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
        print("STEP 4: Claude API 카테고리별 분석")
        print("=" * 50)
        from analyzer import analyze_all
        analyze_all()
        print("\n분석 완료. 리포트: data/reports/\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="cogonggo.co 채용공고 분석 파이프라인")
    parser.add_argument(
        "--step",
        choices=["scrape", "classify", "analyze", "all"],
        default="all",
        help="실행할 단계 (기본: all)",
    )
    args = parser.parse_args()
    run(args.step)
