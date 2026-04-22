# 이 파일이 존재하는 이유:
# Pass1 캐시(jsonl)를 로드해 카테고리 단위 전략 분석을 실행한다.
# LLM 재호출 없이 통계를 직접 계산해 stats_block으로 주입하고,
# LLMPass2Output + 집계 통계를 합쳐 Pass2Output을 저장한다.

import json
import logging
import re
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

from analyzers.config import PASS2_SUMMARY_THRESHOLD
from analyzers.llm_client import call_pass2
from analyzers.schemas import LLMPass2Output, Pass1Output, Pass2Output

load_dotenv()
log = logging.getLogger(__name__)

CACHE_DIR = Path("cache")
OUTPUT_DIR = Path("output")


# ── 유틸 ──────────────────────────────────────────────────────────────────────

def _safe_name(category: str) -> str:
    return re.sub(r"[^\w가-힣]", "_", category)


def _load_pass1_results(cache_path: Path) -> list[Pass1Output]:
    results: list[Pass1Output] = []
    for line in cache_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            results.append(Pass1Output.model_validate_json(line))
        except Exception as exc:
            log.warning("[pass2] Pass1 jsonl 파싱 실패 (스킵): %s", exc)
    return results


# ── 통계 계산 (LLM이 숫자를 세지 않도록 러너가 사전 계산) ─────────────────────

def _compute_stats(results: list[Pass1Output]) -> dict:
    n = len(results)
    high_count = sum(1 for r in results if r.outsource_fit == "high")
    medium_count = sum(1 for r in results if r.outsource_fit == "medium")
    low_count = sum(1 for r in results if r.outsource_fit == "low")
    ax_count = sum(1 for r in results if r.ax_ai.keywords)

    task_dist = Counter(r.task_nature for r in results)
    career_dist = Counter(r.career_level for r in results)
    company_counts = Counter(r.company for r in results)

    all_jtbd: list[str] = []
    for r in results:
        all_jtbd.extend(r.jtbd)
    jtbd_counts = Counter(all_jtbd)

    all_skills: list[str] = []
    for r in results:
        all_skills.extend(r.skills)
    skill_counts = Counter(all_skills)

    all_ax_keywords: list[str] = []
    for r in results:
        all_ax_keywords.extend(r.ax_ai.keywords)
    ax_keyword_counts = Counter(all_ax_keywords)

    return {
        "n_jobs": n,
        "high_outsource_ratio": round(high_count / n, 3) if n else 0.0,
        "n_with_ax_mention": ax_count,
        "outsource_dist": {"high": high_count, "medium": medium_count, "low": low_count},
        "task_dist": dict(task_dist.most_common()),
        "career_dist": dict(career_dist.most_common()),
        "top_companies": dict(company_counts.most_common(10)),
        "top_jtbd": jtbd_counts.most_common(20),   # LLM 클러스터링 힌트
        "top_skills": skill_counts.most_common(15),
        "top_ax_keywords": ax_keyword_counts.most_common(10),
    }


def _format_stats_block(stats: dict, category: str) -> str:
    """LLM에 주입할 사전 집계 통계 텍스트 블록."""
    n = stats["n_jobs"]
    ratio = stats["high_outsource_ratio"]
    ax = stats["n_with_ax_mention"]
    od = stats["outsource_dist"]
    td = stats["task_dist"]
    cd = stats["career_dist"]

    def _dist_str(d: dict) -> str:
        return " / ".join(f"{k} {v}건" for k, v in d.items())

    top_cos = ", ".join(
        f"{co}({cnt}건)" for co, cnt in stats["top_companies"].items()
    )

    top_jtbd_lines = "\n".join(
        f"  · {phrase} ({cnt}건)" for phrase, cnt in stats["top_jtbd"]
    )

    top_skills = ", ".join(f"{s}({c}건)" for s, c in stats["top_skills"])

    top_ax = ", ".join(f"{k}({c}건)" for k, c in stats["top_ax_keywords"]) or "없음"

    return f"""- 카테고리: {category}
- 총 공고 수: {n}건
- outsource_fit 분포: high {od['high']}건 / medium {od['medium']}건 / low {od['low']}건
- outsource_fit=high 비율: {ratio:.1%} (jtbd_clusters.count 산정 시 이 수치 인용)
- AI/AX 키워드 언급 공고: {ax}건 ({ax/n:.1%})  ← n_with_ax_mention
- task_nature 분포: {_dist_str(td)}
- career_level 분포: {_dist_str(cd)}
- 상위 기업 (공고 수 기준): {top_cos}
- 자주 등장한 기술/툴: {top_skills}
- AI 관련 키워드 상위: {top_ax}
- JTBD 패턴 힌트 (의미론적 클러스터링 참고, count는 이 빈도 기반으로 산정):
{top_jtbd_lines}"""


# ── jobs_block 포맷 ────────────────────────────────────────────────────────────

def _format_jobs_block(results: list[Pass1Output], threshold: int) -> str:
    """
    건수가 threshold 이하면 Pass1Output 전체 JSON,
    초과하면 핵심 필드만 압축해 토큰 절약.
    """
    if len(results) <= threshold:
        rows = [r.model_dump() for r in results]
    else:
        rows = [
            {
                "job_id": r.job_id,
                "company": r.company,
                "title": r.title,
                "jtbd": r.jtbd,
                "outsource_fit": r.outsource_fit,
                "task_nature": r.task_nature,
                "career_level": r.career_level,
                "ax_keywords": r.ax_ai.keywords,
                "ax_intent": r.ax_ai.automation_intent,
                "kmong_fit": r.ax_ai.kmong_service_fit,
            }
            for r in results
        ]
    return json.dumps(rows, ensure_ascii=False, indent=2)


# ── 카테고리 단위 실행 ─────────────────────────────────────────────────────────

def run_category(category: str, force: bool = False) -> Pass2Output | None:
    safe = _safe_name(category)
    cache_path = CACHE_DIR / f"pass1_{safe}.jsonl"
    out_path = OUTPUT_DIR / f"pass2_{safe}.json"

    if not force and out_path.exists():
        print(f"[pass2] {category}: 캐시 존재, 스킵 (--force로 재실행)")
        return None

    if not cache_path.exists():
        log.error("[pass2] Pass1 캐시 없음: %s", cache_path)
        return None

    results = _load_pass1_results(cache_path)
    if not results:
        log.warning("[pass2] %s: Pass1 결과 0건, 스킵", category)
        return None

    print(f"[pass2] {category}: Pass1 {len(results)}건 로드 → 집계 중...")

    stats = _compute_stats(results)
    stats_block = _format_stats_block(stats, category)
    jobs_block = _format_jobs_block(results, PASS2_SUMMARY_THRESHOLD)

    if len(results) > PASS2_SUMMARY_THRESHOLD:
        print(
            f"[pass2] {category}: {len(results)}건 > {PASS2_SUMMARY_THRESHOLD} → 압축 포맷 사용"
        )

    print(f"[pass2] {category}: LLM 호출 중 (model=gpt-4.1)...")

    llm_out: LLMPass2Output = call_pass2(category, stats_block, jobs_block)

    # LLM 생성 결과 + 러너 집계 통계 병합 → Pass2Output
    output = Pass2Output(
        **llm_out.model_dump(),
        category=category,
        n_jobs=stats["n_jobs"],
        high_outsource_ratio=stats["high_outsource_ratio"],
        n_with_ax_mention=stats["n_with_ax_mention"],
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        output.model_dump_json(indent=2), encoding="utf-8"
    )
    print(f"[pass2] {category}: 완료 → {out_path}")
    return output


# ── 전체 실행 ──────────────────────────────────────────────────────────────────

def run(category: str | None = None, force: bool = False) -> None:
    """category=None 이면 cache/ 아래 pass1_*.jsonl 전체 실행."""
    if category:
        run_category(category, force=force)
        return

    cache_files = sorted(CACHE_DIR.glob("pass1_*.jsonl"))

    # pass1_failed_*.jsonl 제외
    cache_files = [f for f in cache_files if not f.name.startswith("pass1_failed_")]

    if not cache_files:
        print("[pass2] cache/ 에 pass1_*.jsonl 없음. --step analyze-pass1 먼저 실행하세요.")
        return

    for cache_file in cache_files:
        # "pass1_마케팅.jsonl" → "마케팅"
        cat = cache_file.stem.removeprefix("pass1_")
        run_category(cat, force=force)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default=None, help="특정 카테고리만 실행")
    parser.add_argument("--force", action="store_true", help="출력 파일 덮어쓰기")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run(category=args.category, force=args.force)
