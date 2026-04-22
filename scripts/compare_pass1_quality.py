"""
Pass1 품질 비교: 상세 데이터 유무에 따른 차이 측정
- 버전 A (before): title + company + job + experienceLevel만
- 버전 B (after) : + mainTask, qualifications, preferences, benefits, positionDescription
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

from analyzers.llm_client import call_pass1
from analyzers.schemas import Pass1Output

load_dotenv()

ENRICHED_PATH = Path("data/raw/jobs_enriched.json")
REPORT_PATH = Path("output/quality_compare.json")

DETAIL_FIELDS = [
    "positionDescription", "mainTask", "qualifications",
    "preferences", "benefits", "hiringProcess", "subJob",
]
TITLE_ONLY_FIELDS = [
    "publicId", "title", "companyName", "job",
    "experienceLevel", "salesCountries", "categories",
]


def _strip_to_title_only(job: dict) -> dict:
    return {k: job[k] for k in TITLE_ONLY_FIELDS if k in job}


def _has_detail(job: dict) -> bool:
    return any(job.get(f) for f in DETAIL_FIELDS)


async def _run_pair(job: dict, sem: asyncio.Semaphore) -> dict:
    pid = job.get("publicId", "?")
    company = job.get("companyName", "?")
    title = job.get("title", "?")

    job_a = _strip_to_title_only(job)
    job_b = {k: v for k, v in job.items() if k in TITLE_ONLY_FIELDS + DETAIL_FIELDS}

    async with sem:
        res_a: Pass1Output = await call_pass1(json.dumps(job_a, ensure_ascii=False))
    async with sem:
        res_b: Pass1Output = await call_pass1(json.dumps(job_b, ensure_ascii=False))

    return {
        "publicId": pid,
        "company": company,
        "title": title,
        "A_title_only": res_a.model_dump(),
        "B_with_detail": res_b.model_dump(),
    }


async def main(n: int = 16) -> None:
    enriched = json.load(open(ENRICHED_PATH))

    # 카테고리 다양성 + mainTask 풍부한 것
    candidates = [j for j in enriched if _has_detail(j) and len(j.get("mainTask", "")) > 200]

    by_job: dict[str, list] = {}
    for j in candidates:
        k = j.get("job", "기타")
        by_job.setdefault(k, []).append(j)

    sample: list[dict] = []
    for jobs in sorted(by_job.values(), key=lambda x: -len(x)):
        sample.extend(jobs[:2])
        if len(sample) >= n:
            break
    sample = sample[:n]

    print(f"샘플 {len(sample)}건 비교 시작 (A=title만, B=+상세)...\n")

    sem = asyncio.Semaphore(5)
    results = await asyncio.gather(*[_run_pair(j, sem) for j in sample])

    REPORT_PATH.parent.mkdir(exist_ok=True)
    REPORT_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2))

    # ── 콘솔 요약 출력 ──────────────────────────────────────────────────────────
    fit_changes = {"up": 0, "down": 0, "same": 0}
    ax_gained = 0
    jtbd_len_diff = []

    ORDER = {"high": 2, "medium": 1, "low": 0}

    for r in results:
        a = r["A_title_only"]
        b = r["B_with_detail"]

        fa, fb = ORDER[a["outsource_fit"]], ORDER[b["outsource_fit"]]
        if fb > fa:
            fit_changes["up"] += 1
        elif fb < fa:
            fit_changes["down"] += 1
        else:
            fit_changes["same"] += 1

        if not a["ax_ai"]["keywords"] and b["ax_ai"]["keywords"]:
            ax_gained += 1

        jtbd_len_diff.append(len(b["jtbd"]) - len(a["jtbd"]))

    avg_jtbd_gain = sum(jtbd_len_diff) / len(jtbd_len_diff)

    print("=" * 60)
    print("outsource_fit 변화 (A→B)")
    print(f"  상승: {fit_changes['up']}건  동일: {fit_changes['same']}건  하락: {fit_changes['down']}건")
    print(f"AX/AI 키워드 신규 탐지 (A 없음→B 있음): {ax_gained}건")
    print(f"jtbd 항목 수 평균 증감: {avg_jtbd_gain:+.1f}개")
    print()

    # 개별 케이스 출력
    for r in results:
        a, b = r["A_title_only"], r["B_with_detail"]
        fa, fb = a["outsource_fit"], b["outsource_fit"]
        fit_tag = "→" if fa == fb else f"↑" if ORDER[fb] > ORDER[fa] else "↓"
        print(f"[{r['company']}] {r['title'][:45]}")
        print(f"  outsource_fit : {fa} {fit_tag} {fb}")
        print(f"  jtbd A({len(a['jtbd'])}): {a['jtbd'][:1]}")
        print(f"  jtbd B({len(b['jtbd'])}): {b['jtbd'][:1]}")
        if a["outsource_reason"] != b["outsource_reason"]:
            print(f"  reason A: {(a['outsource_reason'] or '')[:80]}")
            print(f"  reason B: {(b['outsource_reason'] or '')[:80]}")
        if b["ax_ai"]["keywords"] and not a["ax_ai"]["keywords"]:
            print(f"  🔍 AX 신규: {b['ax_ai']['keywords']}")
        print()

    print(f"상세 결과 저장 → {REPORT_PATH}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=16, help="비교할 샘플 건수")
    args = parser.parse_args()
    asyncio.run(main(n=args.n))
