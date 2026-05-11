"""
2단계: 카테고리별 외주 적합도 스코어링

입력: output/step1_tagged.csv
출력:
  - output/step2_outsourcing_matrix.csv   (카테고리 × 4개 차원 + 외주적합도)
  - output/step2_outsourcing_report.json  (LLM 상세 분석)

스코어링 4개 차원 (각 1-5점):
  반복성      ↑ 높을수록 외주 적합
  전략민감도  ↓ 높을수록 외주 부적합
  스킬희소성  ↑ 높을수록 외주 적합 (내부 확보 어려움)
  브랜드의존도 ↓ 높을수록 외주 부적합

외주적합도 = 반복성 + 스킬희소성 - 전략민감도 - 브랜드의존도
  3 이상 → 높음 / -2~2 → 중간 / -3 이하 → 낮음

Usage:
    python -m analyzers.step2_outsourcing_scorer
"""

import csv
import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

load_dotenv()
log = logging.getLogger(__name__)

INPUT_CSV     = Path("output/step1_tagged.csv")
OUTPUT_MATRIX = Path("output/step2_outsourcing_matrix.csv")
OUTPUT_REPORT = Path("output/step2_outsourcing_report.json")

MODEL      = "gpt-4.1"
MAX_TOKENS = 1024
SAMPLE_N   = 15  # 카테고리당 LLM에 보낼 최대 공고 수

CATEGORIES = ["브랜드_PR", "콘텐츠", "퍼포먼스", "기획_전략", "글로벌", "채널운영", "CRM"]

SYSTEM_PROMPT = """당신은 뷰티 브랜드 마케팅 운영 전략 전문가입니다.
아래 채용공고 데이터를 바탕으로 해당 업무 카테고리의 외주 적합성을 평가합니다.

[4개 평가 차원 — 각 1~5점]
1. 반복성 (1=매번 새로운 판단 필요, 5=패턴화·양산 가능)
2. 전략민감도 (1=실행 위주, 5=브랜드 핵심 전략 직결)
3. 스킬희소성 (1=흔한 스킬, 5=시장에서 구하기 어려운 전문 스킬)
4. 브랜드의존도 (1=브랜드 무관, 5=브랜드 톤·맥락 깊이 이해 필수)

[외주 논리 3가지를 균형있게 반영]
- 인하우스를 뽑는 이유: 브랜드 이해, 즉시 대응, 내부 데이터 접근
- 외주를 줄 수밖에 없는 이유: 전문 스킬 확보 어려움, 물량 피크, 채용 공백
- 에이전시에 대한 흔한 불만: 소통 비용, 품질 편차, 브랜드 이해 부족

[출력 규칙]
- 점수는 채용공고에서 실제 확인되는 업무 내용 기반
- 외주가능업무: 공고에 등장한 구체적 업무 단위로 작성 (추상적 표현 금지)
- 외주장벽: 가장 핵심적인 1~2가지만"""

USER_TEMPLATE = """카테고리: {category}
공고 수: {total}건 (신입 {new}건 / 경력 {exp}건 / 신입+경력 {both}건)

[공고 샘플 {n}건 — 주요업무 + 자격요건]
{samples}"""


# ── Pydantic 스키마 ──────────────────────────────────────────────────────────

class CategoryScore(BaseModel):
    반복성: int
    전략민감도: int
    스킬희소성: int
    브랜드의존도: int
    핵심근거: str
    외주가능업무: list[str]
    외주장벽: str


# ── LLM 호출 ─────────────────────────────────────────────────────────────────

_client: OpenAI | None = None

def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


def _is_retryable(exc: BaseException) -> bool:
    from openai import APIStatusError, RateLimitError
    return isinstance(exc, RateLimitError) or (
        isinstance(exc, APIStatusError) and exc.status_code >= 500
    )


@retry(
    retry=retry_if_exception(_is_retryable),
    wait=wait_exponential(multiplier=2, min=2, max=16),
    stop=stop_after_attempt(3),
    reraise=True,
)
def _call_llm(user_msg: str) -> CategoryScore:
    client = _get_client()
    response = client.beta.chat.completions.parse(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        response_format=CategoryScore,
    )
    parsed = response.choices[0].message.parsed
    if parsed is None:
        raise ValueError("parse 실패")
    return parsed


# ── 데이터 준비 ───────────────────────────────────────────────────────────────

def _load_by_category(csv_path: Path) -> dict[str, list[dict]]:
    by_cat: dict[str, list[dict]] = defaultdict(list)
    with csv_path.open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            cats = [c.strip() for c in row.get("업무카테고리", "").split(",") if c.strip()]
            for cat in cats:
                by_cat[cat].append(row)
    return by_cat


def _build_samples(rows: list[dict], n: int) -> str:
    sampled = random.sample(rows, min(n, len(rows)))
    parts = []
    for i, r in enumerate(sampled, 1):
        task = r.get("주요업무", "")[:300]
        qual = r.get("자격요건", "")[:200]
        parts.append(f"[{i}] {r.get('포지션명','')}\n주요업무: {task}\n자격요건: {qual}")
    return "\n\n".join(parts)


# ── 외주적합도 계산 ───────────────────────────────────────────────────────────

def _calc_score(s: CategoryScore) -> tuple[int, str]:
    raw = s.반복성 + s.스킬희소성 - s.전략민감도 - s.브랜드의존도
    level = "높음" if raw >= 3 else ("낮음" if raw <= -3 else "중간")
    return raw, level


# ── 출력 ─────────────────────────────────────────────────────────────────────

def _save_matrix(results: list[dict]) -> None:
    OUTPUT_MATRIX.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "카테고리", "공고수",
        "반복성(1-5)", "전략민감도(1-5)", "스킬희소성(1-5)", "브랜드의존도(1-5)",
        "외주적합도(-8~8)", "외주적합도_레벨",
        "핵심근거", "외주장벽",
    ]
    with OUTPUT_MATRIX.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def _print_matrix(results: list[dict]) -> None:
    print(f"\n{'카테고리':<12} {'공고':>4} {'반복':>4} {'전략':>4} {'스킬':>4} {'브랜드':>6} {'적합도':>6} {'레벨':<6}")
    print("─" * 60)
    for r in sorted(results, key=lambda x: -x["외주적합도(-8~8)"]):
        print(
            f"{r['카테고리']:<12} "
            f"{r['공고수']:>4} "
            f"{r['반복성(1-5)']:>4} "
            f"{r['전략민감도(1-5)']:>4} "
            f"{r['스킬희소성(1-5)']:>4} "
            f"{r['브랜드의존도(1-5)']:>6} "
            f"{r['외주적합도(-8~8)']:>+6} "
            f"{r['외주적합도_레벨']:<6}"
        )


# ── 메인 ─────────────────────────────────────────────────────────────────────

def run() -> None:
    if not INPUT_CSV.exists():
        print(f"[scorer] {INPUT_CSV} 없음. step1 먼저 실행하세요.")
        return

    by_cat = _load_by_category(INPUT_CSV)
    matrix_rows = []
    report = {}

    for cat in CATEGORIES:
        rows = by_cat.get(cat, [])
        if not rows:
            print(f"[scorer] {cat}: 데이터 없음 — 스킵")
            continue

        exp_dist = defaultdict(int)
        for r in rows:
            exp_dist[r.get("경력구분", "")] += 1

        samples_text = _build_samples(rows, SAMPLE_N)
        user_msg = USER_TEMPLATE.format(
            category=cat,
            total=len(rows),
            new=exp_dist["신입"],
            exp=exp_dist["경력"],
            both=exp_dist["신입+경력"],
            n=min(SAMPLE_N, len(rows)),
            samples=samples_text,
        )

        print(f"[scorer] {cat} ({len(rows)}건) 분석 중...", end=" ", flush=True)
        score = _call_llm(user_msg)
        raw, level = _calc_score(score)
        print(f"적합도 {raw:+d} ({level})")

        matrix_rows.append({
            "카테고리": cat,
            "공고수": len(rows),
            "반복성(1-5)": score.반복성,
            "전략민감도(1-5)": score.전략민감도,
            "스킬희소성(1-5)": score.스킬희소성,
            "브랜드의존도(1-5)": score.브랜드의존도,
            "외주적합도(-8~8)": raw,
            "외주적합도_레벨": level,
            "핵심근거": score.핵심근거,
            "외주장벽": score.외주장벽,
        })

        report[cat] = {
            "공고수": len(rows),
            "경력분포": dict(exp_dist),
            "점수": {
                "반복성": score.반복성,
                "전략민감도": score.전략민감도,
                "스킬희소성": score.스킬희소성,
                "브랜드의존도": score.브랜드의존도,
                "외주적합도": raw,
                "레벨": level,
            },
            "핵심근거": score.핵심근거,
            "외주가능업무": score.외주가능업무,
            "외주장벽": score.외주장벽,
        }

    _save_matrix(matrix_rows)
    OUTPUT_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2))

    print("\n" + "=" * 60)
    print("외주 적합도 매트릭스 (반복↑ 스킬↑ 전략↓ 브랜드↓)")
    print("=" * 60)
    _print_matrix(matrix_rows)
    print(f"\n[output] 매트릭스 → {OUTPUT_MATRIX}")
    print(f"[output] 상세 리포트 → {OUTPUT_REPORT}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    run()
