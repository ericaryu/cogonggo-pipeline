"""
2.5단계: Task 단위 추출 + 외주 적합도 재평가

Pass A — Task 추출 (gpt-4o-mini): 카테고리별 공고 주요업무 → 대표 task 10~15개
Pass B — Task 스코어링 (gpt-4.1): task별 4개 차원 점수 + 발주단위 설명

입력: output/step1_tagged.csv
출력: output/step2b_tasks.csv
캐시: cache/step2b_cache.json

Usage:
    python -m analyzers.step2b_task_extractor
    python -m analyzers.step2b_task_extractor --force
"""

import argparse
import csv
import json
import logging
import random
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential
from tqdm import tqdm

load_dotenv()
log = logging.getLogger(__name__)

INPUT_CSV    = Path("output/step1_tagged.csv")
OUTPUT_CSV   = Path("output/step2b_tasks.csv")
CACHE_PATH   = Path("cache/step2b_cache.json")

MODEL_A = "gpt-4o-mini"   # Pass A: task 추출
MODEL_B = "gpt-4.1"       # Pass B: task 스코어링

SAMPLE_N = 20  # 카테고리당 공고 샘플 수

CATEGORIES = ["브랜드_PR", "콘텐츠", "퍼포먼스", "기획_전략", "글로벌", "채널운영", "CRM"]

OUTPUT_COLUMNS = [
    "카테고리", "task명",
    "반복성(1-5)", "전략민감도(1-5)", "스킬희소성(1-5)", "브랜드의존도(1-5)",
    "외주적합도(-8~8)", "외주적합도_레벨",
    "발주단위설명",
]

# ── 시스템 프롬프트 ───────────────────────────────────────────────────────────

PASS_A_SYSTEM = """당신은 뷰티 브랜드 마케팅 운영 전문가입니다.
여러 채용공고의 주요업무 텍스트에서 실제 반복 수행되는 task를 추출합니다.

[추출 기준]
- 팀장이 팀원에게 "이거 해줘"라고 지시할 수 있는 수준의 구체적 단위
- 추상적 표현 제외 (예: "전략 수립", "성장 기여" → 불가)
- 가능한 표현 (예: "인플루언서 후보 리스트업 및 1차 컨택", "광고 소재 리사이징 및 매체별 업로드")
- 10~15개 추출. 중복 제거, 유사한 것은 합치기
- task명은 동사형으로 끝낼 것 (예: "~하기", "~운영", "~작성")"""

PASS_A_USER = """카테고리: {category}
공고 {n}건의 주요업무 모음:

{tasks_text}

위 텍스트에서 이 카테고리 마케터가 반복적으로 수행하는 구체적 task 10~15개를 추출하세요."""

PASS_B_SYSTEM = """당신은 뷰티 브랜드 마케팅 외주화 전략 전문가입니다.
각 task에 대해 외주 적합성을 4개 차원으로 평가합니다.

[4개 차원 — 각 1~5점]
1. 반복성: 1=매번 새로운 판단 필요, 5=패턴화·양산 가능
2. 전략민감도: 1=단순 실행, 5=브랜드 핵심 전략 직결
3. 스킬희소성: 1=누구나 가능, 5=전문 스킬 필수
4. 브랜드의존도: 1=브랜드 무관, 5=브랜드 톤·맥락 깊이 필요

[발주단위설명]
- 실제로 외주 발주할 경우 어떤 형태인지 1문장 (예: "월 OO건 단위 건당 발주", "캠페인 1회 단위 프리랜서 계약")
- 점수에 관계없이 모든 task에 작성

[점수 기준 예시]
- 반복성 5: 광고 소재 리사이징 (규격만 알면 동일 작업 반복)
- 전략민감도 5: 브랜드 연간 마케팅 로드맵 수립
- 스킬희소성 5: 영어 네이티브 수준 글로벌 인플루언서 협상
- 브랜드의존도 5: 브랜드 고유 톤앤매너의 SNS 콘텐츠 기획"""

PASS_B_USER = """카테고리: {category}

아래 task 목록 각각에 대해 4개 차원 점수와 발주단위설명을 작성하세요.

{task_list}"""


# ── Pydantic 스키마 ───────────────────────────────────────────────────────────

class TaskList(BaseModel):
    tasks: list[str]


class TaskScore(BaseModel):
    task: str
    반복성: int
    전략민감도: int
    스킬희소성: int
    브랜드의존도: int
    발주단위설명: str


class CategoryTaskScores(BaseModel):
    scores: list[TaskScore]


# ── Retry ────────────────────────────────────────────────────────────────────

def _is_retryable(exc: BaseException) -> bool:
    from openai import APIStatusError, RateLimitError
    return isinstance(exc, RateLimitError) or (
        isinstance(exc, APIStatusError) and exc.status_code >= 500
    )

_retry = dict(
    retry=retry_if_exception(_is_retryable),
    wait=wait_exponential(multiplier=2, min=2, max=16),
    stop=stop_after_attempt(3),
    reraise=True,
)

# ── LLM 클라이언트 ────────────────────────────────────────────────────────────

_async_client: AsyncOpenAI | None = None
_sync_client: OpenAI | None = None

def _async() -> AsyncOpenAI:
    global _async_client
    if _async_client is None:
        _async_client = AsyncOpenAI()
    return _async_client

def _sync() -> OpenAI:
    global _sync_client
    if _sync_client is None:
        _sync_client = OpenAI()
    return _sync_client


# ── Pass A: Task 추출 ─────────────────────────────────────────────────────────

import asyncio

@retry(**_retry)
async def _extract_tasks(category: str, rows: list[dict]) -> list[str]:
    sampled = random.sample(rows, min(SAMPLE_N, len(rows)))
    tasks_text = "\n\n".join(
        f"[공고 {i+1}] {r.get('포지션명','')}\n{r.get('주요업무','')[:400]}"
        for i, r in enumerate(sampled)
    )
    user_msg = PASS_A_USER.format(
        category=category, n=len(sampled), tasks_text=tasks_text
    )
    resp = await _async().beta.chat.completions.parse(
        model=MODEL_A,
        max_tokens=512,
        messages=[
            {"role": "system", "content": PASS_A_SYSTEM},
            {"role": "user",   "content": user_msg},
        ],
        response_format=TaskList,
    )
    parsed = resp.choices[0].message.parsed
    if parsed is None:
        raise ValueError(f"Pass A parse 실패: {category}")
    return parsed.tasks


# ── Pass B: Task 스코어링 ─────────────────────────────────────────────────────

@retry(**_retry)
def _score_tasks(category: str, tasks: list[str]) -> list[TaskScore]:
    task_list = "\n".join(f"{i+1}. {t}" for i, t in enumerate(tasks))
    user_msg = PASS_B_USER.format(category=category, task_list=task_list)
    resp = _sync().beta.chat.completions.parse(
        model=MODEL_B,
        max_tokens=2048,
        messages=[
            {"role": "system", "content": PASS_B_SYSTEM},
            {"role": "user",   "content": user_msg},
        ],
        response_format=CategoryTaskScores,
    )
    parsed = resp.choices[0].message.parsed
    if parsed is None:
        raise ValueError(f"Pass B parse 실패: {category}")
    return parsed.scores


# ── 외주적합도 계산 ───────────────────────────────────────────────────────────

def _calc(s: TaskScore) -> tuple[int, str]:
    raw = s.반복성 + s.스킬희소성 - s.전략민감도 - s.브랜드의존도
    level = "높음" if raw >= 3 else ("낮음" if raw <= -3 else "중간")
    return raw, level


# ── 캐시 ─────────────────────────────────────────────────────────────────────

def _load_cache() -> dict:
    if not CACHE_PATH.exists():
        return {}
    try:
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}

def _save_cache(cache: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2))


# ── 저장 ─────────────────────────────────────────────────────────────────────

def _save_csv(all_rows: list[dict]) -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(all_rows)


def _print_summary(all_rows: list[dict]) -> None:
    by_cat = defaultdict(list)
    for r in all_rows:
        by_cat[r["카테고리"]].append(r)

    for cat in CATEGORIES:
        rows = by_cat.get(cat, [])
        if not rows:
            continue
        print(f"\n▶ {cat} ({len(rows)}개 task)")
        for r in sorted(rows, key=lambda x: -x["외주적합도(-8~8)"]):
            score = r["외주적합도(-8~8)"]
            level = r["외주적합도_레벨"]
            marker = "★" if level == "높음" else ("△" if level == "중간" else "✕")
            print(f"  {marker} [{score:+d}] {r['task명']}")
            print(f"       → {r['발주단위설명']}")


# ── 메인 ─────────────────────────────────────────────────────────────────────

async def run(force: bool = False) -> None:
    if not INPUT_CSV.exists():
        print(f"[task] {INPUT_CSV} 없음.")
        return

    # 카테고리별 공고 로드
    by_cat: dict[str, list[dict]] = defaultdict(list)
    with INPUT_CSV.open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            for cat in [c.strip() for c in row.get("업무카테고리", "").split(",") if c.strip()]:
                by_cat[cat].append(row)

    cache = {} if force else _load_cache()
    all_rows: list[dict] = []

    for cat in CATEGORIES:
        rows = by_cat.get(cat, [])
        if not rows:
            continue

        # 캐시 확인
        if cat in cache and not force:
            print(f"[task] {cat}: 캐시 로드 ({len(cache[cat])}개 task)")
            scored = [TaskScore(**t) for t in cache[cat]]
        else:
            # Pass A: task 추출
            print(f"[task] {cat} Pass A — task 추출 중...", end=" ", flush=True)
            tasks = await _extract_tasks(cat, rows)
            print(f"{len(tasks)}개 추출")

            # Pass B: 스코어링
            print(f"[task] {cat} Pass B — 스코어링 중...", end=" ", flush=True)
            scored = _score_tasks(cat, tasks)
            print(f"완료")

            # 캐시 저장
            cache[cat] = [s.model_dump() for s in scored]
            _save_cache(cache)

        # CSV 행 생성
        for s in scored:
            raw, level = _calc(s)
            all_rows.append({
                "카테고리": cat,
                "task명": s.task,
                "반복성(1-5)": s.반복성,
                "전략민감도(1-5)": s.전략민감도,
                "스킬희소성(1-5)": s.스킬희소성,
                "브랜드의존도(1-5)": s.브랜드의존도,
                "외주적합도(-8~8)": raw,
                "외주적합도_레벨": level,
                "발주단위설명": s.발주단위설명,
            })

    _save_csv(all_rows)

    print("\n" + "=" * 60)
    print("Task 단위 외주 적합도 요약 (★높음 △중간 ✕낮음)")
    print("=" * 60)
    _print_summary(all_rows)
    print(f"\n[output] {OUTPUT_CSV}  ({len(all_rows)}개 task)")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="캐시 무시 재실행")
    args = parser.parse_args()
    asyncio.run(run(force=args.force))
