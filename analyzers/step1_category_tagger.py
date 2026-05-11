"""
1단계 + 1.5단계: 업무 카테고리 태깅 + 채용 맥락 키워드 플래그

입력: output/recruit_109_jobs.csv
출력:
  - output/step1_tagged.csv    (원본 + 업무카테고리 + 채용맥락키워드)
  - output/step1_heatmap.csv   (카테고리 × 경력구분 히트맵)
캐시: cache/step1_tag_cache.json  (10건마다 저장, 중단 후 재개 가능)

Usage:
    python -m analyzers.step1_category_tagger
    python -m analyzers.step1_category_tagger --force
"""

import argparse
import asyncio
import csv
import json
import logging
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential
from tqdm import tqdm

load_dotenv()
log = logging.getLogger(__name__)

INPUT_CSV   = Path("output/recruit_109_jobs.csv")
OUTPUT_CSV  = Path("output/step1_tagged.csv")
HEATMAP_CSV = Path("output/step1_heatmap.csv")
CACHE_PATH  = Path("cache/step1_tag_cache.json")

MODEL       = "gpt-4o-mini"
CONCURRENCY = 10
BATCH_SIZE  = 10
MAX_TOKENS  = 256

CATEGORIES = ["콘텐츠", "퍼포먼스", "브랜드_PR", "CRM", "채널운영", "글로벌", "기획_전략"]

EXPERIENCE_ORDER = ["신입", "경력", "신입+경력"]

SYSTEM_PROMPT = """당신은 뷰티 마케팅 채용공고 분류 전문가입니다.
주어진 채용공고에서 두 가지를 추출합니다.

[1] 업무 카테고리 (복수 선택 가능, 아래 목록에서만 선택)
- 콘텐츠: SNS 콘텐츠, 상세페이지, 영상 제작, 카피라이팅
- 퍼포먼스: 유료광고 운영(Meta/Google/TikTok), ROAS/CPA 관리, 광고 데이터 분석
- 브랜드_PR: 언론홍보, 인플루언서·바이럴 마케팅, 브랜드 이미지 관리, 이벤트
- CRM: 이메일·앱푸시, 멤버십, 리텐션, CS 연계
- 채널운영: 자사몰, 스마트스토어·쿠팡 등 마켓플레이스 운영·입점
- 글로벌: 해외 채널, 크로스보더, 현지화, 틱톡샵 해외
- 기획_전략: 브랜드 전략, 캠페인 기획, 신제품 런칭 기획, 시장조사

[2] 채용 맥락 키워드 (공고 텍스트에 명시된 경우만, 없으면 빈 리스트)
탐지 대상: 신규, 런칭, 확대, 증원, 리뉴얼, 글로벌진출, 강화, 재건

규칙:
- categories는 반드시 위 목록의 코드를 정확히 사용 (오타 금지)
- 업무 내용 기준으로 판단, 포지션명에 속지 말 것
- context_keywords는 공고 원문에 실제로 등장하는 단어만"""

USER_TEMPLATE = """포지션명: {title}
주요업무: {main_task}
자격요건: {qualifications}"""


# ── Pydantic 스키마 ──────────────────────────────────────────────────────────

class TagOutput(BaseModel):
    categories: list[str]
    context_keywords: list[str]


# ── LLM 호출 ─────────────────────────────────────────────────────────────────

_client: AsyncOpenAI | None = None

def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI()
    return _client


def _is_retryable(exc: BaseException) -> bool:
    from openai import APIStatusError, RateLimitError
    if isinstance(exc, RateLimitError):
        return True
    if isinstance(exc, APIStatusError) and exc.status_code >= 500:
        return True
    return False


@retry(
    retry=retry_if_exception(_is_retryable),
    wait=wait_exponential(multiplier=2, min=2, max=16),
    stop=stop_after_attempt(3),
    reraise=True,
)
async def _call_llm(row: dict) -> TagOutput:
    client = _get_client()
    user_msg = USER_TEMPLATE.format(
        title=row.get("포지션명", ""),
        main_task=row.get("주요업무", "")[:600],
        qualifications=row.get("자격요건", "")[:400],
    )
    response = await client.beta.chat.completions.parse(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        response_format=TagOutput,
    )
    parsed = response.choices[0].message.parsed
    if parsed is None:
        raise ValueError(f"LLM parse 실패: {row.get('포지션명')}")
    # 허용 카테고리 외 값 제거
    parsed.categories = [c for c in parsed.categories if c in CATEGORIES]
    return parsed


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

def _save_tagged(rows: list[dict]) -> None:
    if not rows:
        return
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _build_heatmap(rows: list[dict]) -> None:
    """카테고리 × 경력구분 히트맵 CSV 생성."""
    # {category: {exp_level: count}}
    matrix: dict[str, dict[str, int]] = {c: defaultdict(int) for c in CATEGORIES}
    total: dict[str, int] = defaultdict(int)

    for row in rows:
        cats_raw = row.get("업무카테고리", "")
        cats = [c.strip() for c in cats_raw.split(",") if c.strip()] if cats_raw else []
        exp = row.get("경력구분", "기타")
        for cat in cats:
            if cat in matrix:
                matrix[cat][exp] += 1
                total[cat] += 1

    exp_levels = EXPERIENCE_ORDER + sorted(
        {e for row in rows for e in [row.get("경력구분", "기타")] if e not in EXPERIENCE_ORDER}
    )

    HEATMAP_CSV.parent.mkdir(parents=True, exist_ok=True)
    with HEATMAP_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["카테고리"] + exp_levels + ["합계"])
        for cat in CATEGORIES:
            row_vals = [matrix[cat].get(e, 0) for e in exp_levels]
            writer.writerow([cat] + row_vals + [total[cat]])

    # 콘솔 출력
    print(f"\n{'카테고리':<12}", end="")
    for e in exp_levels:
        print(f"{e:>10}", end="")
    print(f"{'합계':>8}")
    print("─" * (12 + 10 * len(exp_levels) + 8))
    for cat in CATEGORIES:
        print(f"{cat:<12}", end="")
        for e in exp_levels:
            print(f"{matrix[cat].get(e, 0):>10}", end="")
        print(f"{total[cat]:>8}")


# ── 메인 ─────────────────────────────────────────────────────────────────────

async def run(force: bool = False) -> None:
    if not INPUT_CSV.exists():
        print(f"[tagger] {INPUT_CSV} 없음. recruit_109_scraper 먼저 실행하세요.")
        return

    # 원본 CSV 로드
    with INPUT_CSV.open(encoding="utf-8-sig") as f:
        source_rows = list(csv.DictReader(f))
    print(f"[tagger] 공고 로드: {len(source_rows)}건")

    cache: dict = {} if force else _load_cache()
    to_tag = [r for r in source_rows if r.get("publicId") and r["publicId"] not in cache]
    print(
        f"[tagger] 캐시 {len(source_rows) - len(to_tag)}건 스킵 / "
        f"태깅 예정 {len(to_tag)}건"
    )

    if to_tag:
        semaphore = asyncio.Semaphore(CONCURRENCY)
        total_batches = (len(to_tag) + BATCH_SIZE - 1) // BATCH_SIZE

        async def tag_one(row: dict) -> None:
            async with semaphore:
                try:
                    result = await _call_llm(row)
                    cache[row["publicId"]] = {
                        "categories": result.categories,
                        "context_keywords": result.context_keywords,
                    }
                except Exception as exc:
                    log.warning("[tagger] 실패 %s: %s", row.get("포지션명"), exc)
                    cache[row["publicId"]] = {"categories": [], "context_keywords": []}

        with tqdm(total=len(to_tag), desc="카테고리 태깅", unit="건") as pbar:
            for batch_idx in range(0, len(to_tag), BATCH_SIZE):
                batch = to_tag[batch_idx: batch_idx + BATCH_SIZE]

                async def tag_and_update(row: dict) -> None:
                    await tag_one(row)
                    pbar.update(1)

                await asyncio.gather(*(tag_and_update(r) for r in batch))

                _save_cache(cache)
                done = min(batch_idx + BATCH_SIZE, len(to_tag))
                current = batch_idx // BATCH_SIZE + 1
                print(f"\n[저장] 배치 {current}/{total_batches} ({done}/{len(to_tag)}건)")

    # 원본 + 태그 병합
    tagged_rows = []
    for row in source_rows:
        pid = row.get("publicId", "")
        tag = cache.get(pid, {"categories": [], "context_keywords": []})
        merged = dict(row)
        merged["업무카테고리"] = ", ".join(tag.get("categories") or [])
        merged["채용맥락키워드"] = ", ".join(tag.get("context_keywords") or [])
        tagged_rows.append(merged)

    _save_tagged(tagged_rows)
    print(f"\n[output] 태깅 CSV → {OUTPUT_CSV}")

    # 히트맵
    print("\n" + "=" * 50)
    print("업무 카테고리 × 경력구분 히트맵")
    print("=" * 50)
    _build_heatmap(tagged_rows)
    print(f"\n[output] 히트맵 CSV → {HEATMAP_CSV}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="1단계 업무 카테고리 태깅")
    parser.add_argument("--force", action="store_true", help="캐시 무시 재태깅")
    args = parser.parse_args()
    asyncio.run(run(force=args.force))
