"""
3단계: Task별 페르소나 토론 + 점수 재조정

3인 페르소나(팀장/CMO/대표)가 task별 외주화에 심층·비판적 반론을 제기하고,
공고 데이터 기반 재반론을 통해 점수를 재조정. 주요 쟁점 정리.

티어별 토론 강도:
  ★ 높음 (score ≥ +3) : 3인 × 3라운드
  △ 중간 (-2 ≤ score ≤ +2): 2인(A·B) × 2라운드
  ✕ 낮음 (score ≤ -3)  : 1인(A) × 1라운드

근거 표기 원칙:
  [공고데이터] 249개 실측값 — 직접 계산해 프롬프트에 주입
  [출처명, 연도] 실존 확인 가능 보고서만 인용 (불확실하면 미인용)
  [추론] 논리적 근거, 출처 없음 명시

Usage:
    python -m analyzers.step3_persona_debate
    python -m analyzers.step3_persona_debate --force
"""

import argparse
import csv
import json
import logging
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential
from tqdm import tqdm

load_dotenv()
log = logging.getLogger(__name__)

INPUT_TASKS  = Path("output/step2b_tasks.csv")
INPUT_TAGGED = Path("output/step1_tagged.csv")
OUTPUT_CSV   = Path("output/step3_debate_results.csv")
OUTPUT_JSON  = Path("output/step3_debate_full.json")
CACHE_PATH   = Path("cache/step3_debate_cache.json")

MODEL      = "gpt-4.1"
MAX_TOKENS = 2048
BATCH_SIZE = 10

CATEGORIES = ["브랜드_PR","콘텐츠","퍼포먼스","기획_전략","글로벌","채널운영","CRM"]

OUTPUT_COLUMNS = [
    "카테고리","task명","초기점수","최종점수","점수변화","레벨_변화",
    "A_반론(팀장)","B_반론(CMO)","C_반론(대표)",
    "A_재반론_근거","B_재반론_근거","C_재반론_근거",
    "해소된반론","남은장벽",
    "외주제안_최종입장","주요쟁점",
]

# ── 카테고리 통계 (공고 데이터 근거용) ───────────────────────────────────────

CAT_STATS = {
    "브랜드_PR": {"total": 148, "경력": 106, "신입": 15, "신입+경력": 27},
    "콘텐츠":    {"total": 121, "경력": 87,  "신입": 14, "신입+경력": 20},
    "퍼포먼스":  {"total": 114, "경력": 95,  "신입": 8,  "신입+경력": 11},
    "기획_전략": {"total": 102, "경력": 88,  "신입": 4,  "신입+경력": 10},
    "글로벌":    {"total": 77,  "경력": 59,  "신입": 5,  "신입+경력": 13},
    "채널운영":  {"total": 51,  "경력": 45,  "신입": 0,  "신입+경력": 6},
    "CRM":       {"total": 22,  "경력": 18,  "신입": 3,  "신입+경력": 1},
}

# ── 페르소나 정의 ─────────────────────────────────────────────────────────────

PERSONA_A = """페르소나 A — 마케팅팀장 (과장~차장급, 경력 6-8년)
- 실무 총괄. 품질>속도>비용 우선
- 외주 경험: 부정적. 에이전시에 피드백 3회전 이상, 결국 내부에서 재작업
- 입장: 외주 확대 = 자기 팀 축소 시그널로 읽힘. 방어적
- 전형적 발언: "외부 업체가 우리 브랜드 이해하는 데만 한 달 걸려요"
- 숨은 니즈: 반복 작업에서 해방되고 싶지만, 외주 주면 존재 가치가 줄어드는 불안"""

PERSONA_B = """페르소나 B — CMO/마케팅이사 (경력 12-15년)
- 예산 배분, 채널 전략, 경영진 보고
- 입장: 비용효율>품질>속도. "같은 돈이면 고정비보다 변동비"
- 외주 경험: 혼재. 캠페인은 합리적이었으나 일상 운영은 소통 오버헤드가 비용 이상
- 전형적 발언: "경력직 한 명 연봉이면 에이전시 6개월 쓴다. 근데 바꿀 때마다 러닝커브가 문제지"
- 숨은 니즈: 적은 인원으로 많은 아웃풋 — 외주를 잘 쓰는 것이 본인 역량 증명"""

PERSONA_C = """페르소나 C — 대표/경영진 (뷰티 브랜드 오너 또는 COO)
- 최종 예산 승인, 조직 구조 결정
- 입장: 속도>비용>품질(일정 수준 이상이면 됨). "지금 이 시즌 놓치면 1년 날린다"
- 외주 경험: 간접적. 결과물만 봄. "왜 이렇게 오래 걸려?"가 주된 인식
- 전형적 발언: "3개월 채용 못 하고 있으면서 왜 일이 안 돌아가는 건데. 돈 주고 맡겨"
- 숨은 니즈: lean 조직 + 풀팀 수준 아웃풋. 단, 관리 포인트 늘어나는 것은 싫음"""

# ── 시스템 프롬프트 ───────────────────────────────────────────────────────────

def _build_system(tier: str) -> str:
    personas = {
        "높음": f"{PERSONA_A}\n\n{PERSONA_B}\n\n{PERSONA_C}",
        "중간": f"{PERSONA_A}\n\n{PERSONA_B}",
        "낮음": PERSONA_A,
    }[tier]

    rounds = {
        "높음": "3인 × 3라운드 (반론 → 재반론 → 점수 확정)",
        "중간": "2인(A·B) × 2라운드 (반론 → 재반론 → 점수 확정)",
        "낮음": "1인(A) × 1라운드 (반론 → 번복 여부 확인 → 점수 확정)",
    }[tier]

    return f"""당신은 뷰티 마케팅 외주화 의사결정 시뮬레이터입니다.
아래 페르소나들이 특정 마케팅 task의 외주화에 대해 실제처럼 토론합니다.

[등장 인물]
{personas}

[토론 구조: {rounds}]

[근거 표기 원칙 — 반드시 준수]
- [공고데이터] : 제공된 채용공고 통계를 직접 활용
- [출처명, 연도] : 실제 존재가 확실한 보고서만 인용 (예: HubSpot State of Marketing 2024, Gartner CMO Spend Survey 2023). 불확실하면 절대 인용 금지
- [추론] : 논리적 근거이나 출처 없는 경우 반드시 레이블 명시

[심층·비판적 토론 원칙]
- 각 페르소나는 자신의 이해관계와 과거 경험에서 나오는 진짜 반론을 제기
- 표면적 반론이 아니라 실무에서 실제 발생하는 구체적 문제를 지적
- 재반론은 반론의 핵심을 정면으로 반박. 회피하거나 희석하지 않음
- 재반론에도 해소되지 않는 반론은 솔직하게 "남은 장벽"으로 인정
- score_delta는 토론 결과에 따라 정직하게 조정 (-2~+2). 변화 없으면 0"""


# ── User 프롬프트 ─────────────────────────────────────────────────────────────

def _build_user(task_row: dict, stats: dict) -> str:
    cat = task_row["카테고리"]
    task = task_row["task명"]
    score = int(task_row["외주적합도(-8~8)"])
    level = task_row["외주적합도_레벨"]
    s = stats.get(cat, {})

    data_block = (
        f"[공고데이터] {cat} 카테고리 채용공고 {s.get('total', '?')}건 분석:\n"
        f"  - 경력직: {s.get('경력', 0)}건 ({s.get('경력',0)/s.get('total',1)*100:.0f}%)\n"
        f"  - 신입+경력: {s.get('신입+경력', 0)}건\n"
        f"  - 신입: {s.get('신입', 0)}건\n"
        f"  → 이 카테고리는 경력직 비중이 압도적으로 높음 = 즉시 전력 요구 포지션"
    )

    dim_block = (
        f"반복성 {task_row['반복성(1-5)']} / "
        f"전략민감도 {task_row['전략민감도(1-5)']} / "
        f"스킬희소성 {task_row['스킬희소성(1-5)']} / "
        f"브랜드의존도 {task_row['브랜드의존도(1-5)']}"
    )

    return f"""카테고리: {cat}
task: {task}
초기 외주적합도: {score:+d} ({level})
차원 점수: {dim_block}

{data_block}

발주단위 (참고): {task_row.get('발주단위설명', '')}

위 task의 외주화를 두고 토론을 진행하세요.
각 페르소나의 반론은 자신의 실무 경험과 이해관계에서 나오는 날 선 비판이어야 합니다."""


# ── Pydantic 스키마 ───────────────────────────────────────────────────────────

class DebateResult(BaseModel):
    persona_a_objection: str      # 팀장 반론
    persona_a_counter: str        # 팀장 반론 → 재반론 (근거 포함)
    persona_b_objection: str      # CMO 반론 (중간/낮음 tier면 빈 문자열)
    persona_b_counter: str
    persona_c_objection: str      # 대표 반론 (높음 tier만)
    persona_c_counter: str
    resolved_objections: list[str]
    remaining_barriers: list[str]
    score_delta: int              # -2 ~ +2
    final_verdict: str            # 외주 제안 최종 입장
    key_issues: str               # 주요 쟁점 1-3줄


# ── LLM ──────────────────────────────────────────────────────────────────────

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
    wait=wait_exponential(multiplier=2, min=2, max=32),
    stop=stop_after_attempt(3),
    reraise=True,
)
def _call_llm(system: str, user: str) -> DebateResult:
    resp = _get_client().beta.chat.completions.parse(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        response_format=DebateResult,
    )
    parsed = resp.choices[0].message.parsed
    if parsed is None:
        raise ValueError("parse 실패")
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


# ── 점수 레벨 계산 ────────────────────────────────────────────────────────────

def _level(score: int) -> str:
    return "높음" if score >= 3 else ("낮음" if score <= -3 else "중간")


# ── 저장 ─────────────────────────────────────────────────────────────────────

def _save_csv(rows: list[dict]) -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

def _save_json(full: list[dict]) -> None:
    OUTPUT_JSON.write_text(json.dumps(full, ensure_ascii=False, indent=2))


# ── 메인 ─────────────────────────────────────────────────────────────────────

def run(force: bool = False) -> None:
    if not INPUT_TASKS.exists():
        print(f"[debate] {INPUT_TASKS} 없음.")
        return

    tasks = list(csv.DictReader(INPUT_TASKS.open(encoding="utf-8-sig")))
    print(f"[debate] 전체 {len(tasks)}개 task 로드")

    cache = {} if force else _load_cache()

    csv_rows: list[dict] = []
    full_records: list[dict] = []
    to_process = [t for t in tasks if _cache_key(t) not in cache]
    cached_count = len(tasks) - len(to_process)

    print(
        f"[debate] 캐시 {cached_count}개 스킵 / "
        f"토론 예정 {len(to_process)}개"
    )

    with tqdm(total=len(to_process), desc="페르소나 토론", unit="task") as pbar:
        for batch_start in range(0, len(to_process), BATCH_SIZE):
            batch = to_process[batch_start: batch_start + BATCH_SIZE]

            for task_row in batch:
                key = _cache_key(task_row)
                init_score = int(task_row["외주적합도(-8~8)"])
                tier = _level(init_score)

                system = _build_system(tier)
                user = _build_user(task_row, CAT_STATS)

                try:
                    result = _call_llm(system, user)
                    cache[key] = result.model_dump()
                except Exception as exc:
                    log.warning("[debate] 실패 %s: %s", task_row["task명"], exc)
                    cache[key] = _empty_result()

                pbar.update(1)

            # 배치마다 캐시 + CSV 저장
            _save_cache(cache)
            _build_output(tasks, cache, csv_rows, full_records)
            _save_csv(csv_rows)
            _save_json(full_records)

            done = min(batch_start + BATCH_SIZE, len(to_process))
            current = batch_start // BATCH_SIZE + 1
            total_batches = (len(to_process) + BATCH_SIZE - 1) // BATCH_SIZE
            print(f"\n[저장] 배치 {current}/{total_batches} ({done}/{len(to_process)}개)")

    # 캐시에서 빠진 task 포함 최종 저장
    _build_output(tasks, cache, csv_rows, full_records)
    _save_csv(csv_rows)
    _save_json(full_records)

    _print_summary(csv_rows)
    print(f"\n[output] CSV  → {OUTPUT_CSV}")
    print(f"[output] JSON → {OUTPUT_JSON}")


def _cache_key(task_row: dict) -> str:
    return f"{task_row['카테고리']}::{task_row['task명']}"


def _empty_result() -> dict:
    return {k: "" for k in DebateResult.model_fields} | {
        "resolved_objections": [], "remaining_barriers": [], "score_delta": 0
    }


def _build_output(
    tasks: list[dict],
    cache: dict,
    csv_rows: list[dict],
    full_records: list[dict],
) -> None:
    csv_rows.clear()
    full_records.clear()

    for t in tasks:
        key = _cache_key(t)
        r = cache.get(key, _empty_result())
        init_score = int(t["외주적합도(-8~8)"])
        delta = int(r.get("score_delta", 0))
        final_score = max(-8, min(8, init_score + delta))

        csv_rows.append({
            "카테고리": t["카테고리"],
            "task명": t["task명"],
            "초기점수": init_score,
            "최종점수": final_score,
            "점수변화": f"{delta:+d}" if delta != 0 else "±0",
            "레벨_변화": (
                f"{_level(init_score)} → {_level(final_score)}"
                if _level(init_score) != _level(final_score)
                else _level(final_score)
            ),
            "A_반론(팀장)": r.get("persona_a_objection", ""),
            "B_반론(CMO)": r.get("persona_b_objection", ""),
            "C_반론(대표)": r.get("persona_c_objection", ""),
            "A_재반론_근거": r.get("persona_a_counter", ""),
            "B_재반론_근거": r.get("persona_b_counter", ""),
            "C_재반론_근거": r.get("persona_c_counter", ""),
            "해소된반론": " / ".join(r.get("resolved_objections") or []),
            "남은장벽": " / ".join(r.get("remaining_barriers") or []),
            "외주제안_최종입장": r.get("final_verdict", ""),
            "주요쟁점": r.get("key_issues", ""),
        })

        full_records.append({
            "카테고리": t["카테고리"],
            "task명": t["task명"],
            "초기점수": init_score,
            "최종점수": final_score,
            "토론": r,
        })


def _print_summary(csv_rows: list[dict]) -> None:
    score_changes = [r for r in csv_rows if r["점수변화"] != "±0"]
    flipped = [r for r in csv_rows if "→" in r["레벨_변화"]]

    print("\n" + "=" * 60)
    print("토론 결과 요약")
    print("=" * 60)
    print(f"전체 task: {len(csv_rows)}개")
    print(f"점수 변동: {len(score_changes)}개")
    print(f"레벨 변경: {len(flipped)}개")

    if flipped:
        print("\n[레벨 변경된 task]")
        for r in flipped:
            print(f"  {r['카테고리']} | {r['task명']}")
            print(f"    {r['레벨_변화']}  (점수 {r['초기점수']:+d} → {r['최종점수']:+d})")

    print("\n[최종 외주 적합도 높음 task (score ≥ +3)]")
    high = sorted(
        [r for r in csv_rows if int(r["최종점수"]) >= 3],
        key=lambda x: -int(x["최종점수"])
    )
    for r in high:
        print(f"  ★ [{r['최종점수']:+d}] {r['카테고리']} | {r['task명']}")
        if r["남은장벽"]:
            print(f"      장벽: {r['남은장벽'][:60]}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="캐시 무시 재실행")
    args = parser.parse_args()
    run(force=args.force)
