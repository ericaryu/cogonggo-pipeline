"""
4단계: 외주 제안 항목 확정 + 전환 트리거 매핑

입력: output/step3_debate_results.csv (최종점수 ≥ +3 필터)
출력:
  - output/step4_task_detail.csv   (task별 실무 action + 크몽 평가)
  - output/step4_trigger_map.csv   (전환 트리거 × task 매핑)
  - output/step4_proposal.md       (실전 제안 문서)
캐시: cache/step4_cache.json

Usage:
    python -m analyzers.step4_proposal_builder
    python -m analyzers.step4_proposal_builder --force
"""

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

load_dotenv()
log = logging.getLogger(__name__)

INPUT_CSV       = Path("output/step3_debate_results.csv")
OUTPUT_DETAIL   = Path("output/step4_task_detail.csv")
OUTPUT_TRIGGERS = Path("output/step4_trigger_map.csv")
OUTPUT_PROPOSAL = Path("output/step4_proposal.md")
CACHE_PATH      = Path("cache/step4_cache.json")

MODEL      = "gpt-4.1"
MIN_SCORE  = 3

TRIGGER_LABELS = {
    "채용공백":   "채용 공백 3개월+",
    "고정비":     "경력직 고정비 부담",
    "피크":       "시즌·캠페인 피크 인력 부족",
    "스킬부족":   "전문 스킬 내부 확보 불가",
}

# ── Pydantic 스키마 ───────────────────────────────────────────────────────────

class TaskProposal(BaseModel):
    action_steps: list[str]
    hard_step_indices: list[int]   # action_steps 중 외주 시 가장 큰 걸림돌 step의 0-based 인덱스
    easy_step_indices: list[int]   # action_steps 중 외주 대체가 가장 쉬운 step의 0-based 인덱스
    triggers: list[Literal["채용공백", "고정비", "피크", "스킬부족"]]
    kmong_coverage: Literal["가능", "부분가능", "어려움"]
    kmong_blockers: list[str]
    kmong_approach: str
    agency_overlap: bool           # 에이전시가 이미 이 영역을 담당하고 있을 가능성이 높은지
    agency_differentiation: str    # agency_overlap=True 일 때: 기존 에이전시 대비 차별화 포인트 / False일 때: 빈 문자열
    proposal_headline: str


# ── 시스템 프롬프트 ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = """당신은 뷰티 브랜드 마케팅 실무 전문가이자 프리랜서 플랫폼 활용 전략가입니다.
주어진 마케팅 task에 대해 아래 항목을 분석합니다.

[1] action_steps (5~10개)
실제 이 task를 수행하는 실무자가 구체적으로 하는 일들.
- 도구/플랫폼 이름 명시 (인스타그램, 구글 애즈, ERP, 자사몰 관리자 페이지, 노션 등)
- 클릭·전송·입력 단위로 구체화 (예: "구글 애즈 관리자에서 캠페인 탭 → 새 캠페인 생성 클릭")
- 시스템 접근, 권한 필요 여부도 명시 (예: "ERP 로그인 후 재고 현황 조회")
- 판단 필요 순간도 포함

[2] hard_step_indices
action_steps 중 외주를 줄 때 가장 큰 걸림돌이 되는 step들의 0-based 인덱스 목록.
걸림돌 기준 (하나라도 해당하면 hard):
- ERP, 내부 재고 시스템, 자사몰 관리자 페이지 등 사내 시스템 접근 필요
- 브랜드 광고 계정(Meta Business Manager, 구글 애즈 등) 로그인 권한 필요
- 고객 개인정보, 내부 매출 데이터, CRM DB 접근 필요
- 실시간 판단·즉시 대응이 핵심인 step (예: 광고 이상 감지 즉시 중단)
- 브랜드 전략·톤앤매너 깊은 이해 없이 불가능한 창작/판단 step

[3] easy_step_indices
action_steps 중 외주 대체가 가장 쉬운 step들의 0-based 인덱스 목록.
쉬운 기준:
- 공개 데이터·무료 도구만으로 가능 (구글 트렌드, 네이버 데이터랩, 인스타그램 탐색 등)
- 결과물이 파일·문서·리스트 형태로 납품 가능
- 브랜드 내부 정보 없이도 수행 가능
- 반복·패턴화된 작업 (규격 맞추기, 포맷 통일, 데이터 취합)

[4] triggers
이 task를 외주로 전환하게 되는 실제 트리거 상황 (해당하는 것만 선택, 전부 해당하는 경우는 거의 없음):
- 채용공백: 이 포지션이 비어 있을 때 단기 커버 필요
- 고정비: 이 task 전담 인력을 뽑기보다 건당 변동비가 합리적인 경우
- 피크: 시즌·신제품 런칭 때만 물량이 급증하는 task
- 스킬부족: 팀 내 해당 스킬 보유자가 없어서 외부 조달이 불가피한 경우

[5] kmong_coverage — 엄격한 기준 적용
크몽(kmong.com)에서 이 task 전체를 실제로 발주할 수 있는지 평가.
반드시 아래 기준을 적용하고 관대하게 평가하지 말 것:

▶ 가능 (진짜 가능한 경우만)
  - 브랜드 내부 시스템·계정 접근이 전혀 불필요
  - 결과물이 문서·파일·리스트로 납품 가능
  - 크몽에 해당 서비스 카테고리가 실제로 활성화되어 있음
  - 단발 또는 정기 계약으로 발주 구조가 자연스러움

▶ 부분가능 (일부만 가능, 과반은 내부 처리 필요)
  - 핵심 step 중 일부는 내부 처리 필요하지만
  - 준비·리서치·초안 작성 등 보조적 sub-step은 외주 가능
  - "이 task의 일부를 크몽으로 덜 수 있다"는 수준

▶ 어려움 (구조적으로 크몽 발주 불가)
  - ERP, 내부 시스템, 브랜드 계정 로그인이 핵심 step에 포함
  - 실시간 모니터링·즉시 대응이 task의 핵심
  - 내부 데이터(매출, CRM, 재고) 없이는 아무것도 못 하는 task
  - 결과물이 납품 가능한 형태가 아닌 상시 운영형 task

※ 주의: "계정 공유 이슈가 있지만 일부는 가능"이면 '부분가능'이 아니라 task 성격에 따라 '어려움' 판단.
   광고 계정 직접 운용이 핵심인 task(광고 집행, CRM 캠페인 집행 등)는 계정 접근 없이 불가 → '어려움'.

- kmong_blockers: 2~3가지 구체적 걸림돌
- kmong_approach: 가능/부분가능인 경우 실제 발주 방법. 어려움인 경우 "크몽 대신 활용 가능한 대안 채널" 제시

[6] agency_overlap
뷰티 브랜드가 이 task 영역에 이미 에이전시를 쓰고 있을 가능성이 높은지 (true/false).
해당 영역: 광고 집행·매체 운영, 이커머스 채널 운영, 인플루언서 마케팅, 콘텐츠 제작 등.

[7] agency_differentiation
agency_overlap=true 인 경우:
"이미 에이전시를 쓰고 있다면, 크몽/전문 프리랜서로 대체하거나 병행할 때의 차별화 포인트"
- 기존 에이전시의 전형적 한계 (리테이너 비용, 담당자 교체, 브랜드 이해 부족)
- 크몽/프리랜서가 유리한 지점 (단가, 속도, 전문성, 브랜드 온보딩 용이성)
- 에이전시 계약 안에서 이 task만 분리 발주하는 방법
agency_overlap=false 인 경우: 빈 문자열 ""

[8] proposal_headline
"[상황]이라면 [task]는 [이유]로 외주가 합리적입니다" 형태 1줄"""

USER_TEMPLATE = """카테고리: {category}
Task명: {task}
초기 외주적합도 점수: {initial_score:+d}
토론 후 최종 점수: {final_score:+d}
남은 장벽: {barriers}
발주단위 설명: {unit_desc}"""


# ── Retry ─────────────────────────────────────────────────────────────────────

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

_client: OpenAI | None = None

def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


@retry(**_retry)
def _call_llm(task_row: dict) -> TaskProposal:
    user_msg = USER_TEMPLATE.format(
        category=task_row["카테고리"],
        task=task_row["task명"],
        initial_score=int(task_row["초기점수"]),
        final_score=int(task_row["최종점수"]),
        barriers=task_row.get("남은장벽", ""),
        unit_desc=task_row.get("발주단위설명", ""),
    )
    resp = _get_client().beta.chat.completions.parse(
        model=MODEL,
        max_tokens=1500,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        response_format=TaskProposal,
    )
    parsed = resp.choices[0].message.parsed
    if parsed is None:
        raise ValueError(f"parse 실패: {task_row['task명']}")
    # 인덱스 범위 클램프
    n = len(parsed.action_steps)
    parsed.hard_step_indices = [i for i in parsed.hard_step_indices if 0 <= i < n]
    parsed.easy_step_indices = [i for i in parsed.easy_step_indices if 0 <= i < n]
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

DETAIL_COLUMNS = [
    "카테고리", "task명", "최종점수",
    "action_steps", "hard_step_indices", "easy_step_indices",
    "전환트리거",
    "크몽_가능여부", "크몽_걸림돌", "크몽_활용방안",
    "agency_overlap", "agency_differentiation",
    "제안_헤드라인",
]

TRIGGER_COLUMNS = [
    "전환트리거", "트리거_설명",
    "카테고리", "task명", "최종점수",
    "제안_헤드라인",
]


def _save_detail(rows: list[dict]) -> None:
    OUTPUT_DETAIL.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_DETAIL.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=DETAIL_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _save_trigger_map(detail_rows: list[dict]) -> None:
    trigger_rows = []
    for r in detail_rows:
        triggers = [t.strip() for t in r["전환트리거"].split(",") if t.strip()]
        for trig in triggers:
            trigger_rows.append({
                "전환트리거": trig,
                "트리거_설명": TRIGGER_LABELS.get(trig, trig),
                "카테고리": r["카테고리"],
                "task명": r["task명"],
                "최종점수": r["최종점수"],
                "제안_헤드라인": r["제안_헤드라인"],
            })
    trigger_rows.sort(key=lambda x: (x["전환트리거"], -int(x["최종점수"])))
    with OUTPUT_TRIGGERS.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=TRIGGER_COLUMNS)
        writer.writeheader()
        writer.writerows(trigger_rows)


# ── 제안 문서 생성 ────────────────────────────────────────────────────────────

def _build_proposal_md(detail_rows: list[dict]) -> str:
    lines = ["# 뷰티 마케팅 외주화 실전 제안서", "",
             "> 249개 뷰티 마케팅 채용공고 분석 + 3인 페르소나 토론 기반  ",
             "> 최종 외주 적합도 +3 이상 task 23개 대상", ""]

    for coverage, label in [("가능", "✅ 가능"), ("부분가능", "⚠️ 부분가능"), ("어려움", "❌ 어려움")]:
        matched = [r for r in detail_rows if r["크몽_가능여부"] == coverage]
        if not matched:
            continue
        lines += [f"---", "", f"## 크몽 {label} ({len(matched)}개)", ""]
        for r in sorted(matched, key=lambda x: -int(x["최종점수"])):
            score = int(r["최종점수"])
            sign = f"+{score}" if score > 0 else str(score)
            lines += [f"### [{sign}] {r['카테고리']} | {r['task명']}", "",
                      f"**제안:** {r['제안_헤드라인']}", ""]
            steps = r["action_steps"].split(" | ")
            hard_idx = set(int(i) for i in r["hard_step_indices"].split(",") if i.strip().lstrip("-").isdigit())
            easy_idx = set(int(i) for i in r["easy_step_indices"].split(",") if i.strip().lstrip("-").isdigit())
            lines.append("**실무 action (🔴=외주 걸림돌, 🟢=외주 쉬운 step):**")
            for idx, step in enumerate(steps):
                prefix = "🔴 " if idx in hard_idx else ("🟢 " if idx in easy_idx else "   ")
                lines.append(f"- {prefix}{step.strip()}")
            lines.append("")
            lines.append(f"**크몽 걸림돌:**")
            for b in r["크몽_걸림돌"].split(" | "):
                lines.append(f"- {b.strip()}")
            lines += ["", f"**활용 방안:** {r['크몽_활용방안']}", ""]
            if r.get("agency_overlap") == "True":
                lines += ["**⚡ 에이전시 이미 사용 중인 경우:**",
                          r.get("agency_differentiation", ""), ""]
    return "\n".join(lines)


# ── 메인 ─────────────────────────────────────────────────────────────────────

def run(force: bool = False) -> None:
    if not INPUT_CSV.exists():
        print(f"[step4] {INPUT_CSV} 없음. step3 먼저 실행하세요.")
        return

    with INPUT_CSV.open(encoding="utf-8-sig") as f:
        all_tasks = list(csv.DictReader(f))

    unit_map: dict[tuple[str, str], str] = {}
    step2b_path = Path("output/step2b_tasks.csv")
    if step2b_path.exists():
        with step2b_path.open(encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                unit_map[(row["카테고리"], row["task명"])] = row.get("발주단위설명", "")

    target = [r for r in all_tasks if int(r["최종점수"]) >= MIN_SCORE]
    print(f"[step4] 대상 task: {len(target)}개 (최종점수 ≥ +{MIN_SCORE})")

    cache = {} if force else _load_cache()
    detail_rows: list[dict] = []

    for i, task_row in enumerate(target, 1):
        cache_key = f"v2::{task_row['카테고리']}::{task_row['task명']}"
        task_row["발주단위설명"] = unit_map.get((task_row["카테고리"], task_row["task명"]), "")

        if cache_key in cache and not force:
            p = TaskProposal(**cache[cache_key])
            print(f"[step4] ({i:02d}/{len(target)}) 캐시: {task_row['task명']}")
        else:
            print(f"[step4] ({i:02d}/{len(target)}) [{int(task_row['최종점수']):+d}] {task_row['카테고리']} | {task_row['task명']}", end=" ... ", flush=True)
            p = _call_llm(task_row)
            cache[cache_key] = p.model_dump()
            _save_cache(cache)
            print("완료")

        detail_rows.append({
            "카테고리":              task_row["카테고리"],
            "task명":                task_row["task명"],
            "최종점수":              task_row["최종점수"],
            "action_steps":          " | ".join(p.action_steps),
            "hard_step_indices":     ", ".join(str(i) for i in p.hard_step_indices),
            "easy_step_indices":     ", ".join(str(i) for i in p.easy_step_indices),
            "전환트리거":            ", ".join(p.triggers),
            "크몽_가능여부":         p.kmong_coverage,
            "크몽_걸림돌":           " | ".join(p.kmong_blockers),
            "크몽_활용방안":         p.kmong_approach,
            "agency_overlap":        str(p.agency_overlap),
            "agency_differentiation": p.agency_differentiation,
            "제안_헤드라인":         p.proposal_headline,
        })

    _save_detail(detail_rows)
    _save_trigger_map(detail_rows)
    OUTPUT_PROPOSAL.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PROPOSAL.write_text(_build_proposal_md(detail_rows), encoding="utf-8")

    print("\n" + "=" * 65)
    print("Step 4 완료")
    print("=" * 65)

    coverage_counts: dict[str, int] = {"가능": 0, "부분가능": 0, "어려움": 0}
    for r in detail_rows:
        coverage_counts[r["크몽_가능여부"]] = coverage_counts.get(r["크몽_가능여부"], 0) + 1
    marks = {"가능": "✅", "부분가능": "⚠️", "어려움": "❌"}
    print("\n크몽 활용 가능 여부:")
    for k, v in coverage_counts.items():
        print(f"  {marks[k]} {k}: {v}개")

    agency_count = sum(1 for r in detail_rows if r["agency_overlap"] == "True")
    print(f"\n에이전시 이미 사용 중일 가능성 있는 task: {agency_count}개")

    print(f"\n[output] {OUTPUT_DETAIL}")
    print(f"[output] {OUTPUT_TRIGGERS}")
    print(f"[output] {OUTPUT_PROPOSAL}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run(force=args.force)
