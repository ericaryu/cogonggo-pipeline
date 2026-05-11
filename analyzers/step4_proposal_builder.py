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
    triggers: list[Literal["채용공백", "고정비", "피크", "스킬부족"]]
    kmong_coverage: Literal["가능", "부분가능", "어려움"]
    kmong_blockers: list[str]
    kmong_approach: str
    proposal_headline: str


# ── 시스템 프롬프트 ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = """당신은 뷰티 브랜드 마케팅 실무 전문가이자 프리랜서 플랫폼 활용 전략가입니다.
주어진 마케팅 task에 대해 4가지를 분석합니다.

[1] action_steps (5~10개)
실제 이 task를 수행하는 실무자가 구체적으로 하는 일들.
- 도구/플랫폼 이름 명시 (인스타그램, 구글 애즈, 노션, 스프레드시트 등)
- 클릭 단위, 전송 단위로 구체화 (예: "인스타그램 릴스 탭에서 #뷰티 해시태그 검색 후 조회수 5만+ 계정 필터링")
- 판단이 필요한 순간도 포함 (예: "팔로워 대비 댓글 비율 1% 이상인 계정만 선별")
- 추상적 표현 금지 ("리서치 진행" → 불가, "네이버 트렌드·구글 트렌드에서 최근 3개월 검색량 확인" → 가능)

[2] triggers (해당하는 것 모두 선택)
이 task를 외주로 전환하게 되는 트리거 상황:
- 채용공백: 포지션 공백 중 즉시 커버 필요
- 고정비: 경력직 채용 대신 변동비화
- 피크: 시즌·캠페인 물량 급증 대응
- 스킬부족: 내부에 해당 스킬 보유자 없음

[3] kmong 평가
크몽(kmong.com, 국내 최대 프리랜서 플랫폼)에서 이 task를 실제로 발주할 수 있는지 평가:
- kmong_coverage: 가능/부분가능/어려움
  - 가능: 크몽에 해당 카테고리 서비스가 있고 발주 방식도 명확
  - 부분가능: 일부 sub-step은 발주 가능하나 브랜드 전달, 실시간 대응 등 제약
  - 어려움: 크몽 서비스 특성상 이 task 자체를 발주하기 구조적으로 어려움
- kmong_blockers: 크몽에서 이 task를 발주할 때 부딪히는 구체적 걸림돌 2~3가지
  (예: "크몽 콘텐츠 제작 서비스는 단발성 결과물 기준 — 지속 운영형 task는 별도 계약 필요",
       "브랜드 계정 로그인 권한 공유 시 보안 리스크",
       "크몽 셀러 대부분 성과 보장 아닌 납품 보장 — ROAS 책임 불가")
- kmong_approach: 크몽을 실제로 활용한다면 어떻게 발주하는지 (검색 키워드, 서비스 카테고리, 발주 형태, 예상 단가)

[4] proposal_headline
"당신 회사가 [트리거 상황]이라면, [이 task]는 [이유]로 외주가 합리적입니다" 형태의 1줄 제안"""

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
        max_tokens=1200,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        response_format=TaskProposal,
    )
    parsed = resp.choices[0].message.parsed
    if parsed is None:
        raise ValueError(f"parse 실패: {task_row['task명']}")
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
    "action_steps",
    "전환트리거",
    "크몽_가능여부",
    "크몽_걸림돌",
    "크몽_활용방안",
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
    lines = []
    lines.append("# 뷰티 마케팅 외주화 실전 제안서")
    lines.append("")
    lines.append("> 249개 뷰티 마케팅 채용공고 분석 + 3인 페르소나 토론 기반  ")
    lines.append("> 최종 외주 적합도 +3 이상 task 23개 대상")
    lines.append("")

    # 트리거별 섹션
    trigger_order = ["채용공백", "고정비", "피크", "스킬부족"]
    by_trigger: dict[str, list[dict]] = {t: [] for t in trigger_order}
    for r in detail_rows:
        for trig in [t.strip() for t in r["전환트리거"].split(",") if t.strip()]:
            if trig in by_trigger:
                by_trigger[trig].append(r)

    trigger_intros = {
        "채용공백": (
            "마케터 채용에 3개월 이상 걸리고 있다면",
            "공고 분석 결과 국내 뷰티 마케팅 포지션의 평균 채용 소요기간은 상당히 길며, "
            "이 기간 동안 아래 task들은 프리랜서·에이전시로 즉시 커버 가능합니다."
        ),
        "고정비": (
            "경력직 마케터 연봉이 부담이라면",
            "경력 3-5년 마케터 연봉(4,000-6,000만원)을 고정비로 쓰는 대신, "
            "아래 task들을 건당 변동비로 전환하면 실질 비용을 낮출 수 있습니다."
        ),
        "피크": (
            "시즌·캠페인 물량이 몰릴 때 내부 인력이 부족하다면",
            "신제품 런칭, 시즌 프로모션 등 피크 시기에 내부 인력만으로는 커버가 어렵습니다. "
            "아래 task들은 캠페인 단위 계약으로 탄력적 확장이 가능합니다."
        ),
        "스킬부족": (
            "팀 내에 없는 전문 스킬이 필요하다면",
            "글로벌 인플루언서 협상, 퍼포먼스 최적화 등 특정 전문성은 채용보다 "
            "외부 조달이 현실적입니다."
        ),
    }

    for trig in trigger_order:
        rows = by_trigger[trig]
        if not rows:
            continue
        title, intro = trigger_intros[trig]
        lines.append(f"---")
        lines.append("")
        lines.append(f"## {TRIGGER_LABELS[trig]}")
        lines.append(f"### {title}")
        lines.append("")
        lines.append(intro)
        lines.append("")

        for r in sorted(rows, key=lambda x: -int(x["최종점수"])):
            score = int(r["최종점수"])
            kmong = r["크몽_가능여부"]
            kmong_mark = {"가능": "✅", "부분가능": "⚠️", "어려움": "❌"}.get(kmong, "")
            lines.append(f"#### [{score:+d}] {r['카테고리']} | {r['task명']}")
            lines.append("")
            lines.append(f"**제안:** {r['제안_헤드라인']}")
            lines.append("")

            # 실무 action
            lines.append("**실무에서 실제로 하는 일:**")
            for step in r["action_steps"].split(" | "):
                lines.append(f"- {step.strip()}")
            lines.append("")

            # 크몽
            lines.append(f"**크몽 활용 가능성: {kmong_mark} {kmong}**")
            if r["크몽_걸림돌"]:
                lines.append("")
                lines.append("걸림돌:")
                for blocker in r["크몽_걸림돌"].split(" | "):
                    lines.append(f"- {blocker.strip()}")
            lines.append("")
            lines.append(f"활용 방안: {r['크몽_활용방안']}")
            lines.append("")

    # 크몽 가능 여부 요약 테이블
    lines.append("---")
    lines.append("")
    lines.append("## 크몽 활용 가능 여부 요약")
    lines.append("")
    lines.append("| 가능 여부 | task 수 | 대표 task |")
    lines.append("|---|---|---|")
    for coverage, label in [("가능", "✅ 가능"), ("부분가능", "⚠️ 부분가능"), ("어려움", "❌ 어려움")]:
        matched = [r for r in detail_rows if r["크몽_가능여부"] == coverage]
        examples = ", ".join(r["task명"] for r in matched[:2])
        lines.append(f"| {label} | {len(matched)}개 | {examples} |")
    lines.append("")

    return "\n".join(lines)


# ── 메인 ─────────────────────────────────────────────────────────────────────

def run(force: bool = False) -> None:
    if not INPUT_CSV.exists():
        print(f"[step4] {INPUT_CSV} 없음. step3 먼저 실행하세요.")
        return

    with INPUT_CSV.open(encoding="utf-8-sig") as f:
        all_tasks = list(csv.DictReader(f))

    # step2b_tasks.csv에서 발주단위설명 로드
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
        cache_key = f"{task_row['카테고리']}::{task_row['task명']}"
        task_row["발주단위설명"] = unit_map.get(
            (task_row["카테고리"], task_row["task명"]), ""
        )

        if cache_key in cache and not force:
            p = TaskProposal(**cache[cache_key])
            print(f"[step4] ({i:02d}/{len(target)}) 캐시: {task_row['task명']}")
        else:
            print(f"[step4] ({i:02d}/{len(target)}) 분석 중: [{int(task_row['최종점수']):+d}] {task_row['카테고리']} | {task_row['task명']}", end=" ... ", flush=True)
            p = _call_llm(task_row)
            cache[cache_key] = p.model_dump()
            _save_cache(cache)
            print("완료")

        detail_rows.append({
            "카테고리":     task_row["카테고리"],
            "task명":       task_row["task명"],
            "최종점수":     task_row["최종점수"],
            "action_steps": " | ".join(p.action_steps),
            "전환트리거":   ", ".join(p.triggers),
            "크몽_가능여부": p.kmong_coverage,
            "크몽_걸림돌":  " | ".join(p.kmong_blockers),
            "크몽_활용방안": p.kmong_approach,
            "제안_헤드라인": p.proposal_headline,
        })

    _save_detail(detail_rows)
    _save_trigger_map(detail_rows)

    proposal_md = _build_proposal_md(detail_rows)
    OUTPUT_PROPOSAL.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PROPOSAL.write_text(proposal_md, encoding="utf-8")

    # 요약 출력
    print("\n" + "=" * 65)
    print("Step 4 완료 — 외주 제안 항목 확정 + 전환 트리거 매핑")
    print("=" * 65)

    coverage_counts = {"가능": 0, "부분가능": 0, "어려움": 0}
    for r in detail_rows:
        coverage_counts[r["크몽_가능여부"]] += 1

    print(f"\n크몽 활용 가능 여부:")
    marks = {"가능": "✅", "부분가능": "⚠️", "어려움": "❌"}
    for k, v in coverage_counts.items():
        print(f"  {marks[k]} {k}: {v}개")

    print(f"\n전환 트리거별 해당 task 수:")
    for trig, label in TRIGGER_LABELS.items():
        count = sum(1 for r in detail_rows if trig in r["전환트리거"])
        print(f"  {label}: {count}개")

    print(f"\n[output] {OUTPUT_DETAIL}")
    print(f"[output] {OUTPUT_TRIGGERS}")
    print(f"[output] {OUTPUT_PROPOSAL}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="캐시 무시 재실행")
    args = parser.parse_args()
    run(force=args.force)
