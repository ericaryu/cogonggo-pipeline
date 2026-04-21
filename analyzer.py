"""
Step 4: OpenAI API로 카테고리별 배치 분석 후 리포트 저장.
"""
import json
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

COMPRESSED_DIR = Path("data/compressed")
REPORTS_DIR = Path("data/reports")
MODEL = "gpt-4o"
MAX_JOBS_PER_CALL = 60
MAX_TOKENS = 2_048

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

SYSTEM_PROMPT = (
    "당신은 채용 시장 분석 전문가입니다.\n"
    "주어진 채용공고 JSON 데이터를 분석해 핵심 인사이트를 제공합니다.\n"
    "분석은 한국어로, 구체적인 수치와 근거를 포함해 간결하게 작성합니다."
)


def _build_prompt(category: str, jobs: list) -> str:
    sample = jobs[:MAX_JOBS_PER_CALL]
    jobs_json = json.dumps(sample, ensure_ascii=False, indent=2)
    total = len(jobs)
    shown = len(sample)

    return f"""아래는 **'{category}'** 카테고리 채용공고 {total}건 중 {shown}건입니다.

```json
{jobs_json}
```

다음 항목을 분석해주세요:

## 1. 핵심 요약
- 총 공고 수, 주요 회사 유형(스타트업/대기업/외국계 등), 전반적 트렌드

## 2. 자주 요구되는 역량/기술스택
- 상위 5~10개 키워드와 추정 빈도

## 3. 경력 요건 분포
- 신입/경력 비율, 주로 요구되는 연차

## 4. 주목할 만한 포지션
- 특이하거나 매력적인 공고 2~3개 간략 소개

## 5. 시장 시그널
- 이 분야에서 지금 시장이 원하는 것 한 줄 요약
"""


def _analyze_category(category: str, jobs: list) -> str:
    print(f"[analyzer] '{category}' 분석 중... ({len(jobs)}건)")

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_prompt(category, jobs)},
        ],
    )
    return response.choices[0].message.content


def analyze_all() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    cat_files = sorted(COMPRESSED_DIR.glob("*.json"))
    if not cat_files:
        print("[analyzer] 압축 데이터 없음. classifier.py를 먼저 실행하세요.")
        return

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    all_results: list[tuple[str, int, str]] = []

    for cat_file in cat_files:
        category = cat_file.stem
        jobs = json.loads(cat_file.read_text())
        if not jobs:
            continue

        analysis = _analyze_category(category, jobs)
        all_results.append((category, len(jobs), analysis))

        report = (
            f"# {category} 채용공고 분석\n\n"
            f"**분석 일시:** {timestamp}  \n"
            f"**공고 수:** {len(jobs)}건\n\n"
            "---\n\n"
            f"{analysis}\n"
        )
        out = REPORTS_DIR / f"{category}_report.md"
        out.write_text(report, encoding="utf-8")
        print(f"[analyzer]   → {out}")

    total = sum(n for _, n, _ in all_results)
    lines = [
        "# cogonggo.co 채용공고 전체 분석 리포트\n\n",
        f"**분석 일시:** {timestamp}  \n",
        f"**총 카테고리:** {len(all_results)}개  \n",
        f"**총 공고 수:** {total}건\n\n",
        "---\n\n",
    ]
    for cat, count, analysis in sorted(all_results, key=lambda x: -x[1]):
        lines.append(f"## {cat} ({count}건)\n\n")
        lines.append(analysis)
        lines.append("\n\n---\n\n")

    full_path = REPORTS_DIR / "full_report.md"
    full_path.write_text("".join(lines), encoding="utf-8")
    print(f"\n[analyzer] 전체 리포트 저장 → {full_path}")


if __name__ == "__main__":
    analyze_all()
