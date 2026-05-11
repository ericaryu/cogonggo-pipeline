"""
전체 파이프라인 결과를 HTML 리포트로 생성

Usage:
    python generate_report.py
Output:
    output/report.html
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

OUTPUT_PATH = Path("output/report.html")

CATEGORY_COLORS = {
    "브랜드_PR":  "#e07b54",
    "콘텐츠":     "#5c9ead",
    "퍼포먼스":   "#7b68ee",
    "기획_전략":  "#4caf93",
    "글로벌":     "#e8a838",
    "채널운영":   "#d45f7b",
    "CRM":        "#6b8cba",
}

SCORE_COLOR = {
    "높음": "#27ae60",
    "중간": "#f39c12",
    "낮음": "#e74c3c",
}

KMONG_ICON = {"가능": "✅", "부분가능": "⚠️", "어려움": "❌"}

TRIGGER_LABEL = {
    "채용공백":  "채용 공백 3개월+",
    "고정비":    "경력직 고정비 부담",
    "피크":      "시즌·캠페인 피크",
    "스킬부족":  "전문 스킬 확보 불가",
}

CATEGORIES = ["브랜드_PR", "콘텐츠", "퍼포먼스", "기획_전략", "글로벌", "채널운영", "CRM"]


# ── 데이터 로드 ───────────────────────────────────────────────────────────────

def load_csv(path):
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

heatmap   = load_csv("output/step1_heatmap.csv")
matrix    = load_csv("output/step2_outsourcing_matrix.csv")
report2   = load_json("output/step2_outsourcing_report.json")
tasks2b   = load_csv("output/step2b_tasks.csv")
debate    = load_csv("output/step3_debate_results.csv")
detail4   = load_csv("output/step4_task_detail.csv")

# ── 헬퍼 ─────────────────────────────────────────────────────────────────────

def cat_badge(cat, text=None):
    c = CATEGORY_COLORS.get(cat, "#999")
    label = text or cat
    return f'<span class="badge" style="background:{c}">{label}</span>'

def score_bar(score, max_val=8):
    pct = max(0, min(100, (score + max_val) / (max_val * 2) * 100))
    color = "#27ae60" if score >= 3 else ("#f39c12" if score >= 0 else "#e74c3c")
    sign = f"+{score}" if score > 0 else str(score)
    return f'''<div class="score-bar-wrap">
      <span class="score-num" style="color:{color}">{sign}</span>
      <div class="score-bar-bg">
        <div class="score-bar-fill" style="width:{pct}%;background:{color}"></div>
      </div>
    </div>'''

def dim_dots(val, max_v=5):
    filled = "●" * int(val)
    empty  = "○" * (max_v - int(val))
    return f'<span class="dots">{filled}{empty}</span>'


# ── 섹션별 HTML 생성 ──────────────────────────────────────────────────────────

def section_overview():
    total_jobs = sum(int(r["합계"]) for r in heatmap)
    return f"""
<section id="overview" class="section">
  <div class="section-inner">
    <div class="hero">
      <div class="hero-left">
        <div class="hero-tag">뷰티 마케팅 채용공고 분석 리포트</div>
        <h1 class="hero-title">인하우스 마케터,<br>무엇을 맡길 수 있을까?</h1>
        <p class="hero-sub">
          뷰티 브랜드 마케팅 채용공고 {total_jobs}건을 분석하여<br>
          외주화 가능한 task를 발굴하고 설득 논리를 구조화한 분석 결과입니다.
        </p>
        <div class="hero-stats">
          <div class="stat-box"><div class="stat-num">{total_jobs}</div><div class="stat-label">분석 공고 수</div></div>
          <div class="stat-box"><div class="stat-num">7</div><div class="stat-label">업무 카테고리</div></div>
          <div class="stat-box"><div class="stat-num">105</div><div class="stat-label">추출된 Task</div></div>
          <div class="stat-box"><div class="stat-num">23</div><div class="stat-label">외주 추천 Task</div></div>
        </div>
      </div>
      <div class="hero-right">
        <div class="pipeline-flow">
          {"".join(f'''<div class="flow-step">
            <div class="flow-num">{n}</div>
            <div class="flow-text">{t}</div>
          </div>{"<div class='flow-arrow'>↓</div>" if n < 4 else ""}''' for n, t in [
            (1, "업무 카테고리<br>태깅"),
            (2, "카테고리별<br>외주 적합도 스코어링"),
            (3, "Task 단위<br>추출 + 재평가"),
            (4, "페르소나 토론<br>→ 외주 제안 확정"),
          ])}
        </div>
      </div>
    </div>
  </div>
</section>"""


def section_step1():
    exp_cols = ["신입", "경력", "신입+경력"]
    rows_html = ""
    for r in heatmap:
        cat = r["카테고리"]
        total = int(r["합계"])
        exp_rate = round(int(r["경력"]) / total * 100) if total else 0
        c = CATEGORY_COLORS.get(cat, "#999")
        cells = "".join(
            f'<td class="num">{r[e]}<small>({round(int(r[e])/total*100)}%)</small></td>'
            for e in exp_cols
        )
        rows_html += f"""<tr>
          <td><span class="dot" style="background:{c}"></span>{cat}</td>
          {cells}
          <td class="num total">{total}</td>
          <td><div class="exp-bar"><div style="width:{exp_rate}%;background:{c}"></div></div>
              <small>경력 {exp_rate}%</small></td>
        </tr>"""

    return f"""
<section id="step1" class="section alt">
  <div class="section-inner">
    <div class="step-header">
      <span class="step-num">Step 1</span>
      <h2>업무 카테고리 태깅</h2>
      <p>249건 채용공고를 GPT-4o-mini로 7개 카테고리 분류 + 채용 맥락 키워드 플래그</p>
    </div>
    <div class="table-wrap">
      <table class="data-table">
        <thead><tr>
          <th>카테고리</th>
          <th>신입</th><th>경력</th><th>신입+경력</th>
          <th>합계</th><th>경력 비중</th>
        </tr></thead>
        <tbody>{rows_html}</tbody>
      </table>
    </div>
    <div class="insight-box">
      <strong>핵심 인사이트</strong>
      브랜드_PR(148건)·퍼포먼스(114건)·콘텐츠(121건)가 3대 주력 카테고리.
      전체 공고의 80%+ 가 경력직 선호 — 즉시전력 수요가 압도적.
      채널운영은 경력직 비중 88%로 가장 높아 외주 대체 난이도가 높은 영역.
    </div>
  </div>
</section>"""


def section_step2():
    matrix_sorted = sorted(matrix, key=lambda x: -int(x["외주적합도(-8~8)"]))
    rows_html = ""
    for r in matrix_sorted:
        cat = r["카테고리"]
        score = int(r["외주적합도(-8~8)"])
        level = r["외주적합도_레벨"]
        lc = SCORE_COLOR.get(level, "#999")
        c = CATEGORY_COLORS.get(cat, "#999")
        tasks_html = "".join(
            f'<li>{t}</li>' for t in report2.get(cat, {}).get("외주가능업무", [])
        )
        rows_html += f"""<tr>
          <td><span class="dot" style="background:{c}"></span>{cat}</td>
          <td class="num">{r['공고수']}</td>
          <td>{dim_dots(r['반복성(1-5)'])}</td>
          <td>{dim_dots(r['전략민감도(1-5)'])}</td>
          <td>{dim_dots(r['스킬희소성(1-5)'])}</td>
          <td>{dim_dots(r['브랜드의존도(1-5)'])}</td>
          <td>{score_bar(score)}</td>
          <td><span class="level-pill" style="background:{lc}">{level}</span></td>
          <td class="tasks-cell"><ul class="task-mini">{tasks_html}</ul></td>
        </tr>"""

    return f"""
<section id="step2" class="section">
  <div class="section-inner">
    <div class="step-header">
      <span class="step-num">Step 2</span>
      <h2>카테고리별 외주 적합도 스코어링</h2>
      <p>GPT-4.1이 공고 샘플 15건 기준으로 4개 차원 평가 · 외주적합도 = 반복성 + 스킬희소성 − 전략민감도 − 브랜드의존도</p>
    </div>
    <div class="dim-legend">
      <span>반복↑ 외주적합</span>
      <span>스킬희소↑ 외주적합</span>
      <span>전략민감↑ 외주부적합</span>
      <span>브랜드의존↑ 외주부적합</span>
    </div>
    <div class="table-wrap">
      <table class="data-table">
        <thead><tr>
          <th>카테고리</th><th>공고</th>
          <th>반복성</th><th>전략민감도</th><th>스킬희소성</th><th>브랜드의존도</th>
          <th>적합도</th><th>레벨</th><th>외주 가능 업무</th>
        </tr></thead>
        <tbody>{rows_html}</tbody>
      </table>
    </div>
    <div class="insight-box">
      <strong>핵심 인사이트</strong>
      카테고리 단위 분석에서는 모두 '중간' 이하로 평가됨 — 브랜드의존도가 전반적으로 높기 때문.
      기획_전략(−4)·글로벌(−3)은 전략민감도·브랜드의존도 동시 고점으로 외주화 가장 어려움.
      그러나 <strong>task 단위로 쪼개면</strong> 외주 가능 영역이 다수 존재 → Step 2.5로 진행.
    </div>
  </div>
</section>"""


def section_step2b():
    by_cat = defaultdict(list)
    for r in tasks2b:
        by_cat[r["카테고리"]].append(r)

    cats_html = ""
    for cat in CATEGORIES:
        rows = sorted(by_cat.get(cat, []), key=lambda x: -int(x["외주적합도(-8~8)"]))
        c = CATEGORY_COLORS.get(cat, "#999")
        high = sum(1 for r in rows if r["외주적합도_레벨"] == "높음")
        mid  = sum(1 for r in rows if r["외주적합도_레벨"] == "중간")
        low  = sum(1 for r in rows if r["외주적합도_레벨"] == "낮음")
        task_rows = ""
        for r in rows:
            score = int(r["외주적합도(-8~8)"])
            lv = r["외주적합도_레벨"]
            lc = SCORE_COLOR.get(lv, "#999")
            sign = f"+{score}" if score > 0 else str(score)
            task_rows += f"""<tr>
              <td>{r['task명']}</td>
              <td class="num">{r['반복성(1-5)']}</td>
              <td class="num">{r['전략민감도(1-5)']}</td>
              <td class="num">{r['스킬희소성(1-5)']}</td>
              <td class="num">{r['브랜드의존도(1-5)']}</td>
              <td><span class="score-inline" style="color:{lc}">{sign}</span></td>
              <td><span class="level-pill sm" style="background:{lc}">{lv}</span></td>
            </tr>"""

        cats_html += f"""
        <div class="cat-block">
          <div class="cat-header" style="border-left:4px solid {c}">
            <span class="cat-name">{cat}</span>
            <div class="cat-badges">
              <span class="mini-pill green">높음 {high}</span>
              <span class="mini-pill yellow">중간 {mid}</span>
              <span class="mini-pill red">낮음 {low}</span>
            </div>
          </div>
          <div class="table-wrap">
            <table class="data-table sm">
              <thead><tr>
                <th>Task명</th><th>반복</th><th>전략</th><th>스킬</th><th>브랜드</th>
                <th>적합도</th><th>레벨</th>
              </tr></thead>
              <tbody>{task_rows}</tbody>
            </table>
          </div>
        </div>"""

    return f"""
<section id="step2b" class="section alt">
  <div class="section-inner">
    <div class="step-header">
      <span class="step-num">Step 2.5</span>
      <h2>Task 단위 추출 + 외주 적합도 재평가</h2>
      <p>카테고리별 공고 샘플에서 GPT-4o-mini로 대표 task 추출 후 GPT-4.1 스코어링 · 7개 카테고리 × 15개 = 105개 task</p>
    </div>
    <div class="cats-grid">{cats_html}</div>
  </div>
</section>"""


def section_step3():
    # 통계
    total = len(debate)
    pos = sum(1 for r in debate if int(r["점수변화"].replace("±0","0").replace("+","")) > 0)
    neg = sum(1 for r in debate if int(r["점수변화"].replace("±0","0").replace("+","")) < 0)
    neu = total - pos - neg

    level_change_counts = defaultdict(int)
    for r in debate:
        lc = r["레벨_변화"]
        if "→" in lc:
            level_change_counts[lc] += 1

    top_tasks = sorted(
        [r for r in debate if int(r["최종점수"]) >= 3],
        key=lambda x: -int(x["최종점수"])
    )

    cards_html = ""
    for r in top_tasks:
        cat = r["카테고리"]
        c = CATEGORY_COLORS.get(cat, "#999")
        fs = int(r["최종점수"])
        sign = f"+{fs}" if fs > 0 else str(fs)
        delta_raw = r["점수변화"].replace("±0","0").replace("+","")
        delta = int(delta_raw)
        delta_str = f"+{delta}" if delta > 0 else (f"{delta}" if delta < 0 else "±0")
        delta_c = "#27ae60" if delta > 0 else ("#e74c3c" if delta < 0 else "#999")

        # 주요 쟁점 (앞 60자)
        issues = r.get("주요쟁점", "")[:80] + "…" if len(r.get("주요쟁점","")) > 80 else r.get("주요쟁점","")

        cards_html += f"""
        <div class="debate-card">
          <div class="dc-top">
            <span class="dot" style="background:{c}"></span>
            <span class="dc-cat">{cat}</span>
            <span class="dc-score">{sign}</span>
            <span class="dc-delta" style="color:{delta_c}">({delta_str})</span>
          </div>
          <div class="dc-task">{r['task명']}</div>
          <div class="dc-issues">{issues}</div>
        </div>"""

    lc_html = "".join(
        f'<div class="lc-item"><span class="lc-label">{k}</span><span class="lc-count">{v}건</span></div>'
        for k, v in sorted(level_change_counts.items(), key=lambda x: -x[1])
    )

    return f"""
<section id="step3" class="section">
  <div class="section-inner">
    <div class="step-header">
      <span class="step-num">Step 3</span>
      <h2>3인 페르소나 토론 시뮬레이션</h2>
      <p>팀장(A)·CMO(B)·대표(C) 3인이 각 task의 외주화에 반론을 제기하고 재반론 근거로 설득 — 점수 재조정</p>
    </div>

    <div class="persona-row">
      <div class="persona-card">
        <div class="p-role">A — 마케팅 팀장</div>
        <div class="p-stance bad">외주 최소화</div>
        <div class="p-quote">"외부 업체가 우리 브랜드를 이해하는 데만 한 달 걸려요."</div>
        <div class="p-concern">품질 &gt; 속도 &gt; 비용</div>
      </div>
      <div class="persona-card">
        <div class="p-role">B — CMO / 마케팅이사</div>
        <div class="p-stance mid">선별적 확대</div>
        <div class="p-quote">"경력직 연봉이면 에이전시 6개월 쓴다. 러닝커브가 문제지."</div>
        <div class="p-concern">비용 &gt; 품질 &gt; 속도</div>
      </div>
      <div class="persona-card">
        <div class="p-role">C — 대표 / 경영진</div>
        <div class="p-stance good">최대화</div>
        <div class="p-quote">"3개월째 채용 못 하면서 왜 일이 안 돌아가는 건데. 돈 주고 맡겨."</div>
        <div class="p-concern">속도 &gt; 비용 &gt; 품질</div>
      </div>
    </div>

    <div class="stats-row">
      <div class="stat-card">
        <div class="stat-big green">{pos}</div>
        <div class="stat-label">점수 상승</div>
      </div>
      <div class="stat-card">
        <div class="stat-big gray">{neu}</div>
        <div class="stat-label">점수 유지</div>
      </div>
      <div class="stat-card">
        <div class="stat-big red">{neg}</div>
        <div class="stat-label">점수 하락</div>
      </div>
      <div class="stat-card">
        <div class="stat-big blue">23</div>
        <div class="stat-label">최종 외주 추천</div>
      </div>
    </div>

    <div class="lc-row">
      <h3 class="sub-title">레벨 변화</h3>
      {lc_html}
    </div>

    <h3 class="sub-title" style="margin-top:2rem">최종 점수 ≥ +3 task (외주 추천)</h3>
    <div class="debate-cards">{cards_html}</div>
  </div>
</section>"""


def section_step4():
    by_kmong = defaultdict(list)
    for r in detail4:
        by_kmong[r["크몽_가능여부"]].append(r)

    # 트리거별 그룹화
    trigger_order = ["채용공백", "고정비", "피크", "스킬부족"]
    by_trigger = defaultdict(list)
    for r in detail4:
        for trig in [t.strip() for t in r["전환트리거"].split(",") if t.strip()]:
            by_trigger[trig].append(r)

    # 크몽 요약 바
    total4 = len(detail4)
    kmong_bar_html = ""
    for k, color in [("가능", "#27ae60"), ("부분가능", "#f39c12"), ("어려움", "#e74c3c")]:
        cnt = len(by_kmong[k])
        pct = round(cnt / total4 * 100)
        kmong_bar_html += f"""
        <div class="kb-item">
          <span class="kb-icon">{KMONG_ICON[k]}</span>
          <span class="kb-label">{k}</span>
          <div class="kb-bar-bg">
            <div class="kb-bar-fill" style="width:{pct}%;background:{color}"></div>
          </div>
          <span class="kb-count">{cnt}개</span>
        </div>"""

    # task 카드
    task_cards_html = ""
    for r in sorted(detail4, key=lambda x: -int(x["최종점수"])):
        cat = r["카테고리"]
        c = CATEGORY_COLORS.get(cat, "#999")
        score = int(r["최종점수"])
        sign = f"+{score}" if score > 0 else str(score)
        sc = "#27ae60" if score >= 4 else "#f39c12"

        kmong = r["크몽_가능여부"]
        ki = KMONG_ICON[kmong]
        kc = {"가능": "#27ae60", "부분가능": "#f39c12", "어려움": "#e74c3c"}[kmong]

        steps_html = "".join(
            f"<li>{s.strip()}</li>"
            for s in r["action_steps"].split(" | ") if s.strip()
        )
        blockers_html = "".join(
            f"<li>{b.strip()}</li>"
            for b in r["크몽_걸림돌"].split(" | ") if b.strip()
        )
        trigs = [t.strip() for t in r["전환트리거"].split(",") if t.strip()]
        trig_html = "".join(
            f'<span class="trig-pill">{TRIGGER_LABEL.get(t, t)}</span>' for t in trigs
        )

        task_cards_html += f"""
        <div class="proposal-card">
          <div class="pc-header" style="border-top:3px solid {c}">
            <div class="pc-top">
              {cat_badge(cat)}
              <span class="pc-score" style="color:{sc}">{sign}</span>
              <span class="pc-kmong" style="color:{kc}">{ki} 크몽 {kmong}</span>
            </div>
            <div class="pc-task">{r['task명']}</div>
            <div class="pc-headline">{r['제안_헤드라인']}</div>
            <div class="pc-trigs">{trig_html}</div>
          </div>
          <div class="pc-body">
            <div class="pc-col">
              <div class="pc-col-title">실무에서 실제로 하는 일</div>
              <ol class="action-list">{steps_html}</ol>
            </div>
            <div class="pc-col">
              <div class="pc-col-title" style="color:{kc}">{ki} 크몽 활용 분석</div>
              <div class="pc-col-sub">걸림돌</div>
              <ul class="blocker-list">{blockers_html}</ul>
              <div class="pc-col-sub" style="margin-top:.6rem">활용 방안</div>
              <p class="kmong-approach">{r['크몽_활용방안']}</p>
            </div>
          </div>
        </div>"""

    return f"""
<section id="step4" class="section alt">
  <div class="section-inner">
    <div class="step-header">
      <span class="step-num">Step 4</span>
      <h2>외주 제안 확정 + 전환 트리거 매핑</h2>
      <p>최종점수 ≥ +3인 23개 task 대상 · 실무 action 상세화 · 크몽 발주 가능성 평가</p>
    </div>

    <div class="kmong-summary">
      <h3 class="sub-title">크몽(kmong.com) 활용 가능 여부</h3>
      <p class="kmong-desc">국내 최대 프리랜서 플랫폼 크몽에서 실제 발주 가능한지 task별 평가</p>
      <div class="kb-wrap">{kmong_bar_html}</div>
    </div>

    <div class="trigger-summary">
      <h3 class="sub-title">전환 트리거별 해당 task 수</h3>
      <div class="trig-grid">
        {"".join(f'''<div class="trig-card">
          <div class="trig-num">{len(by_trigger[t])}개</div>
          <div class="trig-name">{TRIGGER_LABEL[t]}</div>
        </div>''' for t in trigger_order)}
      </div>
    </div>

    <h3 class="sub-title" style="margin-top:2.5rem">Task별 상세 제안</h3>
    <div class="proposal-cards">{task_cards_html}</div>
  </div>
</section>"""


# ── CSS ───────────────────────────────────────────────────────────────────────

CSS = """
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body {
  font-family: 'Noto Sans KR', 'Apple SD Gothic Neo', sans-serif;
  background: #f4f5f7;
  color: #1a1a2e;
  font-size: 14px;
  line-height: 1.6;
}

/* Nav */
.topnav {
  position: sticky; top: 0; z-index: 100;
  background: #1a1a2e;
  display: flex; align-items: center; gap: 0;
  padding: 0 2rem; height: 52px;
  box-shadow: 0 2px 12px rgba(0,0,0,.3);
}
.topnav-brand { color: #fff; font-weight: 700; font-size: 15px; margin-right: 2rem; white-space: nowrap; }
.topnav a {
  color: #a0a8c0; text-decoration: none; padding: .4rem 1rem; font-size: 13px;
  border-radius: 4px; white-space: nowrap;
}
.topnav a:hover { color: #fff; background: rgba(255,255,255,.08); }

/* Sections */
.section { padding: 4rem 1.5rem; }
.section.alt { background: #fff; }
.section-inner { max-width: 1100px; margin: 0 auto; }

/* Hero */
.hero { display: flex; gap: 3rem; align-items: center; flex-wrap: wrap; }
.hero-left { flex: 1; min-width: 280px; }
.hero-tag { font-size: 12px; font-weight: 600; color: #6c7ae0; letter-spacing: .1em; text-transform: uppercase; margin-bottom: .8rem; }
.hero-title { font-size: 2.2rem; font-weight: 800; line-height: 1.3; color: #1a1a2e; margin-bottom: 1rem; }
.hero-sub { color: #555; margin-bottom: 1.8rem; line-height: 1.8; }
.hero-stats { display: flex; gap: 1rem; flex-wrap: wrap; }
.stat-box { background: #1a1a2e; color: #fff; padding: .8rem 1.2rem; border-radius: 10px; text-align: center; min-width: 80px; }
.stat-num { font-size: 1.8rem; font-weight: 800; color: #7b8cde; }
.stat-label { font-size: 11px; color: #a0a8c0; margin-top: .2rem; }
.hero-right { flex: 0 0 260px; }
.pipeline-flow { background: #fff; border-radius: 14px; padding: 1.5rem; box-shadow: 0 4px 20px rgba(0,0,0,.08); }
.flow-step { display: flex; align-items: center; gap: .8rem; padding: .6rem 0; }
.flow-num { width: 28px; height: 28px; background: #6c7ae0; color: #fff; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 13px; flex-shrink: 0; }
.flow-text { font-size: 13px; color: #333; line-height: 1.4; }
.flow-arrow { text-align: center; color: #bbb; padding: .1rem 0 .1rem 1.8rem; font-size: 16px; }

/* Step header */
.step-header { margin-bottom: 2rem; }
.step-num { display: inline-block; background: #6c7ae0; color: #fff; font-size: 11px; font-weight: 700; padding: .2rem .7rem; border-radius: 4px; margin-bottom: .5rem; letter-spacing: .05em; }
.step-header h2 { font-size: 1.5rem; font-weight: 800; color: #1a1a2e; margin-bottom: .5rem; }
.step-header p { color: #666; font-size: 13px; }

/* Tables */
.table-wrap { overflow-x: auto; margin-bottom: 1.5rem; }
.data-table { width: 100%; border-collapse: collapse; font-size: 13px; }
.data-table th { background: #1a1a2e; color: #a0a8c0; padding: .7rem .9rem; text-align: left; font-weight: 600; font-size: 12px; white-space: nowrap; }
.data-table td { padding: .65rem .9rem; border-bottom: 1px solid #eee; vertical-align: middle; }
.data-table tr:hover td { background: #f8f9ff; }
.data-table.sm td, .data-table.sm th { padding: .4rem .7rem; font-size: 12px; }
.num { text-align: right; font-variant-numeric: tabular-nums; }
.total { font-weight: 700; color: #1a1a2e; }

/* Bars */
.exp-bar { height: 6px; background: #eee; border-radius: 3px; margin-bottom: 2px; width: 100px; }
.exp-bar div { height: 100%; border-radius: 3px; }
.score-bar-wrap { display: flex; align-items: center; gap: .5rem; }
.score-num { font-weight: 700; font-variant-numeric: tabular-nums; min-width: 28px; }
.score-bar-bg { flex: 1; height: 8px; background: #eee; border-radius: 4px; min-width: 60px; }
.score-bar-fill { height: 100%; border-radius: 4px; transition: width .3s; }
.score-inline { font-weight: 700; font-variant-numeric: tabular-nums; }

/* Dots */
.dots { letter-spacing: 1px; }

/* Badges / Pills */
.badge { display: inline-block; color: #fff; padding: .15rem .55rem; border-radius: 12px; font-size: 11px; font-weight: 600; }
.dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; flex-shrink: 0; }
.level-pill { display: inline-block; color: #fff; padding: .15rem .5rem; border-radius: 10px; font-size: 11px; font-weight: 600; }
.level-pill.sm { padding: .1rem .4rem; font-size: 10px; }
.mini-pill { display: inline-block; padding: .1rem .5rem; border-radius: 8px; font-size: 11px; font-weight: 600; color: #fff; }
.mini-pill.green { background: #27ae60; }
.mini-pill.yellow { background: #f39c12; }
.mini-pill.red { background: #e74c3c; }

/* Insight box */
.insight-box { background: #f0f2ff; border-left: 4px solid #6c7ae0; padding: 1rem 1.2rem; border-radius: 0 8px 8px 0; font-size: 13px; color: #333; line-height: 1.7; }
.insight-box strong { color: #6c7ae0; margin-right: .4rem; }

/* Dim legend */
.dim-legend { display: flex; gap: 1.5rem; margin-bottom: 1rem; flex-wrap: wrap; }
.dim-legend span { font-size: 11px; color: #888; padding: .15rem .5rem; background: #f0f2ff; border-radius: 4px; }

/* Tasks mini list */
.task-mini { list-style: none; padding: 0; }
.task-mini li { font-size: 12px; color: #555; padding: .1rem 0; }
.task-mini li::before { content: "· "; color: #aaa; }
.tasks-cell { max-width: 200px; }

/* Step2b cat blocks */
.cats-grid { display: flex; flex-direction: column; gap: 1.5rem; }
.cat-block { background: #f8f9ff; border-radius: 10px; overflow: hidden; }
.cat-header { display: flex; align-items: center; gap: 1rem; padding: .8rem 1rem; background: #fff; }
.cat-name { font-weight: 700; font-size: 14px; }
.cat-badges { display: flex; gap: .4rem; }

/* Personas */
.persona-row { display: flex; gap: 1.2rem; margin-bottom: 2rem; flex-wrap: wrap; }
.persona-card { flex: 1; min-width: 200px; background: #f8f9ff; border-radius: 12px; padding: 1.2rem; border-top: 3px solid #6c7ae0; }
.p-role { font-weight: 700; font-size: 13px; color: #1a1a2e; margin-bottom: .4rem; }
.p-stance { display: inline-block; padding: .15rem .6rem; border-radius: 8px; font-size: 11px; font-weight: 700; color: #fff; margin-bottom: .6rem; }
.p-stance.bad { background: #e74c3c; }
.p-stance.mid { background: #f39c12; }
.p-stance.good { background: #27ae60; }
.p-quote { font-size: 12px; color: #555; font-style: italic; line-height: 1.6; margin-bottom: .5rem; border-left: 2px solid #ddd; padding-left: .7rem; }
.p-concern { font-size: 11px; color: #888; }

/* Stats row */
.stats-row { display: flex; gap: 1rem; margin-bottom: 2rem; flex-wrap: wrap; }
.stat-card { flex: 1; min-width: 100px; background: #fff; border: 1px solid #eee; border-radius: 12px; padding: 1.2rem; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,.04); }
.stat-big { font-size: 2rem; font-weight: 800; }
.stat-big.green { color: #27ae60; }
.stat-big.gray { color: #999; }
.stat-big.red { color: #e74c3c; }
.stat-big.blue { color: #6c7ae0; }
.stat-card .stat-label { font-size: 12px; color: #888; margin-top: .3rem; }

/* Level change row */
.lc-row { margin-bottom: 1.5rem; }
.lc-item { display: inline-flex; align-items: center; gap: .5rem; background: #fff; border: 1px solid #eee; border-radius: 8px; padding: .4rem .9rem; margin: .3rem .3rem 0 0; }
.lc-label { font-size: 13px; color: #333; }
.lc-count { font-size: 12px; font-weight: 700; color: #6c7ae0; }

/* Debate cards */
.debate-cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 1rem; }
.debate-card { background: #fff; border: 1px solid #eee; border-radius: 10px; padding: 1rem; box-shadow: 0 2px 8px rgba(0,0,0,.04); }
.dc-top { display: flex; align-items: center; gap: .4rem; margin-bottom: .4rem; }
.dc-cat { font-size: 11px; color: #888; flex: 1; }
.dc-score { font-weight: 800; font-size: 1.1rem; color: #27ae60; }
.dc-delta { font-size: 12px; font-weight: 600; }
.dc-task { font-weight: 700; font-size: 13px; color: #1a1a2e; margin-bottom: .4rem; }
.dc-issues { font-size: 11px; color: #888; line-height: 1.5; }

/* Kmong summary */
.kmong-summary { background: #fff; border-radius: 12px; padding: 1.5rem; margin-bottom: 2rem; box-shadow: 0 2px 12px rgba(0,0,0,.06); }
.kmong-desc { font-size: 13px; color: #777; margin-bottom: 1rem; }
.kb-wrap { display: flex; flex-direction: column; gap: .7rem; }
.kb-item { display: flex; align-items: center; gap: .8rem; }
.kb-icon { font-size: 16px; width: 24px; text-align: center; }
.kb-label { width: 70px; font-size: 13px; font-weight: 600; }
.kb-bar-bg { flex: 1; height: 10px; background: #eee; border-radius: 5px; }
.kb-bar-fill { height: 100%; border-radius: 5px; }
.kb-count { width: 36px; text-align: right; font-size: 13px; font-weight: 700; color: #333; }

/* Trigger summary */
.trigger-summary { margin-bottom: 2rem; }
.trig-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 1rem; margin-top: 1rem; }
.trig-card { background: #fff; border-radius: 10px; padding: 1.2rem; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,.06); border-top: 3px solid #6c7ae0; }
.trig-num { font-size: 2rem; font-weight: 800; color: #6c7ae0; }
.trig-name { font-size: 12px; color: #666; margin-top: .3rem; line-height: 1.4; }
.trig-pill { display: inline-block; background: #eef0ff; color: #6c7ae0; padding: .15rem .55rem; border-radius: 8px; font-size: 11px; font-weight: 600; margin: .15rem .15rem 0 0; }

/* Proposal cards */
.proposal-cards { display: flex; flex-direction: column; gap: 1.5rem; }
.proposal-card { background: #fff; border-radius: 14px; overflow: hidden; box-shadow: 0 4px 20px rgba(0,0,0,.07); }
.pc-header { padding: 1.3rem 1.5rem 1rem; }
.pc-top { display: flex; align-items: center; gap: .7rem; margin-bottom: .6rem; flex-wrap: wrap; }
.pc-score { font-size: 1.3rem; font-weight: 800; }
.pc-kmong { font-size: 12px; font-weight: 600; margin-left: auto; }
.pc-task { font-size: 1.05rem; font-weight: 800; color: #1a1a2e; margin-bottom: .4rem; }
.pc-headline { font-size: 12px; color: #666; margin-bottom: .6rem; line-height: 1.6; }
.pc-trigs { display: flex; gap: .3rem; flex-wrap: wrap; }
.pc-body { display: grid; grid-template-columns: 1fr 1fr; gap: 0; border-top: 1px solid #f0f0f0; }
.pc-col { padding: 1.2rem 1.5rem; }
.pc-col:first-child { border-right: 1px solid #f0f0f0; }
.pc-col-title { font-weight: 700; font-size: 12px; color: #333; margin-bottom: .7rem; text-transform: uppercase; letter-spacing: .05em; }
.pc-col-sub { font-size: 11px; font-weight: 700; color: #999; margin-bottom: .35rem; text-transform: uppercase; }
.action-list { padding-left: 1.2rem; }
.action-list li { font-size: 12px; color: #444; padding: .25rem 0; line-height: 1.6; }
.blocker-list { padding-left: 1.1rem; }
.blocker-list li { font-size: 12px; color: #555; padding: .2rem 0; line-height: 1.5; }
.kmong-approach { font-size: 12px; color: #555; line-height: 1.7; }

/* Subtitles */
.sub-title { font-size: 15px; font-weight: 700; color: #1a1a2e; margin-bottom: .8rem; }

/* Footer */
.footer { text-align: center; padding: 2rem; color: #aaa; font-size: 12px; background: #1a1a2e; }
.footer strong { color: #7b8cde; }

@media (max-width: 700px) {
  .hero { flex-direction: column; }
  .hero-right { display: none; }
  .pc-body { grid-template-columns: 1fr; }
  .pc-col:first-child { border-right: none; border-bottom: 1px solid #f0f0f0; }
  .topnav a { padding: .4rem .5rem; font-size: 11px; }
}
"""


# ── 메인 조립 ─────────────────────────────────────────────────────────────────

def build_html():
    nav_links = "".join(
        f'<a href="#{href}">{label}</a>'
        for href, label in [
            ("overview", "개요"),
            ("step1", "Step 1 · 태깅"),
            ("step2", "Step 2 · 스코어링"),
            ("step2b", "Step 2.5 · Task"),
            ("step3", "Step 3 · 토론"),
            ("step4", "Step 4 · 제안"),
        ]
    )

    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>뷰티 마케팅 외주화 분석 리포트</title>
<link href="https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;600;700;800&display=swap" rel="stylesheet">
<style>{CSS}</style>
</head>
<body>
<nav class="topnav">
  <span class="topnav-brand">뷰티 마케팅 외주화 분석</span>
  {nav_links}
</nav>
{section_overview()}
{section_step1()}
{section_step2()}
{section_step2b()}
{section_step3()}
{section_step4()}
<footer class="footer">
  <strong>뷰티 마케팅 외주화 분석 리포트</strong> · 채용공고 249건 분석 · GPT-4.1 / GPT-4o-mini 기반
</footer>
</body>
</html>"""


if __name__ == "__main__":
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    html = build_html()
    OUTPUT_PATH.write_text(html, encoding="utf-8")
    size_kb = OUTPUT_PATH.stat().st_size // 1024
    print(f"[완료] {OUTPUT_PATH}  ({size_kb} KB)")
