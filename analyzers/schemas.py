# 이 파일이 존재하는 이유:
# Pass1/Pass2 LLM 출력의 구조를 pydantic으로 정의한다.
# OpenAI Structured Outputs의 response_format= 에 직접 주입되므로,
# 여기서 필드 description을 쓰는 것이 곧 LLM에게 주는 판단 기준이다.

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# ── 공유 Enum ─────────────────────────────────────────────────────────────────
# Pass1.ax_ai.kmong_service_fit 과 Pass2.ax_ai_intel.kmong_service_fit 이 동일 타입 사용.
# 두 레이어 간 drift 방지. 값 추가 시 이 한 줄만 수정.

KmongServiceType = Literal[
    "영상편집",
    "모션그래픽",
    "3D",
    "사진보정",
    "번역·현지화",
    "카피라이팅",
    "SNS콘텐츠기획",
    "웹디자인",
    "브랜딩·로고",
    "인플루언서섭외",
    "데이터분석·리포트",
    "AI활용교육",
    "챗봇구축",
    "광고소재제작",
]


# ── Pass1 모델 ────────────────────────────────────────────────────────────────

class AXAIIntel(BaseModel):
    """공고(Pass1) 또는 카테고리(Pass2)의 AI/자동화 신호 요약."""

    keywords: list[str] = Field(
        description=(
            "원문에 명시된 AI·AX·자동화 관련 단어 또는 구문. "
            "예: 'AI 기반 추천', 'LLM 활용', '자동화 경험 우대'. "
            "원문에 없으면 빈 배열."
        )
    )
    automation_intent: str | None = Field(
        description=(
            "'무엇을 자동화하거나 AI로 대체하려는가'를 1문장으로 기술. "
            "keywords가 비어 있거나 자동화 의도가 불분명하면 null. "
            "추측 금지 — 원문 근거가 있을 때만 기술."
        )
    )
    kmong_service_fit: KmongServiceType | None = Field(
        description=(
            "공고에 명시적 단서가 있을 때만 enum 선택. "
            "예: '3D 모델링' 명시 → '3D', '모션그래픽' 명시 → '모션그래픽'. "
            "'비주얼 콘텐츠', '크리에이티브' 같은 포괄어만 있으면 반드시 null. "
            "애매하면 null. Pass2의 ax_ai_intel이 카테고리 맥락에서 재해석함."
        )
    )


class Pass1Output(BaseModel):
    """채용공고 1건에서 외주 세일즈 논거 도출에 필요한 정보를 추출한 결과."""

    job_id: str = Field(
        description="원본 공고의 식별자. 없으면 회사명+직함을 합성해 고유값으로 사용."
    )
    company: str = Field(
        description="채용 기업명. 원문 그대로."
    )
    title: str = Field(
        description="공고 직함(포지션명). 원문 그대로."
    )
    jtbd: list[str] = Field(
        description=(
            "이 포지션이 실제로 수행할 과업(Jobs To Be Done). "
            "동사 중심 구문으로, 최대 4개. "
            "예: '뷰티 브랜드 SNS 콘텐츠를 기획·제작한다', '인플루언서 캠페인을 운영한다'. "
            "공고에 명시된 내용만 — 추측 금지."
        )
    )
    task_nature: Literal["project", "operation", "mixed"] = Field(
        description=(
            "과업의 성격. "
            "'project': 산출물이 명확하고 기간이 한정된 작업(캠페인 집행, 영상 제작 등). "
            "'operation': 상시 반복 업무(고객 응대, 재고 관리 등). "
            "'mixed': 둘 다 혼재."
        )
    )
    skills: list[str] = Field(
        description=(
            "요구하는 툴·기술·플랫폼 이름. 원문에 명시된 것만. "
            "예: ['Figma', 'Google Analytics', 'Meta Ads Manager']. "
            "소프트 스킬(커뮤니케이션 등)은 제외."
        )
    )
    career_level: Literal["신입", "주니어", "미들", "시니어", "불명"] = Field(
        description=(
            "요구 경력 수준. "
            "'신입': 0년 또는 인턴. "
            "'주니어': 1~3년. "
            "'미들': 3~7년. "
            "'시니어': 7년 이상 또는 리드·헤드 직함. "
            "'불명': 공고에 경력 조건이 없거나 불명확."
        )
    )
    outsource_fit: Literal["high", "medium", "low"] = Field(
        description=(
            "이 과업을 인하우스 채용 대신 프리랜서/에이전시 외주로 대체할 수 있는 가능성. "
            "'high': task_nature=project이고 산출물이 명확해 외주 범위 설정이 쉬움. "
            "'medium': 혼재하거나 운영 비중이 있어 부분 외주 가능. "
            "'low': 상시 내부 조율이 필수인 운영 업무."
        )
    )
    outsource_reason: str | None = Field(
        description=(
            "outsource_fit 판정의 핵심 근거 1문장. "
            "예: '캠페인 영상 제작 과업이 명확히 분리되어 있어 프리랜서 범위 설정 용이'. "
            "low일 경우에도 반드시 이유를 기술. 추측 금지."
        )
    )
    ax_ai: AXAIIntel = Field(
        description="공고 내 AI/자동화 관련 신호."
    )


# ── Pass2 전용 AX/AI 모델 ──────────────────────────────────────────────────────

class AXAICase(BaseModel):
    """AI/자동화 언급이 있는 개별 공고 사례. 공통점 없어도 보존."""

    company: str
    role_title: str
    what_they_want: str = Field(
        description="이 공고가 AI/자동화로 달성하려는 것. 공고 원문 근거 기반."
    )
    quote: str = Field(
        description="공고 본문에서 AI/자동화 관련 원문 인용 1문장."
    )
    kmong_angle: str | None = Field(
        description=(
            "Kmong이 이 수요에 접근할 각도 제안. "
            "확실한 서비스 매칭이 보이면 작성, 애매하면 null. "
            "억지로 만들지 말 것."
        )
    )


class Pass2AXAIIntel(AXAIIntel):
    """Pass2 전용 AX/AI 요약. AXAIIntel에 개별 사례 보존 필드 추가."""

    individual_cases: list[AXAICase] = Field(
        default_factory=list,
        description=(
            "AI/자동화 언급이 있는 모든 개별 공고를 사례 단위로 보존. "
            "공통점으로 묶이지 않는 1회성 사례도 반드시 포함. "
            "automation_targets(공통 테마 클러스터)와 별개로, "
            "독자가 개별 기업의 고유 수요를 그대로 볼 수 있게 함."
        )
    )


# ── Pass2 중첩 모델 ────────────────────────────────────────────────────────────

class JTBDCluster(BaseModel):
    """Pass1 jtbd 배열에서 의미론적으로 묶인 과업 패턴 1개."""

    pattern: str = Field(
        description=(
            "여러 공고에서 반복 등장하는 과업을 일반화한 구문. "
            "예: '브랜드 SNS 콘텐츠 기획·제작', '인플루언서 캐스팅 및 캠페인 운영'."
        )
    )
    count: int = Field(
        description="이 패턴이 확인된 공고 건수. Pass1 jtbd 배열에서 직접 집계."
    )
    outsource_argument: str = Field(
        description=(
            "HR 담당자에게 '이 과업은 인하우스 대신 외주가 낫다'고 설득할 논거 1~2문장. "
            "채용 비용·속도·전문성·유연성 중 가장 강한 축 하나를 선택해 구체적으로."
        )
    )


class PitchClaim(BaseModel):
    """HR 담당자를 설득하는 외주 전환 논거 1개."""

    claim: str = Field(
        description=(
            "핵심 주장 1문장. "
            "예: '콘텐츠 제작 과업의 67%는 프로젝트 단위로 분리 가능해 프리랜서 외주가 즉시 적용 가능합니다.'"
        )
    )
    evidence: str = Field(
        description=(
            "claim을 뒷받침하는 Pass1 데이터 근거. "
            "건수·비율·대표 기업명 등 수치 포함 필수. "
            "예: '분석한 23개 공고 중 15건(65%)이 task_nature=project로 분류됨.'"
        )
    )
    strength: Literal["강", "중", "약"] = Field(
        description=(
            "이 논거의 설득력. "
            "'강': 수치가 명확하고 반박이 어려움. "
            "'중': 방향은 맞으나 예외가 존재. "
            "'약': 추세 수준이며 추가 검증 필요."
        )
    )


class EducationTheme(BaseModel):
    """카테고리 실무자 대상 AI 교육 주제 후보 1개."""

    topic: str = Field(
        description=(
            "교육 주제 제목. 구체적일수록 좋음. "
            "예: 'Meta Ads Manager AI 최적화 활용법', '뷰티 제품 사진 AI 보정 워크플로우'."
        )
    )
    rationale: str = Field(
        description=(
            "이 주제가 이 카테고리 실무자에게 필요한 이유 1문장. "
            "공고에서 직접 도출 가능한 근거만. 추측 금지."
        )
    )


class IndividualTarget(BaseModel):
    """세일즈 접근 가치가 높은 개별 기업."""

    company: str = Field(description="기업명.")
    unique_need: str = Field(
        description=(
            "이 기업이 현재 해결하려는 고유한 과업 또는 니즈. "
            "공고 원문에서 직접 도출. 추측 금지. "
            "예: '일본 시장 론칭을 위한 뷰티 콘텐츠 현지화 및 인플루언서 캐스팅'."
        )
    )
    inferred_why: str = Field(
        description=(
            "이 기업이 외주를 고려할 가능성이 높은 이유 1문장. "
            "공고 수·task_nature·AI 신호 중 가장 강한 근거 하나 사용. "
            "예: '3개월 내 출시 일정과 명확한 산출물(영상 12편)이 명시되어 있어 상시 채용보다 프로젝트 외주가 효율적.'"
        )
    )
    suggested_approach: str = Field(
        description=(
            "이 기업에 대한 세일즈 훅 제안. 1~2문장, 한국어. "
            "반드시 unique_need 또는 inferred_why의 근거를 문장 안에 녹일 것. "
            "'제안하기', '소개하기', '보여주기' 등 행동 제안형 동사로 시작. "
            "일반론 금지, 이 기업 고유 맥락에 기반한 접근만."
        )
    )


# ── Pass2 출력 모델 ────────────────────────────────────────────────────────────

class LLMPass2Output(BaseModel):
    """LLM이 직접 생성하는 해석·추론 부분. Pass2 러너가 이 스키마로 OpenAI에 요청."""

    jtbd_clusters: list[JTBDCluster] = Field(
        description=(
            "반복 등장하는 과업 패턴. count 내림차순. 최소 3개, 최대 7개. "
            "count는 Pass2 러너가 사전 집계한 수치를 그대로 사용."
        )
    )
    pitch_to_hr: list[PitchClaim] = Field(
        description=(
            "HR 담당자 설득용 외주 전환 논거. strength='강' 우선, 2~4개. "
            "가장 count가 높은 jtbd_clusters 기반."
        )
    )
    individual_targets: list[IndividualTarget] = Field(
        description=(
            "세일즈 우선 접근 대상 기업. 최대 3개. "
            "outsource_fit=high이거나 AI 투자 신호가 뚜렷한 기업 우선."
        )
    )
    ax_ai_intel: Pass2AXAIIntel = Field(
        description=(
            "카테고리 전체에서 집계된 AI/자동화 신호 요약. "
            "keywords는 카테고리 내 공통 AI 키워드 상위 5개 이내. "
            "kmong_service_fit은 카테고리 맥락에서 가장 강하게 매칭되는 단일 서비스 타입. "
            "individual_cases는 AI/자동화 언급이 있는 공고를 사례 단위로 전부 보존."
        )
    )
    education_themes: list[EducationTheme] = Field(
        description=(
            "이 카테고리 실무자 대상 AI 교육 주제 후보. "
            "공고에서 직접 도출 가능한 것만. 없으면 빈 배열. 최대 5개."
        )
    )
    market_signal: str = Field(
        description=(
            "이 카테고리 채용 시장이 지금 원하는 것을 외주 관점에서 한 줄 요약. "
            "수치(건수, 비율) 포함 필수."
        )
    )


class Pass2Output(LLMPass2Output):
    """최종 출력. 러너가 Pass1 집계 통계를 LLMPass2Output에 병합해 생성.
    LLM이 채우는 필드 없음 — 모두 Pass1 결과에서 직접 계산.
    """

    category: str
    n_jobs: int
    high_outsource_ratio: float   # outsource_fit='high' 건수 / n_jobs
    n_with_ax_mention: int        # ax_ai.keywords가 비어있지 않은 Pass1 건수
