"""
Step 3: Rule-based 카테고리 분류 + 압축.
data/raw/jobs_raw.json → data/compressed/<카테고리>.json
"""
import json
import re
from collections import defaultdict
from pathlib import Path

RAW_PATH = Path("data/raw/jobs_raw.json")
COMPRESSED_DIR = Path("data/compressed")

# 카테고리별 키워드 (낮은 순서일수록 우선순위 높음)
CATEGORIES: dict[str, list[str]] = {
    "개발": [
        "개발", "engineer", "developer", "backend", "frontend", "fullstack",
        "full-stack", "devops", "sre", "ios", "android", "mobile", "qa",
        "테스트", "데이터 엔지니어", "ml engineer", "ai engineer", "인프라",
        "cloud", "security", "보안", "software", "python", "java",
        "javascript", "typescript", "react", "node", "flutter",
    ],
    "데이터/AI": [
        "데이터 분석", "data analyst", "data scientist", "머신러닝",
        "machine learning", "딥러닝", "deep learning", "ai", "llm",
        "nlp", "vision", "분석가", "데이터 사이언티스트",
    ],
    "디자인": [
        "디자인", "designer", "ux", "ui", "브랜드", "brand", "graphic",
        "그래픽", "visual", "illustration", "motion", "product designer",
    ],
    "기획/PM": [
        "기획", "pm ", "product manager", "product owner", " po ", "서비스 기획",
        "사업 기획", "전략기획", "planning", "프로덕트",
    ],
    "마케팅": [
        "마케팅", "marketing", "growth", "그로스", "퍼포먼스", "performance",
        "콘텐츠 마케", "content market", "sns", "seo", "crm", "광고",
        "ads", "브랜딩",
    ],
    "MD/SCM": [
        " md", "md ", "머천다이징", "merchandising", "바이어", "buyer",
        "scm", "supply chain", "구매", "물류", "logistics", "planner",
        "inventory", "재고", "소싱", "sourcing", "vmd", "vm specialist",
        "이커머스 md", "글로벌 md", "온라인 md", "아마존", "틱톡샵",
    ],
    "영업/비즈니스": [
        "영업", "sales", "biz dev", "business development", "bd ",
        "파트너십", "partnership", "account", "enterprise",
    ],
    "운영/CS": [
        "운영", "operation", "고객센터", "cs ", "고객 서비스", "customer success",
        "customer support", "cx ", "공급망",
    ],
    "콘텐츠/영상": [
        "콘텐츠 제작", "영상 제작", " pd ", "pd/", "촬영", "편집",
        "영상 pd", "콘텐츠 pd", "video", "creator",
    ],
    "R&A/품질/인허가": [
        "품질관리", "품질 관리", "qa ", "qc ", "인허가", "ra ", " ra/", "regulatory",
        "연구", "r&d", "research", "포뮬라", "formula", "성분", "ingredient",
        "bm ", " bm", "brand manager", "brand management",
    ],
    "글로벌/해외사업": [
        "글로벌", "global", "해외", "overseas", "일본", "중화권", "동남아", "북미",
        "아시아", "asia", "export", "수출", "cross-border",
    ],
    "재무/회계": [
        "재무", "finance", "회계", "accounting", "tax", "세무", "투자",
        " ir ", "cfo",
    ],
    "인사/HR": [
        "인사", " hr ", "human resource", "채용", "recruiting", "talent",
        "조직문화", "people ops", "culture", "복지",
    ],
    "법무/컴플라이언스": [
        "법무", "legal", "컴플라이언스", "compliance", "정책", "privacy",
    ],
}

# 압축 시 보존할 필드 (한/영 혼용 대응)
KEEP_FIELDS = [
    # 식별자
    "id", "publicId", "jobId", "job_id",
    # 기본 정보
    "title", "name", "position", "company", "companyName", "company_name",
    "department", "category", "categories", "tags",
    # 직무 내용
    "job", "jobType", "subJob", "description", "requirements", "qualifications", "preferred",
    "skills", "techStack", "tech_stack",
    # 상세 페이지 필드 (job_detail_scraper 수집)
    "positionDescription", "mainTask", "preferences", "benefits", "hiringProcess",
    # 경력/고용
    "experienceLevel", "careerLevel", "employmentType",
    "minExperience", "maxExperience",
    # 브랜드/지역
    "brands", "salesBrand", "salesCountries", "location",
    # 기타
    "salary", "deadline", "url", "link", "applyUrl",
    # 한국어 키
    "제목", "회사", "회사명", "직무", "포지션", "부서", "카테고리",
    "태그", "기술스택", "설명", "자격요건", "우대사항", "연봉",
    "지역", "마감일", "고용형태", "경력",
]


def _searchable(job: dict) -> str:
    priority_fields = [
        "title", "name", "position", "category", "department", "tags",
        "제목", "직무", "포지션", "부서", "카테고리",
    ]
    parts = []
    for key in priority_fields:
        val = job.get(key, "")
        if isinstance(val, list):
            parts.extend(str(v) for v in val)
        elif val:
            parts.append(str(val))
    return " ".join(parts).lower()


def classify(job: dict) -> str:
    text = _searchable(job)
    scores: dict[str, int] = defaultdict(int)

    for cat, keywords in CATEGORIES.items():
        for kw in keywords:
            if kw in text:
                scores[cat] += 1

    if scores:
        return max(scores, key=lambda c: scores[c])
    return "기타"


def compress(job: dict) -> dict:
    result = {}
    for key in KEEP_FIELDS:
        if key not in job:
            continue
        val = job[key]
        # 너무 긴 문자열은 잘라냄
        if isinstance(val, str) and len(val) > 800:
            val = val[:800] + "…"
        result[key] = val

    # 매칭된 필드가 너무 적으면 원본의 앞부분을 그대로 사용
    if len(result) < 2:
        result = {
            k: (str(v)[:300] if isinstance(v, str) else v)
            for k, v in list(job.items())[:12]
        }

    return result


def classify_and_compress(jobs: list) -> dict[str, list]:
    COMPRESSED_DIR.mkdir(parents=True, exist_ok=True)

    categorized: dict[str, list] = defaultdict(list)
    for job in jobs:
        cat = classify(job)
        categorized[cat].append(compress(job))

    for cat, cat_jobs in sorted(categorized.items(), key=lambda x: -len(x[1])):
        safe = re.sub(r"[^\w가-힣]", "_", cat)
        out = COMPRESSED_DIR / f"{safe}.json"
        out.write_text(json.dumps(cat_jobs, ensure_ascii=False, indent=2))
        print(f"[classifier] {cat:20s}: {len(cat_jobs):4d}건 → {out.name}")

    return dict(categorized)


if __name__ == "__main__":
    if not RAW_PATH.exists():
        print(f"[classifier] {RAW_PATH} 없음. scraper.py 먼저 실행하세요.")
    else:
        jobs = json.loads(RAW_PATH.read_text())
        classify_and_compress(jobs)
