# 이 파일이 존재하는 이유:
# LLM 모델명·공통 파라미터를 한 곳에서 관리한다.
# 모델명 하드코딩을 방지하고, 모델 교체 시 이 파일만 수정하면 된다.

PASS1_MODEL = "gpt-4o-mini"   # 공고 1건 구조화 추출 — 속도·비용 우선
PASS2_MODEL = "gpt-4.1"       # 카테고리 전략 분석 — 긴 컨텍스트·추론 품질 우선

# tenacity retry 설정
RETRY_MAX_ATTEMPTS = 3
RETRY_WAIT_SECONDS = 2        # 초기 대기 (2 → 4 → 8s exponential)

# Pass1 동시 실행 제한
PASS1_CONCURRENCY = 10

# Pass1 LLM 출력 최대 토큰
PASS1_MAX_TOKENS = 512

# Pass2 LLM 출력 최대 토큰
PASS2_MAX_TOKENS = 2048

# Pass2 컨텍스트 압축 기준 (Pass1 결과가 이 건수를 넘으면 사전 집계 후 주입)
PASS2_SUMMARY_THRESHOLD = 50
