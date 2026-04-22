# 이 파일이 존재하는 이유:
# OpenAI API 호출을 한 곳에서 관리한다.
# Pass1(async, gpt-4o-mini)과 Pass2(sync, gpt-4.1) 호출,
# retry 로직, 토큰 로깅을 여기서만 처리해 러너가 LLM 세부사항에 의존하지 않게 한다.

import logging

from openai import APIStatusError, AsyncOpenAI, OpenAI, RateLimitError
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from analyzers.config import (
    PASS1_MAX_TOKENS,
    PASS1_MODEL,
    PASS2_MAX_TOKENS,
    PASS2_MODEL,
    RETRY_MAX_ATTEMPTS,
    RETRY_WAIT_SECONDS,
)
from analyzers.prompts import (
    PASS1_SYSTEM,
    PASS1_USER_TEMPLATE,
    PASS2_SYSTEM,
    PASS2_USER_TEMPLATE,
)
from analyzers.schemas import LLMPass2Output, Pass1Output

log = logging.getLogger(__name__)

# ── 클라이언트 싱글톤 ──────────────────────────────────────────────────────────
# api_key는 환경변수 OPENAI_API_KEY에서 자동 로드 (python-dotenv는 러너에서 load_dotenv)

_async_client: AsyncOpenAI | None = None
_sync_client: OpenAI | None = None


def _get_async_client() -> AsyncOpenAI:
    global _async_client
    if _async_client is None:
        _async_client = AsyncOpenAI()
    return _async_client


def _get_sync_client() -> OpenAI:
    global _sync_client
    if _sync_client is None:
        _sync_client = OpenAI()
    return _sync_client


# ── Retry 조건 ─────────────────────────────────────────────────────────────────
# 429(RateLimitError)와 5xx(APIStatusError)만 재시도. 4xx 오류는 재시도 무의미.

def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, RateLimitError):
        return True
    if isinstance(exc, APIStatusError) and exc.status_code >= 500:
        return True
    return False


_retry_policy = dict(
    retry=retry_if_exception(_is_retryable),
    wait=wait_exponential(multiplier=RETRY_WAIT_SECONDS, min=RETRY_WAIT_SECONDS, max=8),
    stop=stop_after_attempt(RETRY_MAX_ATTEMPTS),
    reraise=True,
)


# ── 토큰 로깅 ──────────────────────────────────────────────────────────────────

def _log_usage(model: str, usage) -> None:
    if usage is None:
        return
    log.info(
        "[token] model=%s prompt=%d completion=%d total=%d",
        model,
        usage.prompt_tokens,
        usage.completion_tokens,
        usage.total_tokens,
    )


# ── Pass1: 공고 1건 구조화 추출 (async) ───────────────────────────────────────

@retry(**_retry_policy)
async def call_pass1(job_json: str) -> Pass1Output:
    """공고 JSON 문자열 1건을 받아 Pass1Output을 반환한다. AsyncOpenAI 사용."""
    client = _get_async_client()
    response = await client.beta.chat.completions.parse(
        model=PASS1_MODEL,
        max_tokens=PASS1_MAX_TOKENS,
        messages=[
            {"role": "system", "content": PASS1_SYSTEM},
            {"role": "user", "content": PASS1_USER_TEMPLATE.format(job_json=job_json)},
        ],
        response_format=Pass1Output,
    )
    _log_usage(PASS1_MODEL, response.usage)
    parsed = response.choices[0].message.parsed
    if parsed is None:
        raise ValueError("Pass1 parse 실패: parsed=None (refusal 또는 스키마 불일치)")
    return parsed


# ── Pass2: 카테고리 전략 분석 (sync) ──────────────────────────────────────────

@retry(**_retry_policy)
def call_pass2(category: str, stats_block: str, jobs_block: str) -> LLMPass2Output:
    """Pass1 집계 결과를 받아 LLMPass2Output을 반환한다. 동기 OpenAI 사용."""
    client = _get_sync_client()
    response = client.beta.chat.completions.parse(
        model=PASS2_MODEL,
        max_tokens=PASS2_MAX_TOKENS,
        messages=[
            {"role": "system", "content": PASS2_SYSTEM},
            {
                "role": "user",
                "content": PASS2_USER_TEMPLATE.format(
                    category=category,
                    stats_block=stats_block,
                    jobs_block=jobs_block,
                ),
            },
        ],
        response_format=LLMPass2Output,
    )
    _log_usage(PASS2_MODEL, response.usage)
    parsed = response.choices[0].message.parsed
    if parsed is None:
        raise ValueError(f"Pass2 parse 실패: category={category}, parsed=None")
    return parsed
