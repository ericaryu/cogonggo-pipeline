# 이 파일이 존재하는 이유:
# scraper.py가 수집하는 목록 API에는 description 류 필드가 없다.
# 이 스크립터는 공고별 상세 API를 호출해 mainTask / qualifications /
# preferences / benefits / positionDescription 을 추가 수집한다.
#
# 엔드포인트:
#   /_next/data/{buildId}/cg/{publicId}.json?id={publicId}
#   - buildId는 메인 페이지 __NEXT_DATA__에서 동적으로 읽음
#   - robots.txt: Allow: / 확인 완료

import asyncio
import json
import logging
import random
import re
import ssl
import time
from pathlib import Path

import aiohttp
import certifi
from tqdm import tqdm

log = logging.getLogger(__name__)

RAW_PATH = Path("data/raw/jobs_raw.json")
ENRICHED_PATH = Path("data/raw/jobs_enriched.json")
CACHE_PATH = Path("cache/detail_cache.json")

CONCURRENCY = 6
SLEEP_MIN = 0.3
SLEEP_MAX = 0.5

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.8",
    "Referer": "https://www.cogonggo.co/recruit",
}

# 상세 페이지에만 있고 목록 API에는 없는 필드
DETAIL_FIELDS = [
    "positionDescription",
    "mainTask",
    "qualifications",
    "preferences",
    "benefits",
    "hiringProcess",
    "subJob",
    "locationDetail",
    "identifier",
    "referenceLink",
]


# ── buildId 동적 취득 ───────────────────────────────────────────────────────────

async def _fetch_build_id(session: aiohttp.ClientSession, ssl_ctx=None) -> str:
    """메인 페이지 HTML에서 Next.js buildId를 추출한다."""
    url = "https://www.cogonggo.co/recruit"
    async with session.get(url, headers=HEADERS, ssl=ssl_ctx) as resp:
        html = await resp.text()

    # __NEXT_DATA__ JSON에서 buildId 추출
    m = re.search(r'"buildId"\s*:\s*"([^"]+)"', html)
    if m:
        build_id = m.group(1)
        log.info("[detail] buildId: %s", build_id)
        return build_id

    raise RuntimeError("buildId를 HTML에서 추출하지 못했습니다. 페이지 구조가 바뀌었을 수 있습니다.")


# ── 단일 공고 상세 수집 ─────────────────────────────────────────────────────────

async def _fetch_detail(
    session: aiohttp.ClientSession,
    build_id: str,
    public_id: str,
    semaphore: asyncio.Semaphore,
    ssl_ctx=None,
    retry: int = 3,
) -> dict | None:
    """publicId → 상세 dict 반환. 실패 시 None."""
    url = (
        f"https://www.cogonggo.co/_next/data/{build_id}/cg/{public_id}.json"
        f"?id={public_id}"
    )
    async with semaphore:
        for attempt in range(1, retry + 1):
            try:
                async with session.get(url, headers=HEADERS, timeout=aiohttp.ClientTimeout(total=15), ssl=ssl_ctx) as resp:
                    if resp.status == 200:
                        body = await resp.json(content_type=None)
                        data = body.get("pageProps", {}).get("data", {})
                        await asyncio.sleep(random.uniform(SLEEP_MIN, SLEEP_MAX))
                        return data
                    elif resp.status in (429, 503):
                        wait = 2 ** attempt + random.uniform(0, 1)
                        log.warning("[detail] %s %s → 재시도 %ds", resp.status, public_id, int(wait))
                        await asyncio.sleep(wait)
                    elif resp.status == 404:
                        log.debug("[detail] 404 %s — 스킵", public_id)
                        return None
                    else:
                        log.warning("[detail] %s %s → 스킵", resp.status, public_id)
                        return None
            except Exception as exc:
                wait = 2 ** attempt
                log.warning("[detail] 오류 %s %s → %ds 후 재시도", public_id, exc, wait)
                await asyncio.sleep(wait)
        return None


# ── 캐시 로드/저장 ──────────────────────────────────────────────────────────────

def _load_cache() -> dict[str, dict]:
    """이미 수집한 publicId → 상세 dict 맵 반환."""
    if not CACHE_PATH.exists():
        return {}
    try:
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_cache(cache: dict[str, dict]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2))


# ── 메인 실행 ──────────────────────────────────────────────────────────────────

async def run(limit: int | None = None, force: bool = False) -> list[dict]:
    """
    jobs_raw.json 의 publicId 목록으로 상세를 수집,
    목록 데이터와 병합해 jobs_enriched.json 저장.

    Args:
        limit: 수집 건수 제한 (테스트용)
        force: 캐시 무시하고 전체 재수집
    """
    if not RAW_PATH.exists():
        print(f"[detail] {RAW_PATH} 없음. --step scrape-list 먼저 실행하세요.")
        return []

    raw_jobs: list[dict] = json.loads(RAW_PATH.read_text(encoding="utf-8"))
    if limit:
        raw_jobs = raw_jobs[:limit]

    cache: dict[str, dict] = {} if force else _load_cache()

    to_fetch = [j for j in raw_jobs if j.get("publicId") and j["publicId"] not in cache]
    skipped = len(raw_jobs) - len(to_fetch)
    print(
        f"[detail] 전체 {len(raw_jobs)}건 / 캐시 스킵 {skipped}건 / "
        f"수집 예정 {len(to_fetch)}건"
    )

    if to_fetch:
        ssl_ctx = ssl.create_default_context(cafile=certifi.where())
        connector = aiohttp.TCPConnector(limit=CONCURRENCY + 2, ssl=ssl_ctx)
        async with aiohttp.ClientSession(connector=connector) as session:
            build_id = await _fetch_build_id(session, ssl_ctx)
            semaphore = asyncio.Semaphore(CONCURRENCY)

            with tqdm(total=len(to_fetch), desc="상세 수집", unit="건") as pbar:

                async def fetch_and_cache(job: dict) -> None:
                    pid = job["publicId"]
                    detail = await _fetch_detail(session, build_id, pid, semaphore, ssl_ctx)
                    if detail:
                        cache[pid] = detail
                    pbar.update(1)

                await asyncio.gather(*(fetch_and_cache(j) for j in to_fetch))

        _save_cache(cache)
        print(f"[detail] 캐시 저장 ({len(cache)}건) → {CACHE_PATH}")

    # ── 목록 데이터와 상세 데이터 병합 ──────────────────────────────────────────
    enriched: list[dict] = []
    for job in raw_jobs:
        pid = job.get("publicId")
        merged = dict(job)
        detail = cache.get(pid, {})
        for field in DETAIL_FIELDS:
            val = detail.get(field)
            if val is not None and val != "" and val != []:
                merged[field] = val
        enriched.append(merged)

    ENRICHED_PATH.parent.mkdir(parents=True, exist_ok=True)
    ENRICHED_PATH.write_text(json.dumps(enriched, ensure_ascii=False, indent=2))

    n_enriched = sum(1 for j in enriched if j.get("mainTask") or j.get("qualifications"))
    print(
        f"[detail] 완료: {len(enriched)}건 저장 → {ENRICHED_PATH}\n"
        f"  mainTask/qualifications 있는 공고: {n_enriched}건 "
        f"({n_enriched/len(enriched)*100:.0f}%)"
    )
    return enriched


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="수집 건수 제한 (테스트용)")
    parser.add_argument("--force", action="store_true", help="캐시 무시 재수집")
    args = parser.parse_args()

    asyncio.run(run(limit=args.limit, force=args.force))
