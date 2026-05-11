"""
cogonggo.co/recruit/109 채용공고 크롤러
포지션명, 포지션 소개, 주요업무, 자격요건, 급여 및 처우, 우대사항 → CSV/JSON

중단 후 이어서 실행 가능:
  - 목록  : cache/recruit_109_list.json 에 저장 → 재시작 시 Playwright 스킵
  - 상세  : 10개 배치마다 캐시 + CSV/JSON 저장 → 중간에 꺼도 이어서 수집

Usage:
    python -m scraper.recruit_109_scraper
    python -m scraper.recruit_109_scraper --limit 10   # 테스트
    python -m scraper.recruit_109_scraper --force       # 캐시 무시 처음부터
"""

import argparse
import asyncio
import csv
import json
import logging
import random
import re
import ssl
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

import aiohttp
import certifi
from playwright.async_api import async_playwright
from tqdm import tqdm

log = logging.getLogger(__name__)

TARGET_URL = "https://www.cogonggo.co/recruit/109"
LIST_CACHE_PATH = Path("cache/recruit_109_list.json")   # 목록 캐시
DETAIL_CACHE_PATH = Path("cache/recruit_109_cache.json") # 상세 캐시
OUTPUT_CSV = Path("output/recruit_109_jobs.csv")
OUTPUT_JSON = Path("output/recruit_109_jobs.json")

BATCH_SIZE = 10
CONCURRENCY = 6
SLEEP_MIN = 0.3
SLEEP_MAX = 0.5

BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)
HEADERS = {
    "User-Agent": BROWSER_UA,
    "Accept": "application/json",
    "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.8",
    "Referer": TARGET_URL,
}

DETAIL_FIELDS = [
    "positionDescription", "mainTask", "qualifications", "preferences", "benefits",
    "companyName", "experienceLevel", "minExperience", "maxExperience",
]
JOB_KEYS = {"title", "publicid", "id", "position", "name"}
SKIP_FRAGMENTS = [
    "analytics", "gtm", "google", "facebook", "fonts",
    ".css", ".js", ".png", ".jpg", ".svg", ".woff", "hotjar", "tally",
]
OUTPUT_COLUMNS = [
    "기업명", "포지션명", "경력구분", "최소경력(년)", "최대경력(년)",
    "포지션 소개", "주요업무", "자격요건", "급여 및 처우", "우대사항", "publicId", "url",
]

EXPERIENCE_LEVEL_MAP = {
    "new": "신입",
    "experienced": "경력",
    "both": "신입+경력",
}


# ── 텍스트 정제 ──────────────────────────────────────────────────────────────

def _clean(val) -> str:
    if val is None:
        return ""
    if isinstance(val, list):
        return "\n".join(_clean(v) for v in val if v)
    text = str(val)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&[a-zA-Z]+;", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── 공고 응답 판별 ────────────────────────────────────────────────────────────

def _is_job_like(obj) -> bool:
    if not isinstance(obj, dict):
        return False
    return bool(JOB_KEYS & {str(k).lower() for k in obj})


def _extract_items(body) -> list | None:
    if isinstance(body, list) and body and _is_job_like(body[0]):
        return body
    if isinstance(body, dict):
        for key in ["data", "items", "jobs", "results", "list", "content", "posts", "recruits"]:
            val = body.get(key)
            if isinstance(val, list) and val and _is_job_like(val[0]):
                return val
    return None


# ── 캐시: 목록 ───────────────────────────────────────────────────────────────

def _load_list_cache() -> list | None:
    if not LIST_CACHE_PATH.exists():
        return None
    try:
        return json.loads(LIST_CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None


def _save_list_cache(jobs: list) -> None:
    LIST_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    LIST_CACHE_PATH.write_text(json.dumps(jobs, ensure_ascii=False, indent=2))


# ── 캐시: 상세 ───────────────────────────────────────────────────────────────

def _load_detail_cache() -> dict:
    if not DETAIL_CACHE_PATH.exists():
        return {}
    try:
        return json.loads(DETAIL_CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_detail_cache(cache: dict) -> None:
    DETAIL_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    DETAIL_CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2))


# ── Step 1: 목록 수집 (Playwright) ───────────────────────────────────────────

async def _detect_list_api(page) -> dict | None:
    captured: list[dict] = []

    async def on_response(response):
        url = response.url
        if any(s in url for s in SKIP_FRAGMENTS):
            return
        try:
            ct = response.headers.get("content-type", "")
            if "json" not in ct:
                return
            body = await response.json()
            items = _extract_items(body)
            if items:
                total = body.get("count") or body.get("total") or body.get("totalCount")
                captured.append({"url": url, "items": items, "total": total})
                log.info("[list] intercept: %s  (%d건)", url, len(items))
        except Exception:
            pass

    page.on("response", on_response)
    await page.goto(TARGET_URL, wait_until="domcontentloaded", timeout=30_000)
    try:
        await page.wait_for_response(
            lambda r: "job-posting" in r.url.lower() and r.status == 200,
            timeout=10_000,
        )
    except Exception:
        pass
    await page.wait_for_timeout(2_000)

    if not captured:
        return None
    return max(captured, key=lambda c: len(c["items"]))


async def _js_fetch(page, url: str) -> dict | None:
    result = await page.evaluate(
        """async (url) => {
            try {
                const r = await fetch(url, {
                    credentials: "include",
                    headers: {"Accept": "application/json"}
                });
                if (!r.ok) return {__error: r.status};
                return await r.json();
            } catch(e) {
                return {__error: String(e)};
            }
        }""",
        url,
    )
    if isinstance(result, dict) and "__error" in result:
        log.warning("[fetch] 오류 %s → %s", url, result["__error"])
        return None
    return result


async def _collect_all_pages(page, api_info: dict) -> list:
    all_items = list(api_info["items"])
    api_url = api_info["url"]

    parsed = urlparse(api_url)
    params = {k: v[0] for k, v in parse_qs(parsed.query).items()}

    page_param = next(
        (k for k in params if k.lower() in {"page", "p", "pageno", "pagenum"}), None
    )
    limit_param = next(
        (k for k in params if k.lower() in {"limit", "size", "pagesize", "per_page"}), None
    )
    limit = int(params.get(limit_param, 20)) if limit_param else 20

    if not page_param:
        return all_items

    page_num = 2
    stall = 0
    while stall < 2:
        params[page_param] = str(page_num)
        url = urlunparse(parsed._replace(query=urlencode(params)))
        body = await _js_fetch(page, url)
        if body is None:
            stall += 1
        else:
            items = _extract_items(body)
            if not items:
                stall += 1
            else:
                all_items.extend(items)
                log.info("[list] page %d: +%d건 (누적 %d)", page_num, len(items), len(all_items))
                stall = 0
                if len(items) < limit:
                    break
        page_num += 1
        await asyncio.sleep(0.4)

    return all_items


async def _scrape_list() -> list:
    """Playwright로 /recruit/109 공고 목록 전체 수집."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=["--disable-blink-features=AutomationControlled"],
        )
        context = await browser.new_context(
            user_agent=BROWSER_UA,
            extra_http_headers={"Accept-Language": "ko-KR,ko;q=0.9,en;q=0.8"},
        )
        await context.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
        )
        pw_page = await context.new_page()

        api_info = await _detect_list_api(pw_page)
        if not api_info:
            await browser.close()
            raise RuntimeError("공고 API를 탐지하지 못했습니다. 페이지를 확인하세요.")

        print(f"[list] API 탐지 성공: {api_info['url']}")
        jobs = await _collect_all_pages(pw_page, api_info)
        await browser.close()

    return jobs


# ── Step 2: 상세 수집 ────────────────────────────────────────────────────────

async def _fetch_build_id(session: aiohttp.ClientSession, ssl_ctx) -> str:
    async with session.get(
        "https://www.cogonggo.co/recruit", headers=HEADERS, ssl=ssl_ctx
    ) as resp:
        html = await resp.text()
    m = re.search(r'"buildId"\s*:\s*"([^"]+)"', html)
    if not m:
        raise RuntimeError("buildId 추출 실패.")
    build_id = m.group(1)
    log.info("[detail] buildId: %s", build_id)
    return build_id


async def _fetch_detail(
    session: aiohttp.ClientSession,
    build_id: str,
    public_id: str,
    semaphore: asyncio.Semaphore,
    ssl_ctx,
    retry: int = 3,
) -> dict | None:
    url = (
        f"https://www.cogonggo.co/_next/data/{build_id}/cg/{public_id}.json"
        f"?id={public_id}"
    )
    async with semaphore:
        for attempt in range(1, retry + 1):
            try:
                async with session.get(
                    url, headers=HEADERS,
                    timeout=aiohttp.ClientTimeout(total=15), ssl=ssl_ctx,
                ) as resp:
                    if resp.status == 200:
                        body = await resp.json(content_type=None)
                        data = body.get("pageProps", {}).get("data", {})
                        await asyncio.sleep(random.uniform(SLEEP_MIN, SLEEP_MAX))
                        return data
                    elif resp.status in (429, 503):
                        wait = 2 ** attempt + random.uniform(0, 1)
                        log.warning("[detail] %s %s → %ds 후 재시도", resp.status, public_id, int(wait))
                        await asyncio.sleep(wait)
                    elif resp.status == 404:
                        return None
                    else:
                        log.warning("[detail] %s %s → 스킵", resp.status, public_id)
                        return None
            except Exception as exc:
                log.warning("[detail] 오류 %s %s → 재시도", public_id, exc)
                await asyncio.sleep(2 ** attempt)
        return None


# ── 병합 + 저장 ──────────────────────────────────────────────────────────────

def _merge(jobs_raw: list, cache: dict) -> list:
    """목록 데이터 + 상세 캐시 병합 (원본 불변)."""
    result = []
    for job in jobs_raw:
        merged = dict(job)
        detail = cache.get(job.get("publicId", ""), {})
        for field in DETAIL_FIELDS:
            val = detail.get(field)
            if val is not None and val != "" and val != []:
                merged[field] = val
        result.append(merged)
    return result


def _to_rows(jobs: list[dict]) -> list[dict]:
    rows = []
    for job in jobs:
        pid = job.get("publicId", "")
        min_exp = job.get("minExperience")
        max_exp = job.get("maxExperience")
        rows.append({
            "기업명": _clean(job.get("companyName", "")),
            "포지션명": _clean(job.get("title", "")),
            "경력구분": EXPERIENCE_LEVEL_MAP.get(job.get("experienceLevel", ""), ""),
            "최소경력(년)": min_exp if isinstance(min_exp, int) and min_exp > 0 else "",
            "최대경력(년)": max_exp if isinstance(max_exp, int) and 0 < max_exp < 99 else "",
            "포지션 소개": _clean(job.get("positionDescription", "")),
            "주요업무": _clean(job.get("mainTask", "")),
            "자격요건": _clean(job.get("qualifications", "")),
            "급여 및 처우": _clean(job.get("benefits", "")),
            "우대사항": _clean(job.get("preferences", "")),
            "publicId": pid,
            "url": f"https://www.cogonggo.co/cg/{pid}" if pid else "",
        })
    return rows


def _save(rows: list[dict], *, quiet: bool = False) -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    OUTPUT_JSON.write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    if not quiet:
        print(f"[output] CSV  → {OUTPUT_CSV}")
        print(f"[output] JSON → {OUTPUT_JSON}")


# ── 메인 ────────────────────────────────────────────────────────────────────

async def run(limit: int | None = None, force: bool = False) -> list[dict]:
    # ── Step 1: 목록 수집 ──────────────────────────────────────────────────
    print("=" * 50)
    print("STEP 1: 공고 목록 수집 (/recruit/109)")
    print("=" * 50)

    jobs_raw = None if force else _load_list_cache()
    if jobs_raw:
        print(f"[list] 캐시 로드: {len(jobs_raw)}건  (재수집하려면 --force)")
    else:
        jobs_raw = await _scrape_list()
        _save_list_cache(jobs_raw)
        print(f"[list] 수집 완료: {len(jobs_raw)}건 → {LIST_CACHE_PATH}")

    if limit:
        jobs_raw = jobs_raw[:limit]

    # ── Step 2: 상세 수집 (BATCH_SIZE개마다 저장) ──────────────────────────
    print("\n" + "=" * 50)
    print("STEP 2: 공고 상세 수집 (JD 5개 필드, 10개 배치 저장)")
    print("=" * 50)

    cache: dict = {} if force else _load_detail_cache()
    to_fetch = [j for j in jobs_raw if j.get("publicId") and j["publicId"] not in cache]
    print(
        f"[detail] 전체 {len(jobs_raw)}건 / "
        f"캐시 {len(jobs_raw) - len(to_fetch)}건 스킵 / "
        f"수집 예정 {len(to_fetch)}건"
    )

    if to_fetch:
        ssl_ctx = ssl.create_default_context(cafile=certifi.where())
        connector = aiohttp.TCPConnector(limit=CONCURRENCY + 2, ssl=ssl_ctx)
        async with aiohttp.ClientSession(connector=connector) as session:
            build_id = await _fetch_build_id(session, ssl_ctx)
            semaphore = asyncio.Semaphore(CONCURRENCY)

            total_batches = (len(to_fetch) + BATCH_SIZE - 1) // BATCH_SIZE
            with tqdm(total=len(to_fetch), desc="상세 수집", unit="건") as pbar:
                for batch_idx in range(0, len(to_fetch), BATCH_SIZE):
                    batch = to_fetch[batch_idx: batch_idx + BATCH_SIZE]

                    # 배치 내 병렬 수집
                    async def fetch_one(job: dict) -> None:
                        pid = job["publicId"]
                        detail = await _fetch_detail(
                            session, build_id, pid, semaphore, ssl_ctx
                        )
                        if detail:
                            cache[pid] = detail
                        pbar.update(1)

                    await asyncio.gather(*(fetch_one(j) for j in batch))

                    # 배치 완료 → 캐시 + CSV/JSON 즉시 저장
                    _save_detail_cache(cache)
                    _save(_to_rows(_merge(jobs_raw, cache)), quiet=True)

                    done = min(batch_idx + BATCH_SIZE, len(to_fetch))
                    current_batch = batch_idx // BATCH_SIZE + 1
                    print(
                        f"\n[저장] 배치 {current_batch}/{total_batches} 완료 "
                        f"({done}/{len(to_fetch)}건) → {OUTPUT_CSV.name}"
                    )

    # ── Step 3: 최종 저장 ─────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("STEP 3: 최종 저장")
    print("=" * 50)

    rows = _to_rows(_merge(jobs_raw, cache))
    _save(rows)

    filled = sum(1 for r in rows if r["주요업무"] or r["자격요건"])
    print(f"\n완료: {len(rows)}건 (JD 있는 공고: {filled}건)")
    return rows


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="cogonggo /recruit/109 크롤러")
    parser.add_argument("--limit", type=int, default=None, help="수집 건수 제한 (테스트)")
    parser.add_argument("--force", action="store_true", help="캐시 무시, 처음부터 재수집")
    args = parser.parse_args()
    asyncio.run(run(limit=args.limit, force=args.force))
