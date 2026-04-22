"""
Step 1 & 2: Playwright 페이지의 JS fetch()로 API를 호출해 전체 공고 수집.
(브라우저 내에서 fetch해야 쿠키/CORS 인증이 통과됨)

확인된 API: https://api.cogonggo.co/v2/home/job-postings
총 공고 수: count 필드로 제공, size=20 페이지네이션
"""
import asyncio
import json
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

from playwright.async_api import async_playwright

BASE_URL = "https://cogonggo.co/recruit"
JOBS_API = (
    "https://api.cogonggo.co/v2/home/job-postings"
    "?jobId=0&maxExperience=999&minExperience=0&sort=latest&size=20&page={page}"
)
RAW_PATH = Path("data/raw/jobs_raw.json")

JOB_KEYS = {
    "title", "name", "position", "company", "id",
    "제목", "회사", "직무", "공고", "포지션", "companyname",
}

SKIP_URL_FRAGMENTS = [
    "analytics", "gtm", "google", "facebook", "fonts",
    ".css", ".js", ".png", ".jpg", ".svg", ".woff", "hotjar", "tally",
]

PAGINATION_PARAMS = {"page", "p", "pageno", "pagenum", "offset", "skip", "start"}

BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


def _is_job_like(obj: Any) -> bool:
    if not isinstance(obj, dict):
        return False
    return bool(JOB_KEYS & {str(k).lower() for k in obj})


def _extract_items(body: Any) -> Optional[list]:
    if isinstance(body, list) and body and _is_job_like(body[0]):
        return body
    if isinstance(body, dict):
        for key in ["data", "items", "jobs", "results", "list", "content", "posts", "recruits"]:
            val = body.get(key)
            if isinstance(val, list) and val and _is_job_like(val[0]):
                return val
    return None


def _pagination_score(url: str, count: int) -> tuple:
    params = {k.lower() for k in parse_qs(urlparse(url).query)}
    return (bool(PAGINATION_PARAMS & params), count)


# ── JS fetch 헬퍼 ────────────────────────────────────────────────────────────

async def _js_fetch(page, url: str) -> Optional[dict]:
    """브라우저 컨텍스트에서 fetch() 실행. credentials=include로 쿠키 자동 전송."""
    # JS 템플릿에 URL을 직접 삽입하지 않고 argument로 전달 (특수문자 안전)
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
        print(f"[scraper] JS fetch error for {url}: {result['__error']}")
        return None
    return result


# ── 네트워크 인터셉트로 API URL 자동 탐지 ────────────────────────────────────

async def _detect_api_via_intercept(page) -> Optional[dict]:
    """응답 인터셉트로 공고 API를 찾는다. 실패해도 OK (fallback 있음)."""
    captured: list[dict] = []

    async def on_response(response):
        url = response.url
        if any(s in url for s in SKIP_URL_FRAGMENTS):
            return
        try:
            ct = response.headers.get("content-type", "")
            if "json" not in ct:
                return
            body = await response.json()
            items = _extract_items(body)
            if items:
                captured.append({"url": url, "count": len(items), "items": items})
                print(f"[scraper] intercept: {url}  ({len(items)} items)")
        except Exception:
            pass

    page.on("response", on_response)
    print(f"[scraper] Opening {BASE_URL} ...")
    await page.goto(BASE_URL, wait_until="domcontentloaded", timeout=30_000)

    try:
        await page.wait_for_response(
            lambda r: "job-postings" in r.url and r.status == 200,
            timeout=10_000,
        )
    except Exception:
        pass

    await page.wait_for_timeout(2_000)

    if captured:
        return max(captured, key=lambda c: _pagination_score(c["url"], c["count"]))
    return None


# ── 전체 페이지 수집 (알려진 API + JS fetch) ──────────────────────────────────

async def _collect_all(page, api_url: str, first_items: list) -> list:
    """page 파라미터를 올리며 전체 공고를 JS fetch로 수집."""
    all_jobs = list(first_items)

    parsed = urlparse(api_url)
    params = {k: v[0] for k, v in parse_qs(parsed.query).items()}

    page_param = next(
        (k for k in params if k.lower() in {"page", "p", "pageno", "pagenum", "currentpage"}),
        None,
    )
    offset_param = next(
        (k for k in params if k.lower() in {"offset", "skip", "start"}),
        None,
    )
    limit_param = next(
        (k for k in params if k.lower() in {"limit", "size", "pagesize", "per_page", "count"}),
        None,
    )
    limit = int(params.get(limit_param, 20)) if limit_param else 20

    if not page_param and not offset_param:
        print("[scraper] 페이지네이션 파라미터 없음 → 첫 응답만 사용")
        return all_jobs

    page_num = 2
    offset = limit
    stall = 0

    while stall < 2:
        if page_param:
            params[page_param] = str(page_num)
        else:
            params[offset_param] = str(offset)

        url = urlunparse(parsed._replace(query=urlencode(params)))
        body = await _js_fetch(page, url)

        if body is None:
            stall += 1
        else:
            items = _extract_items(body)
            if not items:
                stall += 1
            else:
                all_jobs.extend(items)
                print(f"[scraper] page {page_num}: +{len(items)}건 (누적 {len(all_jobs)})")
                stall = 0
                if len(items) < limit:
                    break

        page_num += 1
        offset += limit
        await asyncio.sleep(0.4)

    return all_jobs


# ── 무한스크롤 fallback ───────────────────────────────────────────────────────

async def _scroll_scrape(page) -> list:
    jobs: list[dict] = []
    prev_count = 0
    stall = 0
    await page.wait_for_timeout(2_000)

    while stall < 3:
        items = await page.evaluate("""() => {
            const sels = ['[class*="recruit"]','[class*="job"]','[class*="card"]',
                          '[class*="item"]','article','li[data-id]'];
            for (const sel of sels) {
                const els = Array.from(document.querySelectorAll(sel));
                if (els.length > 5) {
                    return els.map(el => ({
                        text: el.innerText?.trim().slice(0, 600) || '',
                        href: el.querySelector('a')?.href || '',
                        dataset: Object.assign({}, el.dataset),
                    })).filter(x => x.text.length > 10);
                }
            }
            return [];
        }""")
        if items:
            jobs = items

        if len(jobs) == prev_count:
            stall += 1
        else:
            stall = 0
            prev_count = len(jobs)

        print(f"[scraper] 스크롤: {len(jobs)}건 (stall={stall})")
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        await page.wait_for_timeout(1_500)
        for btn_sel in ["button[class*='more']", "[class*='load-more']", "[class*='더보기']"]:
            for btn in await page.query_selector_all(btn_sel):
                try:
                    await btn.click()
                    await page.wait_for_timeout(1_000)
                except Exception:
                    pass

    return jobs


# ── 메인 ────────────────────────────────────────────────────────────────────

async def scrape() -> list:
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)

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
        page = await context.new_page()

        # 1. 인터셉트로 API URL 탐지 (페이지 로드 중 자연스럽게 발생하는 API 캡처)
        api_info = await _detect_api_via_intercept(page)

        # 2. 인터셉트 실패 시 알려진 엔드포인트를 JS fetch로 직접 호출
        if not api_info:
            print("[scraper] 인터셉트 실패 → JS fetch 직접 호출")
            known_url = JOBS_API.format(page=1)
            body = await _js_fetch(page, known_url)
            if body:
                items = _extract_items(body)
                total = body.get("count", "?")
                if items:
                    print(f"[scraper] JS fetch 성공: {len(items)}건 (전체 {total}건)")
                    api_info = {"url": known_url, "count": len(items), "items": items}

        if api_info:
            jobs = await _collect_all(page, api_info["url"], api_info["items"])
        else:
            print("[scraper] API 완전 실패 → 스크롤 수집")
            jobs = await _scroll_scrape(page)

        await browser.close()

    RAW_PATH.write_text(json.dumps(jobs, ensure_ascii=False, indent=2))
    print(f"\n[scraper] 완료: {len(jobs)}건 저장 → {RAW_PATH}")
    return jobs


if __name__ == "__main__":
    asyncio.run(scrape())
