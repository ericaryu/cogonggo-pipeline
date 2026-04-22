# 이 파일이 존재하는 이유:
# Pass1 LLM 호출을 카테고리별로 병렬 실행한다.
# asyncio.Semaphore로 동시 호출 수를 제한하고,
# 건별 jsonl append로 중단 시 이어 재실행이 가능하다.

import asyncio
import json
import logging
import re
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from analyzers.config import PASS1_CONCURRENCY
from analyzers.llm_client import call_pass1
from analyzers.schemas import Pass1Output

load_dotenv()
log = logging.getLogger(__name__)

COMPRESSED_DIR = Path("data/compressed")
CACHE_DIR = Path("cache")


# ── 유틸 ──────────────────────────────────────────────────────────────────────

def _safe_name(category: str) -> str:
    """classifier.py 와 동일한 파일명 정규화. 이미 정규화된 값엔 idempotent."""
    return re.sub(r"[^\w가-힣]", "_", category)


def _get_job_id(job: dict) -> str:
    """공고 dict에서 고유 ID를 추출. 없으면 회사명+직함으로 합성."""
    for key in ("id", "publicId", "jobId", "job_id"):
        if val := job.get(key):
            return str(val)
    company = (
        job.get("company")
        or job.get("companyName")
        or job.get("company_name")
        or job.get("회사")
        or job.get("회사명")
        or "unknown"
    )
    title = (
        job.get("title")
        or job.get("name")
        or job.get("position")
        or job.get("제목")
        or job.get("직무")
        or "unknown"
    )
    return f"{company}__{title}"


def _load_cached_ids(cache_path: Path) -> set[str]:
    """기존 캐시 jsonl에서 이미 처리된 job_id 집합을 반환."""
    if not cache_path.exists():
        return set()
    ids: set[str] = set()
    for line in cache_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if job_id := obj.get("job_id"):
                ids.add(str(job_id))
        except json.JSONDecodeError:
            pass
    return ids


# ── 단일 공고 처리 ─────────────────────────────────────────────────────────────

async def _process_job(
    job: dict,
    semaphore: asyncio.Semaphore,
    cache_path: Path,
    failed_path: Path,
    write_lock: asyncio.Lock,
    pbar: tqdm,
) -> None:
    job_id = _get_job_id(job)
    async with semaphore:
        try:
            result: Pass1Output = await call_pass1(
                json.dumps(job, ensure_ascii=False)
            )
            line = result.model_dump_json() + "\n"
            async with write_lock:
                with cache_path.open("a", encoding="utf-8") as f:
                    f.write(line)

        except Exception as exc:
            log.warning("[pass1] 실패 job_id=%s  error=%s", job_id, exc)
            failed_record = json.dumps(
                {"job_id": job_id, "error": str(exc), "raw": job},
                ensure_ascii=False,
            )
            async with write_lock:
                with failed_path.open("a", encoding="utf-8") as f:
                    f.write(failed_record + "\n")

        finally:
            pbar.update(1)


# ── 카테고리 단위 실행 ─────────────────────────────────────────────────────────

async def run_category(category: str, force: bool = False) -> None:
    safe = _safe_name(category)
    compressed_path = COMPRESSED_DIR / f"{safe}.json"

    if not compressed_path.exists():
        log.error("[pass1] 압축 파일 없음: %s", compressed_path)
        return

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"pass1_{safe}.jsonl"
    failed_path = CACHE_DIR / f"pass1_failed_{safe}.jsonl"

    jobs: list[dict] = json.loads(compressed_path.read_text(encoding="utf-8"))

    if force and cache_path.exists():
        cache_path.unlink()
        log.info("[pass1] --force: 캐시 초기화 (%s)", safe)

    cached_ids = _load_cached_ids(cache_path)
    remaining = [j for j in jobs if _get_job_id(j) not in cached_ids]
    skipped = len(jobs) - len(remaining)

    print(
        f"[pass1] {category}: "
        f"전체 {len(jobs)}건 / 캐시 스킵 {skipped}건 / 처리 예정 {len(remaining)}건"
    )

    if not remaining:
        return

    semaphore = asyncio.Semaphore(PASS1_CONCURRENCY)
    write_lock = asyncio.Lock()

    with tqdm(total=len(remaining), desc=category, unit="건", leave=True) as pbar:
        tasks = [
            _process_job(job, semaphore, cache_path, failed_path, write_lock, pbar)
            for job in remaining
        ]
        await asyncio.gather(*tasks)

    # 라인 수 기준 집계 (job_id 빈값 문제 우회)
    lines_after = sum(1 for l in cache_path.read_text(encoding="utf-8").splitlines() if l.strip())
    success_count = lines_after - len(cached_ids)
    failed_count = len(remaining) - success_count
    print(
        f"[pass1] {category} 완료: "
        f"성공 {success_count}건 / 실패 {failed_count}건"
        + (f" → {failed_path.name}" if failed_count else "")
    )


# ── 전체 실행 ──────────────────────────────────────────────────────────────────

async def run(category: str | None = None, force: bool = False) -> None:
    """category=None 이면 data/compressed/ 아래 전체 카테고리 순차 실행."""
    if category:
        await run_category(category, force=force)
        return

    cat_files = sorted(COMPRESSED_DIR.glob("*.json"))
    if not cat_files:
        print("[pass1] data/compressed/ 에 파일 없음. --step classify 먼저 실행하세요.")
        return

    for cat_file in cat_files:
        await run_category(cat_file.stem, force=force)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default=None, help="특정 카테고리만 실행")
    parser.add_argument("--force", action="store_true", help="캐시 무시하고 재실행")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    asyncio.run(run(category=args.category, force=args.force))
