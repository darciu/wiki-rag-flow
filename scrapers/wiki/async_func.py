import asyncio
import logging
from pathlib import Path

import aiohttp
import anyio
from tqdm import tqdm

from scrapers.wiki.utils import check_md5

logger = logging.getLogger(__name__)

MAX_CONCURRENT_DOWNLOADS = 3

HEADERS = {"User-Agent": "wiki-rag-flow (contact: giemzadariusz@gmail.com)"}


async def download_file(
    url: str,
    wiki_md5: str,
    download_path: str,
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
) -> None:
    """Downloads a file from a URL with retry logic and MD5 verification."""
    filename = url.split("/")[-1]
    file_full_path = Path(download_path) / filename
    max_retries = 3

    if anyio.Path(file_full_path).exists():
        logger.info(f"File {filename} already exists")
        return

    async with semaphore:
        for attempt in range(1, max_retries + 1):
            try:
                if attempt > 1:
                    asyncio.sleep(10)
                logging.info(f"Attempt: {attempt} | Started downloading: {filename}")
                async with session.get(url, timeout=None) as response:
                    response.raise_for_status()

                    total_size = int(response.headers.get("content-length", 0))

                    progress_bar = tqdm(
                        total=total_size,
                        unit="B",
                        unit_scale=True,
                        desc=filename,
                        leave=False,
                    )

                    with Path.open(file_full_path, "wb") as f:
                        async for chunk in response.content.iter_chunked(
                            1024 * 64
                        ):  # 64KB chunks
                            if chunk:
                                f.write(chunk)
                                progress_bar.update(len(chunk))
                    logger.info(f"Finished downloading: {filename}")

                    if not check_md5(file_full_path, wiki_md5):
                        logger.error(
                            f"MD5 of downloaded file {filename} is not correct"
                        )
                        file = anyio.Path(file_full_path)
                        file.unlink()
                        logger.info(f"File {filename} has been deleted")
                    else:
                        logger.info(f"File {filename} has passed md5 check")
            except Exception as e:
                logger.exception(f"Following exception has occured: {e}")


async def run_scraper(download_urls: list[dict[str, str]], download_path: str) -> None:
    """Orchestrates the scraping process using a semaphore to limit concurrency."""
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_DOWNLOADS)

    async with aiohttp.ClientSession(headers=HEADERS) as session:
        tasks = []
        for item in download_urls:
            task = asyncio.create_task(
                download_file(
                    item["url"], item["md5"], download_path, session, semaphore
                )
            )
            tasks.append(task)

        await asyncio.gather(*tasks)
