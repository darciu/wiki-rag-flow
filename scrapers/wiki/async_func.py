import asyncio
import aiohttp
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import logging
from scrapers.wiki.utils import check_md5

logger = logging.getLogger(__name__)

MAX_CONCURRENT_DOWNLOADS = 3

HEADERS = {
    "User-Agent": "wiki-rag-flow (contact: giemzadariusz@gmail.com)"
}

async def download_file(url, wiki_md5, session, semaphore):
    filename = url.split('/')[-1]
    max_retries = 3

    if Path('data/'+filename).exists():
        logger.info(f'File {filename} already exists')
        return
    
    
    async with semaphore:
        for attempt in range(1, max_retries+1):
            
            try:
                if attempt > 1:
                    asyncio.sleep(10)
                logging.info(f"Attempt: {attempt} | Started downloading: {filename}")
                async with session.get(url, timeout=None) as response:
                    response.raise_for_status()

                    total_size = int(response.headers.get('content-length', 0))
                    
                    progress_bar = tqdm(
                        total=total_size, 
                        unit='B', 
                        unit_scale=True, 
                        desc=filename, 
                        leave=False
                    )

                    with open('data/'+filename, 'wb') as f:
                        async for chunk in response.content.iter_chunked(1024 * 64): # 64KB chunks
                            if chunk:
                                f.write(chunk)
                                progress_bar.update(len(chunk))
                    logger.info(f"Finished downloading: {filename}")

                    if not check_md5('tmp/'+filename, wiki_md5):
                        logger.error(f'MD5 of downloaded file {filename} is not correct')
                        file = Path('tmp/'+filename)
                        file.unlink()
                        logger.info(f'File {filename} has been deleted')
                    else:
                        logger.info(f'File {filename} has passed md5 check')
            except Exception as e:
                logger.exception(f"Following exception has occured: {e}")
                


async def run_scraper(download_urls):
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_DOWNLOADS)
    
    async with aiohttp.ClientSession(headers=HEADERS) as session:
        tasks = []
        for item in download_urls:
            task = asyncio.create_task(download_file(item['url'], item['md5'], session, semaphore))
            tasks.append(task)
        
        await asyncio.gather(*tasks)

