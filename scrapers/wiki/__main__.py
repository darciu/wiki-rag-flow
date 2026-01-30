from pathlib import Path
import asyncio
import pickle
from scrapers.wiki.utils import get_latest_dumpstatus_url, fetch_dumpstatus, is_dump_done, get_download_urls, pair_wiki_files, get_unique_indices, read_multistream
from scrapers.wiki.async_func import run_scraper
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    logger.info('WIKI SCRAPER')
    Path("data").mkdir(parents=True, exist_ok=True)
    dumpstatus_url = get_latest_dumpstatus_url()
    dumpstatus = fetch_dumpstatus(dumpstatus_url)
    
    articlesmultistreamdump = dumpstatus.get('jobs',{}).get('articlesmultistreamdump')
    if not is_dump_done(articlesmultistreamdump):
        exit(1)
    download_urls = get_download_urls(articlesmultistreamdump)
    print(download_urls)
    asyncio.run(run_scraper(download_urls))

    index_multistream_pairs = pair_wiki_files("data")

    for pair in index_multistream_pairs:
        indices = get_unique_indices("data/"+pair['index'])
        all_pages = read_multistream("data/"+pair['multistream'], indices)
        with open(f"{'temp/'+pair['multistream']}.pkl", "wb") as f:
            pickle.dump(all_pages, f)

