from pathlib import Path
import asyncio
import os
from scrapers.wiki.utils import get_latest_dumpstatus_url, fetch_dumpstatus, is_dump_done, get_download_urls, pair_wiki_files, get_unique_indices, multistream_to_mongodb
from scrapers.wiki.async_func import run_scraper
from scrapers.config import WikiScraperSettings
from backend.db.mongodb.connection import MongoManager
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

wiki_scraper_settings = WikiScraperSettings()

download_path = os.getenv('WIKI_DOWNLOAD_PATH', './data/wiki_dumps')
MONGODB_URI = wiki_scraper_settings.mongodb_uri
mongodb_client = MongoManager(MONGODB_URI, "scraper_db")
RSS_URL = wiki_scraper_settings.RSS_URL

if __name__ == "__main__":
    logger.info('WIKI SCRAPER')
    if not mongodb_client.is_healthy():
        logger.critical('Could not establish connection with MongoDB')
        exit(1)
    Path(download_path).mkdir(parents=True, exist_ok=True)
    dumpstatus_url = get_latest_dumpstatus_url(RSS_URL)
    dumpstatus = fetch_dumpstatus(dumpstatus_url)
    
    articlesmultistreamdump = dumpstatus.get('jobs',{}).get('articlesmultistreamdump')
    if not is_dump_done(articlesmultistreamdump):
        exit(1)
    download_urls = get_download_urls(articlesmultistreamdump)
    asyncio.run(run_scraper(download_urls, download_path))

    index_multistream_pairs = pair_wiki_files(download_path)

    for pair in index_multistream_pairs:
        indices = get_unique_indices(download_path+pair['index'])
        multistream_to_mongodb(mongodb_client, download_path+pair['multistream'], indices)


