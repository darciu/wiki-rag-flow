from pathlib import Path
import asyncio
import pickle
from scrapers.wiki.utils import get_latest_dumpstatus_url, fetch_dumpstatus, is_dump_done, get_download_urls, pair_wiki_files, get_unique_indices, multistream_to_mongodb
from scrapers.wiki.async_func import run_scraper
from backend.db.mongodb.connection import MongoManager
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
download_path = 'scrapers/wiki/downloaded_data/'
MONGODB_URI = "mongodb://admin:pass123@localhost:27017/"
mongodb_client = MongoManager(MONGODB_URI, "scraper_db")
rss_url = "https://dumps.wikimedia.org/plwiki/latest/plwiki-latest-pages-articles-multistream-index.txt.bz2-rss.xml"

if __name__ == "__main__":
    logger.info('WIKI SCRAPER')
    if not mongodb_client.is_healthy():
        logger.critical('Could not establish connection with MongoDB')
        exit(1)
    Path(download_path).mkdir(parents=True, exist_ok=True)
    dumpstatus_url = get_latest_dumpstatus_url(rss_url)
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


