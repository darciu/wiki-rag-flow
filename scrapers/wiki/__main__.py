import asyncio
import logging
from pathlib import Path

from backend.db.mongodb.connection import MongoManager
from config import MongoDBSettings, ScraperSettings
from scrapers.wiki.async_func import run_scraper
from scrapers.wiki.utils import (
    fetch_dumpstatus,
    get_download_urls,
    get_latest_dumpstatus_url,
    get_unique_indices,
    is_dump_done,
    multistream_to_mongodb,
    pair_wiki_files,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

mongodb_settings = MongoDBSettings()
scraper_settings = ScraperSettings()

MONGODB_URI = mongodb_settings.mongodb_uri
mongodb_client = MongoManager(MONGODB_URI, "scraper_db")

WIKI_DOWNLOAD_PATH = scraper_settings.WIKI_DOWNLOAD_PATH
RSS_URL = scraper_settings.RSS_URL

if __name__ == "__main__":
    logger.info("SCRAPER WIKI")
    if not mongodb_client.is_healthy():
        logger.critical("SCRAPER WIKI Could not establish connection with MongoDB")
        exit(1)
    Path(WIKI_DOWNLOAD_PATH).mkdir(parents=True, exist_ok=True)
    dumpstatus_url = get_latest_dumpstatus_url(RSS_URL)
    if dumpstatus_url is None:
        logger.critical("Could not find dumpstatus url")
        exit(1)
    dumpstatus = fetch_dumpstatus(dumpstatus_url)

    articlesmultistreamdump = dumpstatus.get("jobs", {}).get("articlesmultistreamdump")
    if not is_dump_done(articlesmultistreamdump):
        exit(1)
    download_urls = get_download_urls(articlesmultistreamdump)
    asyncio.run(run_scraper(download_urls, WIKI_DOWNLOAD_PATH))

    index_multistream_pairs = pair_wiki_files(WIKI_DOWNLOAD_PATH)

    for pair in index_multistream_pairs:
        indices = get_unique_indices(WIKI_DOWNLOAD_PATH + pair["index"])
        multistream_to_mongodb(
            mongodb_client, WIKI_DOWNLOAD_PATH + pair["multistream"], indices
        )
