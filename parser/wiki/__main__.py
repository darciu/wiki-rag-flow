import logging
import mwparserfromhell
import re
import html
from backend.db.mongodb.connection import MongoManager
from config import MongoDBSettings
from parser.nlp.wiki_tools import fetch_wiki_categories, fetch_wiki_infobox_data, fetch_wiki_clean_sections

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

mongodb_settings = MongoDBSettings()

MONGODB_URI = mongodb_settings.mongodb_uri
mongodb_client = MongoManager(MONGODB_URI, "scraper_db")


if __name__ == "__main__":
    logger.info("PARSER WIKI")
    if not mongodb_client.is_healthy():
        logger.critical("PARSER WIKI Could not establish connection with MongoDB")
        exit(1)
    # analogicznie sprawdzenie połączenia z weaviate
        
    
    generator = mongodb_client.fetch_batches("wikipedia", batch_size=100)
    for batch in generator:

        for record in batch:
            if any(x in record['title'] for x in ['Kategoria:','Wątek:','Wikipedia:','Szablon:', 'Moduł:', 'Portal:', 'MediaWiki:', 'Pomoc:', 'Wikiprojekt:', 'Plik:']):
                pass
            else:
                content = record['content']
                wikicode = mwparserfromhell.parse(content)
                wiki_categories = fetch_wiki_categories(wikicode)
                wiki_infobox_data = fetch_wiki_infobox_data(wikicode)
                wiki_sections = fetch_wiki_clean_sections(content)

