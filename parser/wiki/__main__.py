# parser/wiki/__main__.py
import logging
import math
import sys
import time

from backend.db.mongodb.connection import MongoManager
from backend.db.weaviate.connection import WeaviateManager
from config import MongoDBSettings, WeaviateSettings
from parser.nlp.toolkit import NLPToolkit
from parser.nlp.utils import process_batch

logger = logging.getLogger(__name__)


def main():
    mongodb_settings = MongoDBSettings()
    weaviate_settings = WeaviateSettings()
    mongo_uri = mongodb_settings.mongodb_local_uri
    weaviate_api_key = weaviate_settings.WEAVIATE_APIKEY_KEY

    logger.info("Hello!")
    batch_size = 512

    mongodb_client = MongoManager(mongo_uri, "scraper_db")
    if not mongodb_client.is_healthy():
        sys.exit(1)

    weaviate_client = WeaviateManager(
        api_key=weaviate_api_key,
        host="127.0.0.1",
        native_embedding_url="http://127.0.0.1:8008/embed",
    )
    if not weaviate_client.is_healthy():
        sys.exit(1)

    nlp_toolkit = NLPToolkit()

    with mongodb_client, weaviate_client:
        total_docs = mongodb_client.get_document_count("wikipedia")
        docs_already_loaded = mongodb_client.get_document_count("wiki_plain_articles")
        expected_total_batches = math.ceil(
            (total_docs - docs_already_loaded) / batch_size
        )

        generator = mongodb_client.fetch_unprocessed_batches(
            "wikipedia",
            projection={"_id": 1, "title": 1, "content": 1},
            batch_size=batch_size,
        )

        time_start = time.time()

        for batch_idx, batch in enumerate(generator):
            process_batch(
                batch,
                batch_idx,
                expected_total_batches,
                time_start,
                mongodb_client,
                weaviate_client,
                nlp_toolkit,
            )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
