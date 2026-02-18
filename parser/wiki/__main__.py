# parser/wiki/__main__.py
import multiprocessing as mp
import concurrent.futures
import time
import math
import sys
import logging
import traceback

logger = logging.getLogger(__name__)

def run_multiprocessing():
    # Importy dopiero tutaj (żeby child process przy spawn nie odpalał ciężkich importów)
    from config import MongoDBSettings, WeaviateSettings
    from backend.db.mongodb.connection import MongoManager
    from parser.mp_workers import init_worker_safe, worker_task


    mongodb_settings = MongoDBSettings()
    weaviate_settings = WeaviateSettings()
    mongo_uri = mongodb_settings.mongodb_uri
    weaviate_key = weaviate_settings.WEAVIATE_APIKEY_KEY

    logger.info("Hello!")
    batch_size = 32
    max_workers = 1

    main_mongo = MongoManager(mongo_uri, "scraper_db")
    total_docs = main_mongo.get_document_count("wikipedia")
    docs_already_loaded = main_mongo.get_document_count("wiki_plain_articles")
    expected_total_batches = math.ceil((total_docs - docs_already_loaded) / batch_size)

    generator = main_mongo.fetch_batches("wikipedia", "wiki_plain_articles", batch_size=batch_size)
    logger.info("Generator!")
    time_start = time.time()

    def arg_iter():
        for batch_idx, batch_data in enumerate(generator):
            yield (batch_idx, batch_data, expected_total_batches, time_start)

    ctx = mp.get_context("fork")

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=ctx,
        initializer=init_worker_safe,
        initargs=(mongo_uri, weaviate_key),
    ) as executor:
        try:
            # map pobiera argumenty leniwie; nie trzyma wszystkich futures w pamięci
            for _ in executor.map(worker_task, arg_iter(), chunksize=1):
                pass
        except Exception:
            logger.error("Worker crashed:\n%s", traceback.format_exc())
            executor.shutdown(wait=False, cancel_futures=True)
            sys.exit(1)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_multiprocessing()