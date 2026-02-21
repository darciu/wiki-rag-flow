# parser/wiki/worker.py
import logging
import time

from backend.db.mongodb.connection import MongoManager
from backend.db.weaviate.connection import WeaviateManager
from parser.nlp.utils import process_batch

logger = logging.getLogger(__name__)

_worker_mongo_client = None
_worker_weaviate_client = None


def init_worker(mongo_uri, weaviate_api_key):
    global _worker_mongo_client, _worker_weaviate_client

    # 1. Lekki jitter, aby procesy nie uderzyły w bazę w tej samej mikrosekundzie
    time.sleep(2)

    # 2. Inicjalizacja MongoDB
    _worker_mongo_client = MongoManager(mongo_uri, "scraper_db")

    # 3. Inicjalizacja Weaviate z prostym retry-logiem
    max_retries = 3
    for attempt in range(max_retries):
        try:
            _worker_weaviate_client = WeaviateManager(
                api_key=weaviate_api_key,
                host="127.0.0.1",  # Zamiast "local_weaviate"
                native_embedding_url="http://127.0.0.1:8008/embed",  # Zamiast host.docker.internal
            )
            # Jeśli WeaviateManager nie sprawdza połączenia w __init__,
            # warto tu wywołać np. _worker_weaviate_client.client.is_live()
            break
        except Exception as e:
            if attempt == max_retries - 1:
                logger.error(
                    f"Worker failed to connect to Weaviate after {max_retries} attempts: {e}"
                )
                raise
            wait_time = (attempt + 1) * 2
            logger.warning(
                f"Connection attempt {attempt + 1} failed, retrying in {wait_time}s..."
            )
            time.sleep(wait_time)


def process_batch_wrapper(task_args):
    # ... (kod pozostaje bez zmian)
    batch_idx, batch_data, expected_total_batches, time_start = task_args
    return process_batch(
        batch_data,
        batch_idx,
        expected_total_batches,
        time_start,
        _worker_mongo_client,
        _worker_weaviate_client,
    )
