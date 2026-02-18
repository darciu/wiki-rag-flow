# parser/mp_workers.py
import atexit
import sys
import traceback
import faulthandler
import logging

mongodb_client = None
weaviate_client = None
nlp_toolkit = None

def init_worker(mongo_uri: str, weaviate_key: str):
    # Lazy importy ograniczają ryzyko efektów ubocznych podczas importu modułu w child process
    from backend.db.mongodb.connection import MongoManager
    from backend.db.weaviate.connection import WeaviateManager

    global mongodb_client, weaviate_client, nlp_toolkit
    mongodb_client = MongoManager(mongo_uri, "scraper_db")
    weaviate_client = WeaviateManager(weaviate_key)
    atexit.register(_close_clients)

def init_worker_safe(mongo_uri: str, weaviate_key: str):
    # Dzięki temu zobaczysz traceback jeśli init_worker padnie,
    # a przy segfault dostaniesz dump z faulthandlera
    faulthandler.enable()
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s %(processName)s %(name)s %(levelname)s %(message)s",
    )
    try:
        init_worker(mongo_uri, weaviate_key)
    except Exception:
        print("init_worker crashed:\n" + traceback.format_exc(), file=sys.stderr, flush=True)
        raise

def _close_clients():
    global mongodb_client, weaviate_client
    try:
        if weaviate_client:
            weaviate_client.close()
    except Exception:
        pass
    try:
        if mongodb_client:
            mongodb_client.close()
    except Exception:
        pass

def worker_task(args):
    from parser.nlp.utils import process_batch 
    batch_idx, batch_data, expected_total_batches, time_start = args
    return process_batch(
        batch_data,
        batch_idx,
        expected_total_batches,
        time_start,
        mongodb_client,
        weaviate_client,
    )