import logging
import os

import logging_loki


def setup_logging(service: str):
    loki_url = os.getenv("LOKI_ENDPOINT", "http://localhost:3100/loki/api/v1/push")

    loki_handler = logging_loki.LokiHandler(
        url=loki_url,
        tags={"app": "wiki_rag_flow", "env": "production", "service": service},
        version="1",
    )
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - [%(pathname)s:%(lineno)d] - %(funcName)s() - %(message)s"
    )
    console_handler.setFormatter(console_formatter)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    root_logger.handlers.clear()
    root_logger.addHandler(loki_handler)
    root_logger.addHandler(console_handler)
