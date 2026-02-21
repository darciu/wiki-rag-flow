# Makefile

SHELL := /bin/bash

EMB_DIR := embedding-server
EMB_VENV := $(EMB_DIR)/.venv
EMB_PY := $(EMB_VENV)/bin/python
EMB_UVICORN := $(EMB_VENV)/bin/uvicorn
EMB_PID := $(EMB_DIR)/embedding_server.pid
EMB_LOG := $(EMB_DIR)/embedding_server.log

HOST ?= 0.0.0.0

EMB_PORT ?= 8008
EMB_BASE := http://$(HOST):$(EMB_PORT)


.PHONY: emb-venv emb-run-bg emb-stop emb-health

build-scraper:
	docker-compose build scraper

run-scraper:
	docker-compose run --rm scraper

build-parser:
	docker-compose build parser

run-parser:
	docker-compose run parser

run-mongo:
	docker-compose up -d mongodb

run-weaviate:
	docker-compose up -d weaviate

# --- embedding server (native) ---
emb-venv:
	python3 -m venv $(EMB_VENV)
	$(EMB_PY) -m pip install -U pip
	$(EMB_PY) -m pip install fastapi "uvicorn[standard]" sentence-transformers torch

emb-run-bg: emb-venv
	@mkdir -p $(EMB_DIR)
	@nohup $(EMB_UVICORN) embedding_server:app --app-dir $(EMB_DIR) --host $(HOST) --port $(EMB_PORT) \
		> $(EMB_LOG) 2>&1 & echo $$! > $(EMB_PID)
	@echo "Started on $(EMB_BASE) (pid: `cat $(EMB_PID)`)"

emb-stop:
	@test -f $(EMB_PID) && kill `cat $(EMB_PID)` && rm -f $(EMB_PID) || true

emb-health:
	@curl -fsS "$(EMB_BASE)/health" && echo || (echo "Healthcheck failed. See $(EMB_LOG)"; exit 1)

project-up: emb-run-bg
	docker-compose up --build -d

project-down: emb-stop
	docker-compose down



NLP_HOST ?= 127.0.0.1
NLP_PORT ?= 8009
NLP_WORKERS ?= 1
NLP_APP_MODULE ?= local_servers.nlp_toolkit:app

# --- Konfiguracja Modeli (Domyślne wartości) ---
NLP_NER_MODEL ?= herbert
NLP_KEYWORDS_MODEL ?= keybert
NLP_CHUNKING_MODEL ?= langchain

.PHONY: help run-nlp


nlp-run: ## Uruchamia serwer na pojedynczym procesie
	NLP_NER_MODEL=$(NLP_NER_MODEL) \
	NLP_KEYWORDS_MODEL=$(NLP_KEYWORDS_MODEL) \
	NLP_CHUNKING_MODEL=$(NLP_CHUNKING_MODEL) \
	uvicorn $(NLP_APP_MODULE) --host $(NLP_HOST) --port $(NLP_PORT)

nlp-health: ## Sprawdza status serwera
	@curl -f http://$(NLP_HOST):$(NLP_PORT)/health || echo "\nBŁĄD: Serwer nie odpowiada na porcie $(NLP_PORT)"

nlp-test-chunk: ## Testuje endpoint chunkowania
	@echo "Wysyłanie testowego tekstu do http://$(NLP_HOST):$(NLP_PORT)/chunk..."
	@curl -X POST http://$(NLP_HOST):$(NLP_PORT)/chunk \
		-H "Content-Type: application/json" \
		-d '{"texts": ["To jest bardzo długi tekst, który powinien zostać podzielony na mniejsze części przez model langchain.To jest bardzo długi tekst, który powinien zostać podzielony na mniejsze części przez model langchain.To jest bardzo długi tekst, który powinien zostać podzielony na mniejsze części przez model langchain."], "max_tokens": 60}' \
		-w "\n"



# prod: ## Uruchamia serwer dla multiprocessingu z określoną liczbą workerów
# 	NLP_NER_MODEL=$(NLP_NER_MODEL) \
# 	NLP_KEYWORDS_MODEL=$(NLP_KEYWORDS_MODEL) \
# 	NLP_CHUNKING_MODEL=$(NLP_CHUNKING_MODEL) \
# 	uvicorn $(APP_MODULE) --host $(HOST) --port $(PORT) --workers $(WORKERS)
