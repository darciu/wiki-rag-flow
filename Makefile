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

run-parser:
	dotenv -e .env python -m parser.wiki

run-mongo:
	docker-compose up -d mongodb

run-weaviate:
	docker-compose up -d weaviate

# --- embedding server (native) ---
emb-venv:
	python3 -m venv $(EMB_VENV)
	$(EMB_PY) -m pip install -U pip
	$(EMB_PY) -m pip install fastapi "uvicorn[standard]" sentence-transformers torch

emb-clean-port:
	@echo "Cleaning port $(EMB_PORT)..."
	@lsof -t -i:$(EMB_PORT) | xargs kill -9 2>/dev/null || echo "Port $(EMB_PORT) has been already unoccupied"

emb-run-bg: emb-clean-port emb-venv
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