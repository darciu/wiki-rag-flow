# Makefile

SHELL := /bin/bash

EMB_DIR := embedding-server
EMB_VENV := $(EMB_DIR)/.venv
EMB_PY := $(EMB_VENV)/bin/python
EMB_UVICORN := $(EMB_VENV)/bin/uvicorn
EMB_PID := $(EMB_DIR)/embedding_server.pid
EMB_LOG := $(EMB_DIR)/embedding_server.log
OLLAMA_PID := ollama.pid

HOST ?= 0.0.0.0

EMB_PORT ?= 8008
EMB_BASE := http://$(HOST):$(EMB_PORT)

ifeq ($(OS),Windows_NT)
    OPEN := start
else
    UNAME_S := $(shell uname -s)
    ifeq ($(UNAME_S),Linux)
        OPEN := xdg-open
    endif
    ifeq ($(UNAME_S),Darwin)
        OPEN := open
    endif
endif

FRONTEND_URL := http://localhost:8501
BACKEND_URL := http://localhost:8000
PHOENIX_URL := http://localhost:6006/projects


.PHONY: emb-venv emb-run-bg emb-stop emb-health ollama-up ollama-stop ollama-pull-llama ollama-pull-qwen project-up project-down open-hosts

open-hosts:
	@echo "Otwieram hosty w przeglądarce..."
	@$(OPEN) $(FRONTEND_URL)
	@$(OPEN) $(BACKEND_URL)
	@$(OPEN) $(PHOENIX_URL)

build-scraper:
	docker-compose build scraper

run-scraper:
	docker-compose run --rm scraper

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

project-up: emb-run-bg ollama-up
	docker-compose up --build -d

project-down: emb-stop ollama-stop
	docker-compose down

uvicorn-up:
	uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000

streamlit-up:
	streamlit run frontend/streamlit.py

ollama-set-env:
	launchctl setenv OLLAMA_HOST "0.0.0.0"
	@echo "Zmienna OLLAMA_HOST ustawiona na 0.0.0.0. Zrestartuj aplikację Ollama, jeśli była otwarta."

ollama-up:
	OLLAMA_HOST=0.0.0.0 OLLAMA_KEEP_ALIVE=-1 nohup ollama serve > ollama.log 2>&1 & echo $$! > $(OLLAMA_PID)
	@sleep 2

ollama-stop:
	kill $$(cat $(OLLAMA_PID)) 2>/dev/null || true
	rm -f $(OLLAMA_PID)
	
ollama-pull-llama: ollama-up
	ollama pull llama3.2

ollama-pull-gemma: ollama-up
	ollama pull gemma3:4b

ollama-pull-qwen: ollama-up
	ollama pull qwen2.5:7b-instruct

ollama-health:
	@curl -fsS "http://localhost:11434/api/tags" > /dev/null && echo "Ollama is Healthy" || (echo "Ollama is NOT running"; exit 1)





