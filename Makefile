# Makefile

build-scraper:
	docker-compose build scraper

run-scraper:
	docker-compose run --rm scraper

run-mongo:
	docker-compose up -d mongodb

project-up:
	docker-compose up --build -d