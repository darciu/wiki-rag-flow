// init-db.js
// create database
db = db.getSiblingDB('scraper_db');

db.createCollection('wikipedia');
db.createCollection('wiki_plain_articles');


db.wikipedia.createIndex({ "_id": 1 }, { unique: true });
db.wiki_plain_articles.createIndex({ "_id": 1 }, { unique: true });

print('Database scraper_db is now initialized')