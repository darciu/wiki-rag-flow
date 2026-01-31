// init-db.js
// create database
db = db.getSiblingDB('scraper_db');

db.createCollection('wikipedia');


db.wikipedia.createIndex({ "_id": 1 }, { unique: true });

print('Database scraper_db is now initialized')