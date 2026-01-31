from pymongo import MongoClient, UpdateOne
from pymongo.errors import ConnectionFailure, OperationFailure


class MongoManager:
    def __init__(self, uri, db_name):
        self.client = MongoClient(uri)
        self.db = self.client[db_name]


    def is_healthy(self):
        try:
            self.client.admin.command('ping')
            return True
        except (ConnectionFailure, OperationFailure):
            return False

    
    def bulk_upsert(self, collection_name: str, batch: list, id_field="_id"):
        
        operations = [
            UpdateOne(
                {id_field: doc[id_field]},
                {"$set": doc},
                upsert=True
            )
            for doc in batch
        ]
        
        if operations:
            collection = self.db[collection_name]
            return collection.bulk_write(operations, ordered=False)