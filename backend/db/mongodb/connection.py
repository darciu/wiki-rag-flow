from types import TracebackType
from typing import Any, Generator

from pymongo import MongoClient, UpdateOne
from pymongo.database import Database
from pymongo.errors import ConnectionFailure, OperationFailure
from pymongo.results import BulkWriteResult


class MongoManager:
    client: MongoClient[Any]
    db: Database[Any]

    def __init__(self, uri: str, db_name: str):

        self.client = MongoClient(uri)
        self.db = self.client[db_name]

    def __enter__(self) -> "MongoManager":
        """Enter the runtime context and return the instance"""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit the runtime context and close the connection."""
        self.close()

    def close(self) -> None:
        """Close manually the MongoDB client connection"""
        self.client.close()

    def is_healthy(self) -> bool:
        """Verify the connection to the database server using a ping command."""
        try:
            self.client.admin.command("ping")
            return True
        except (ConnectionFailure, OperationFailure):
            return False

    def bulk_upsert(
        self, collection_name: str, batch: list[dict[str, Any]], id_field: str = "_id"
    ) -> BulkWriteResult | None:
        """
        Perform a batch update/insert (upsert) operation.
        """
        operations = [
            UpdateOne({id_field: doc[id_field]}, {"$set": doc}, upsert=True)
            for doc in batch
        ]

        if operations:
            collection = self.db[collection_name]
            return collection.bulk_write(operations, ordered=False)
        return None
    
    def fetch_batches(
        self, 
        collection_name: str, 
        filter_query: dict[str, Any] | None = None, 
        projection: dict[str, Any] | None = None, 
        batch_size: int = 1000
    ) -> Generator[list[dict[str, Any]], None, None]:

        collection = self.db[collection_name]

        cursor = collection.find(
            filter_query or {}, 
            projection or {}
        ).batch_size(batch_size)

        batch = []
        for doc in cursor:
            batch.append(doc)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        
        if batch:
            yield batch
