from typing import List, Dict, Any, Optional, Type
from types import TracebackType
from pymongo import MongoClient, UpdateOne
from pymongo.errors import ConnectionFailure, OperationFailure
from pymongo.results import BulkWriteResult


class MongoManager:
    def __init__(self, uri: str, db_name: str):

        self.client = MongoClient(uri)
        self.db = self.client[db_name]

    def __enter__(self) -> "MongoManager":
        """Enter the runtime context and return the instance"""
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
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
        self, collection_name: str, batch: List[Dict[str, Any]], id_field: str = "_id"
    ) -> Optional[BulkWriteResult]:
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
