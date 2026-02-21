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

    def mark_processed(self, collection_name: str, ids: list[Any]) -> None:
        if not ids:
            return
        col = self.db[collection_name]
        col.update_many({"_id": {"$in": ids}}, {"$set": {"processed": True}})
    
    def fetch_unprocessed_batches(
        self,
        collection_name: str,
        filter_query: dict[str, Any] | None = None,
        projection: dict[str, Any] | None = None,
        batch_size: int = 1000,
    ) -> Generator[list[dict[str, Any]], None, None]:

        col = self.db[collection_name]
        last_id = None

        base = filter_query.copy() if filter_query else {}
        base["processed"] = {"$ne": True}  # brak pola traktuje jako nieprzetworzone

        while True:
            q = dict(base)
            if last_id is not None:
                q["_id"] = {"$gt": last_id}

            batch = list(
                col.find(q, projection or {})
                .sort("_id", 1)
                .limit(batch_size)
            )
            if not batch:
                break

            last_id = batch[-1]["_id"]
            yield batch

    def clear_collection(self, collection_name: str) -> int:
        """
        Usuwa wszystkie dokumenty z podanej kolekcji. 
        Zachowuje samą kolekcję oraz zdefiniowane na niej indeksy.
        
        :param collection_name: Nazwa kolekcji do wyczyszczenia.
        :return: Liczba usuniętych dokumentów.
        """
        collection = self.db[collection_name]
        result = collection.delete_many({})
        return result.deleted_count
    
    def get_collections_info(self) -> list[dict[str, Any]]:
        """
        Zwraca listę wszystkich kolekcji wraz z ich metadanymi:
        liczbą dokumentów, rozmiarem danych oraz zdefiniowanymi indeksami.
        """
        collections_metadata = []
        
        # Pobieramy nazwy wszystkich kolekcji (wykluczając systemowe)
        collection_names = self.db.list_collection_names()
        
        for name in collection_names:
            collection = self.db[name]
            
            # Pobieramy statystyki (rozmiar w bajtach, liczba dokumentów)
            # Uwaga: collstats jest bardzo szybkie
            stats = self.db.command("collStats", name)
            
            # Pobieramy listę indeksów
            indexes = collection.index_information()
            
            collections_metadata.append({
                "name": name,
                "count": stats.get("count", 0),
                "size_kb": round(stats.get("size", 0) / 1024, 2),
                "storage_size_kb": round(stats.get("storageSize", 0) / 1024, 2),
                "indexes": indexes
            })
            
        return collections_metadata
    

    def get_document_count(self, collection_name: str) -> int:
        collection = self.db[collection_name]
        return collection.estimated_document_count()
