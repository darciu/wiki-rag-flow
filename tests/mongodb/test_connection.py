from unittest.mock import patch

import mongomock

from backend.db.mongodb.connection import MongoManager


def test_is_healthy_success():

    with patch("backend.db.mongodb.connection.MongoClient") as mock_client_class:
        mock_instance = mock_client_class.return_value
        mock_instance.admin.command.return_value = {"ok": 1.0}
        manager = MongoManager("fake_uri", "test_db")
        assert manager.is_healthy() is True


def test_bulk_upsert_logic():
    with patch("backend.db.mongodb.connection.MongoClient", mongomock.MongoClient):
        manager = MongoManager("mongodb://localhost", "test_db")
        data = [
            {"_id": "1", "val": "<p>Hello world!</p>"},
            {"_id": "2", "val": '<div class="test">Some wiki article</div>'},
        ]

        result = manager.bulk_upsert("test_col", data)

        assert manager.db["test_col"].count_documents({}) == 2
        assert (
            manager.db["test_col"].find_one({"_id": "1"})["val"]
            == "<p>Hello world!</p>"
        )
        assert result is not None
        assert result.upserted_count == 2
