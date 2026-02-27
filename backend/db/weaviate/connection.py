from types import TracebackType
from typing import Any
import weaviate.classes.query as wq
import requests
import weaviate
import weaviate.classes.config as wc
from weaviate.classes.init import Auth
from weaviate.collections import Collection
from weaviate.util import generate_uuid5


class WeaviateManager:
    def __init__(
        self,
        api_key: str,
        native_embedding_url: str = "http://host.docker.internal:8008/embed",
        host="local_weaviate",
        port: int = 8080,
        grpc_port: int = 50051,
    ):
        self.native_embedding_url = native_embedding_url
        self.client = weaviate.connect_to_custom(
            http_host=host,
            http_port=port,
            http_secure=False,
            grpc_host=host,
            grpc_port=grpc_port,
            grpc_secure=False,
            auth_credentials=Auth.api_key(api_key),
        )

    def __enter__(self):
        """
        Context manager entry point.
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """
        Context manager exit point. Ensures the client is closed.
        """
        self.close()

    def close(self) -> None:
        """
        Close the connection to Weaviate.
        """
        if self.client.is_connected():
            self.client.close()

    def is_healthy(self) -> bool:
        """
        Check if Weaviate is ready.
        """
        return self.client.is_ready()

    def get_metadata(self):
        """
        Get cluster metadata.
        """
        return self.client.get_meta()

    def create_wiki_chunk_collection(self) -> Collection:
        """
        Get the WikiChunk collection, creating it if it doesn't exist.
        """
        collection_name = "WikiChunk"

        if not self.client.collections.exists(collection_name):
            self.client.collections.create(
                name=collection_name,
                vectorizer_config=None,
                properties=[
                    wc.Property(
                        name="source_id",
                        data_type=wc.DataType.TEXT,
                        skip_vectorization=True,
                        tokenization=wc.Tokenization.FIELD,
                    ),
                    wc.Property(
                        name="source_title",
                        data_type=wc.DataType.TEXT,
                        skip_vectorization=False,
                    ),
                    wc.Property(
                        name="wiki_categories",
                        data_type=wc.DataType.TEXT_ARRAY,
                        skip_vectorization=False,
                    ),
                    wc.Property(
                        name="chunk_text",
                        data_type=wc.DataType.TEXT,
                        skip_vectorization=False,
                    ),
                    wc.Property(
                        name="imie_i_nazwisko",
                        data_type=wc.DataType.TEXT,
                        skip_vectorization=True,
                    ),
                    wc.Property(
                        name="imie", data_type=wc.DataType.TEXT, skip_vectorization=True
                    ),
                    wc.Property(
                        name="data_urodzenia",
                        data_type=wc.DataType.TEXT,
                        skip_vectorization=True,
                    ),
                    wc.Property(
                        name="miejsce_urodzenia",
                        data_type=wc.DataType.TEXT,
                        skip_vectorization=True,
                    ),
                    wc.Property(
                        name="data_smierci",
                        data_type=wc.DataType.TEXT,
                        skip_vectorization=True,
                    ),
                    wc.Property(
                        name="miejsce_smierci",
                        data_type=wc.DataType.TEXT,
                        skip_vectorization=True,
                    ),
                    wc.Property(
                        name="obywatelstwo",
                        data_type=wc.DataType.TEXT,
                        skip_vectorization=True,
                    ),
                    wc.Property(
                        name="nazwa",
                        data_type=wc.DataType.TEXT,
                        skip_vectorization=True,
                    ),
                    wc.Property(
                        name="nazwa_zwyczajowa",
                        data_type=wc.DataType.TEXT,
                        skip_vectorization=True,
                    ),
                    wc.Property(
                        name="panstwo",
                        data_type=wc.DataType.TEXT,
                        skip_vectorization=True,
                    ),
                    wc.Property(
                        name="kraj", data_type=wc.DataType.TEXT, skip_vectorization=True
                    ),
                    wc.Property(
                        name="miejscowosc",
                        data_type=wc.DataType.TEXT,
                        skip_vectorization=True,
                    ),
                    wc.Property(
                        name="tytul",
                        data_type=wc.DataType.TEXT,
                        skip_vectorization=True,
                    ),
                    wc.Property(
                        name="liczba_ludnosci",
                        data_type=wc.DataType.TEXT,
                        skip_vectorization=True,
                    ),
                    wc.Property(
                        name="rok", data_type=wc.DataType.TEXT, skip_vectorization=True
                    ),
                    wc.Property(name="chunk_id", data_type=wc.DataType.INT),
                ],
            )

        collection = self.client.collections.get(collection_name)
        return collection

    def embed_batch_natively(self, texts: list[str]) -> list[list[float]]:
        """
        Call the native (bare-metal) embedding service.
        """
        try:
            r = requests.post(
                self.native_embedding_url,
                json={"texts": texts, "normalize": True},
                timeout=120,
            )
            r.raise_for_status()
            return r.json()["vectors"]
        except requests.RequestException as e:
            print(f"Error connecting to embedding service: {e}")
            raise e

    def build_embedding_input_wiki_chunk(self, item: dict) -> str:
        """
        Construct the text representation to be embedded.
        """
        title = (item.get("source_title") or "").strip()
        chunk = (item.get("chunk_text") or "").strip()
        cats = item.get("wiki_categories") or []
        if isinstance(cats, (list, tuple)):
            cats_str = ", ".join([str(c).strip() for c in cats if c])
        else:
            cats_str = str(cats).strip()

        parts = []
        if title:
            parts.append(f"Tytuł: {title}")
        if cats_str:
            parts.append(f"Kategorie: {cats_str}")
        if chunk:
            parts.append(f"Treść: {chunk}")

        return "\n".join(parts)

    def bulk_upsert(self, data_items: list[dict[str, Any]]) -> None:
        """
        Embed and insert a batch of items into Weaviate.
        """
        if not data_items:
            print("No items to process.")
            return
        texts = [self.build_embedding_input_wiki_chunk(item) for item in data_items]
        vectors = self.embed_batch_natively(texts)

        collection = self.create_wiki_chunk_collection()

        with collection.batch.dynamic() as batch:
            for item, vector in zip(data_items, vectors, strict=True):
                unique_id_str = f"{item['source_id']}_{item['chunk_id']}"
                # Create deterministic uuid based on source_id and chunk_id
                object_uuid = generate_uuid5(unique_id_str)

                batch.add_object(
                    properties=item,
                    uuid=object_uuid,
                    vector=vector,
                )

        if collection.batch.failed_objects:
            print(f"Errors: {len(collection.batch.failed_objects)}")
            print(f"First error: {collection.batch.failed_objects[0]}")
        else:
            print(f"Successfully loaded batch of {len(data_items)} items.")

    def clear_collection(self, collection_name: str) -> None:
        """
        Remove collection definition with all the data inside
        """
        self.client.collections.delete(collection_name)

    

    def single_wikichunk_hybrid_fetch(self, query_text, query_vector, weaviate_limit, alpha):

        collection_name = "WikiChunk"
        collection = self.client.collections.get(collection_name)
        response = collection.query.hybrid(
                query=query_text,
                vector=query_vector,
                limit=weaviate_limit,
                alpha=alpha,
                return_properties=["source_id", "source_title", "chunk_id", "chunk_text"],
                return_metadata=wq.MetadataQuery(score=True) 
            )
        
        query_results = []
        for obj in response.objects:
            query_results.append({
                "source_id": obj.properties["source_id"],
                "source_title": obj.properties["source_title"],
                "chunk_id": obj.properties["chunk_id"],
                "chunk_text": obj.properties["chunk_text"],
                "score": obj.metadata.score
            })

        return query_results
    
    def wikichunk_combined_filter(self, grouped_source_chunk_id):

        filters = []
        for s_id, c_ids in grouped_source_chunk_id.items():
            f = wq.Filter.by_property("source_id").equal(s_id) & wq.Filter.by_property("chunk_id").contains_any(c_ids)
            filters.append(f)

        combined_filter = wq.Filter.any_of(filters) if len(filters) > 1 else filters[0]

        return combined_filter
    
    def batch_wikichunk_fetch(self, grouped_source_chunk_id):

        collection_name = "WikiChunk"
        collection = self.client.collections.get(collection_name)
        combined_filter = self.wikichunk_combined_filter(grouped_source_chunk_id)

        response = collection.query.fetch_objects(
            filters=combined_filter,
            limit=len(grouped_source_chunk_id) +1,
            return_properties=["source_id", "source_title", "chunk_id", "chunk_text"]
        )
        fetched_chunks = []
        for obj in response.objects:
            fetched_chunks.append({
                "source_id": obj.properties["source_id"],
                "source_title": obj.properties.get("source_title", ""),
                "chunk_id": obj.properties["chunk_id"],
                "chunk_text": obj.properties["chunk_text"],
                "score": 0.0, 
                "rank_score": -999.0 
            })

        return fetched_chunks
