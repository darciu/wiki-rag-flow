from pydantic_settings import BaseSettings, SettingsConfigDict


class MongoDBSettings(BaseSettings):
    MONGO_INITDB_ROOT_USERNAME: str = ""
    MONGO_INITDB_ROOT_PASSWORD: str = ""
    MONGO_PORT: int = 27017

    @property
    def mongodb_uri(self) -> str:
        return f"mongodb://{self.MONGO_INITDB_ROOT_USERNAME}:{self.MONGO_INITDB_ROOT_PASSWORD}@mongodb:{self.MONGO_PORT}/?authSource=admin"
    
    @property
    def mongodb_local_uri(self) -> str:
        return f"mongodb://{self.MONGO_INITDB_ROOT_USERNAME}:{self.MONGO_INITDB_ROOT_PASSWORD}@localhost:{self.MONGO_PORT}/?authSource=admin"

    # Load envs from .env file, get only relevant variables, variables are case sensitive
    model_config = SettingsConfigDict(
        env_file=".env", extra="ignore", case_sensitive=True
    )
class WeaviateSettings(BaseSettings):
    WEAVIATE_APIKEY_KEY: str = ""
    WEAVIATE_EMBEDDING_SERVER_URL: str = "http://0.0.0.0:8001/embed"
    WEAVIATE_PORT: int = 8080
    WEAVIATE_GRPC_PORT: int = 50051

    # Load envs from .env file, get only relevant variables, variables are case sensitive
    model_config = SettingsConfigDict(
        env_file=".env", extra="ignore", case_sensitive=True
    )


class ScraperSettings(BaseSettings):

    WIKI_DOWNLOAD_PATH: str = "/app/data/wiki_dumps/"
    RSS_URL: str = "https://dumps.wikimedia.org/plwiki/latest/plwiki-latest-pages-articles-multistream-index.txt.bz2-rss.xml"