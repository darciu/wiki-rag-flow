from pydantic_settings import BaseSettings, SettingsConfigDict


class WikiScraperSettings(BaseSettings):
    MONGO_INITDB_ROOT_USERNAME: str = ""
    MONGO_INITDB_ROOT_PASSWORD: str = ""
    WIKI_DOWNLOAD_PATH: str = "/app/data/wiki_dumps/"
    MONGO_PORT: int = 27017
    RSS_URL: str = "https://dumps.wikimedia.org/plwiki/latest/plwiki-latest-pages-articles-multistream-index.txt.bz2-rss.xml"

    @property
    def mongodb_uri(self) -> str:
        return f"mongodb://{self.MONGO_INITDB_ROOT_USERNAME}:{self.MONGO_INITDB_ROOT_PASSWORD}@mongodb:{self.MONGO_PORT}/?authSource=admin"  # log as admin

    # Load envs from .env file, get only relevant variables, variables are case sensitive
    model_config = SettingsConfigDict(
        env_file=".env", extra="ignore", case_sensitive=True
    )
