from typing import Literal

from parser.base import NLP
from parser.entities import NEREntities
from parser.nlp.chunking import LangchainSplitterClient, StatisticalChunkerClient
from parser.nlp.keywords import KeyBERTKeywordsClient, VLT5KeywordsClient
from parser.nlp.ner import (
    HerbertNERClient,
    StanzaNERClient,
)
from parser.nlp.spacy import SpacyUtils

NERModelName = Literal["herbert", "stanza"]
KeywordsModelName = Literal["vlt5", "keybert"]
ChunkingModelName = Literal["langchain", "statistical_chunker"]


class NLPToolkit(NLP):
    """
    NLP toolkit.

    Args:
        ner_model_name: The name of the NER model to use ("herbert" or "stanza").
    """

    ner_client: HerbertNERClient | StanzaNERClient
    keywords_client: KeyBERTKeywordsClient | VLT5KeywordsClient
    chunking_client: LangchainSplitterClient | StatisticalChunkerClient

    def __init__(
        self,
        ner_model_name: NERModelName = "herbert",
        keywords_model_name: KeywordsModelName = "keybert",
        chunking_model_name: ChunkingModelName = "langchain",
    ):
        if ner_model_name == "herbert":
            self.ner_client = HerbertNERClient()
        elif ner_model_name == "stanza":
            self.ner_client = StanzaNERClient()

        if keywords_model_name == "keybert":
            self.keywords_client = KeyBERTKeywordsClient()
        elif keywords_model_name == "vlt5":
            self.keywords_client = VLT5KeywordsClient()

        if chunking_model_name == "langchain":
            self.chunking_client = LangchainSplitterClient()
        elif chunking_model_name == "statistical_chunker":
            self.chunking_client = StatisticalChunkerClient()

        self._spacy_utils = SpacyUtils()

    def extract_ner_entities(self, texts: list[str]) -> list[NEREntities]:
        """Uses the initialized NER model to extract entities from text."""
        return self.ner_client.parse_entities(texts)

    def extract_keywords(self, texts: list[str]) -> list[tuple]:
        """Uses the initialized model to extract keywords from text."""
        return self.keywords_client.extract_keywords(texts)

    def lemmatize(self, names: list[str], batch_size: int = 512) -> list[str]:
        """
        Lemmatizes input list of texts using spaCy
        """
        if isinstance(names, list):
            return self._spacy_utils.lemmatize_names(names, batch_size=batch_size)

        raise TypeError("names must be a list of strings")

    def texts_readability_fog(
        self, texts: list[str], batch_size: int = 100
    ) -> list[float]:
        """
        Gunning FOG Index (text readability) for list of texts calculated using spaCy
        """
        if isinstance(texts, list):
            return self._spacy_utils.texts_readability_fog(texts, batch_size=batch_size)

        raise TypeError("texts must be a list of strings")

    def chunk_texts(self, texts: list[str], max_tokens) -> list[list[str]]:
        """Partition input texts into semantic or logical chunks"""

        return self.chunking_client.chunk_texts(texts, max_tokens)
