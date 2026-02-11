from typing import List, Literal, Union, Tuple

from parser.base import NLP
from parser.nlp.ner import (
    HerbertNERClient,
    StanzaNERClient,
)
from parser.entities import NEREntities
from parser.nlp.spacy import SpacyUtils
from parser.nlp.keywords import VLT5KeywordsClient, KeyBERTKeywordsClient
from parser.nlp.chunking import LangchainSplitterClient, StatisticalChunkerClient

NERModelName = Literal["herbert", "stanza"]
KeywordsModelName = Literal["vlt5", "keybert"]
ChunkingModelName = Literal["langchain","statistical_chunker"]


class NLPToolkit(NLP):
    """
    NLP toolkit.

    Args:
        ner_model_name: The name of the NER model to use ("herbert" or "stanza").
    """

    ner_client: Union[HerbertNERClient, StanzaNERClient]

    def __init__(self, ner_model_name: NERModelName = "stanza", keywords_model_name: KeywordsModelName = 'keybert', chunking_model_name: ChunkingModelName = 'langchain'):
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

    def extract_ner_entities(self, text: str) -> NEREntities:
        """Uses the initialized NER model to extract entities from text."""
        return self.ner_client.parse_entities(text)
    
    def extract_keywords(self, texts: List[str]) -> List[Tuple]:
        """Uses the initialized model to extract keywords from text."""
        return self.keywords_client.extract_keywords(texts)


    def lemmatize(self, names: List[str], batch_size: int = 512) -> List[str]:
        """
        Lemmatizes input list of texts using spaCy
        """
        if isinstance(names, list):
            return self._spacy_utils.lemmatize_names(names, batch_size=batch_size)

        raise TypeError("names must be a list of strings")
    
    def texts_readability_fog(self, texts: list[str], batch_size: int = 100) -> list[float]:
        """
        Gunning FOG Index (text readability) for list of texts calculated using spaCy
        """
        if isinstance(texts, list):
            return self._spacy_utils.texts_readability_fog(texts, batch_size=batch_size)

        raise TypeError("texts must be a list of strings")
    
    def chunk_texts(self, texts: list[str], max_tokens) -> List[list[str]]:
        """Partition input texts into semantic or logical chunks"""

        return self.chunking_client.chunk_texts(texts, max_tokens)

    

    
