from typing import List, Literal, Union, Tuple

from parser.base import NLP
from parser.nlp.ner import (
    HerbertNERClient,
    StanzaNERClient,
)
from parser.entities import NEREntities
from parser.nlp.spacy import SpacyUtils
from parser.nlp.keywords import VLT5KeywordsClient, KeyBERTKeywordsClient

NERModelName = Literal["herbert", "stanza"]
KeywordsModelName = Literal["vlt5", "keybert"]


class NLPToolkit(NLP):
    """
    NLP toolkit.

    Args:
        ner_model_name: The name of the NER model to use ("herbert" or "stanza").
    """

    ner_client: Union[HerbertNERClient, StanzaNERClient]

    def __init__(self, ner_model_name: NERModelName = "stanza", keywords_model_name: KeywordsModelName = 'keybert'):
        if ner_model_name == "herbert":
            self.ner_client = HerbertNERClient()
        elif ner_model_name == "stanza":
            self.ner_client = StanzaNERClient()

        if keywords_model_name == "keybert":
            self.keywords_client = KeyBERTKeywordsClient()
        elif keywords_model_name == "vlt5":
            self.keywords_client = VLT5KeywordsClient()

        self.spacy_utils = SpacyUtils()

    def extract_ner_entities(self, text: str) -> NEREntities:
        """Uses the initialized NER model to extract entities from text."""
        return self.ner_client.parse_entities(text)
    
    def extract_keywords(self, texts: List[str]) -> List[Tuple]:
        """Uses the initialized model to extract keywords from text."""
        return self.keywords_client.extract_keywords(texts)


    def lemmatize(self, names: List[str], batch_size: int = 512) -> List[str]:
        """
        Lemmatizes input list of texts using spaCy.
        """
        if isinstance(names, list):
            return self.spacy_utils.lemmatize_names(names, batch_size=batch_size)

        raise TypeError("names must be a list of strings")
    
    def texts_readability_fog(self, texts: list[str], batch_size: int = 100) -> list[float]:
        """
        Gunning FOG Index (text readability) for list of texts calculated using spaCy.
        """
        if isinstance(texts, list):
            return self.spacy_utils.texts_readability_fog(texts, batch_size=batch_size)

        raise TypeError("texts must be a list of strings")

    

    
