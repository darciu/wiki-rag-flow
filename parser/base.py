
from abc import ABCMeta, abstractmethod
from typing import List, Tuple
from parser.entities import NEREntities

class NLP(metaclass=ABCMeta):
    """Abstract interface for NLP toolkit"""

    @abstractmethod
    def extract_ner_entities(self, text: str) -> NEREntities:
        """Extract Named Entity Recognition entities from text."""
        pass

    @abstractmethod
    def extract_keywords(self, text: List[str]) -> List[Tuple]:
        """Extract keywords from text."""
        pass

    @abstractmethod
    def lemmatize(self, text_data: List[str], batch_size: int) -> List[str]:
        """Lemmatize names and sunrnames in list"""
        pass

    @abstractmethod
    def texts_readability_fog(self, texts: list[str], batch_size: int) -> list[float]:
        """
        Gunning FOG Index (text readability) for list of texts calculated using spaCy.
        """
        pass