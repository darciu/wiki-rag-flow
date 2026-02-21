from abc import ABCMeta, abstractmethod

from parser.entities import NEREntities


class NLP(metaclass=ABCMeta):
    """Abstract interface for NLP toolkit"""

    @abstractmethod
    def extract_ner_entities(self, texts: list[str]) -> list[NEREntities]:
        """Extract Named Entity Recognition entities from text."""
        pass

    @abstractmethod
    def extract_keywords(self, text: list[str]) -> list[tuple]:
        """Extract keywords from text."""
        pass

    @abstractmethod
    def lemmatize(self, text_data: list[str], batch_size: int) -> list[str]:
        """Lemmatize names and sunrnames in list"""
        pass

    @abstractmethod
    def texts_readability_fog(self, texts: list[str], batch_size: int) -> list[float]:
        """
        Gunning FOG Index (text readability) for list of texts calculated using spaCy.
        """
        pass

    @abstractmethod
    def chunk_texts(self, texts: list[str], max_tokens) -> list[list[str]]:
        """Partition input texts into semantic or logical chunks"""
        pass
