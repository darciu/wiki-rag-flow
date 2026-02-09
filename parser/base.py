
from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Any, List, Literal, Union, overload
from parser.entities import NEREntities

class NLP(metaclass=ABCMeta):
    """Abstract interface for NLP toolkit"""

    @abstractmethod
    def extract_ner_entities(self, text: str) -> NEREntities:
        """Extract Named Entity Recognition entities from text."""
        pass

    @overload
    @abstractmethod
    def lemmatize(self, text_data: str) -> str: ...

    @overload
    @abstractmethod
    def lemmatize(self, text_data: List[str]) -> List[str]: ...

    @abstractmethod
    def lemmatize(self, text_data: Union[str, List[str]]) -> Union[str, List[str]]:
        """Lemmatize text or list of texts."""
        pass