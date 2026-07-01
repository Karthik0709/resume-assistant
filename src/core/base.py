from abc import ABC, abstractmethod
from typing import List
from src.models.schemas import Document

class BaseDocumentLoader(ABC):
    """Abstract base for document loaders"""
    
    @abstractmethod
    def load(self) -> List[Document]:
        """Load and chunk documents. Must be implemented by subclasses."""
        pass