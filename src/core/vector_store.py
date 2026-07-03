from pathlib import Path
from typing import List
import chromadb
from src.logger import get_logger
from src.config import VectorStoreConfig
from src.models.schemas import Document


logger = get_logger(__name__)

class ChromaVectorStore:

    def __init__(self,path: str):
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.create_collection(VectorStoreConfig.collection_name)

    def check_connected(self) -> bool:
        return self.client.heartbeat()
    
    def add_vectors(self, vectors: List[tuple[Document,List[float]]]):
        pass

    
        
        

