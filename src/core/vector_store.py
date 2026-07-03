from pathlib import Path
from typing import List
import uuid
import chromadb
from src.exceptions import VectorStoreError
from src.logger import get_logger
from src.config import VectorStoreConfig
from src.models.schemas import Document


logger = get_logger(__name__)

class ChromaVectorStore:

    def __init__(self,config: RagConfig):
        logger.info("Initiating Chroma DB and creating collection")
        self.config = config
        try:
            custom_path = Path(__file__).resolve().parent.parent.parent / config.lstrip("./")
            custom_path.mkdir(parents=True, exist_ok=True)
        
        except Exception as e:
            logger.error("Unable to create path vector db")
        
        try:
            
            self.client = chromadb.PersistentClient(path=str(custom_path))
            self.collection = self.client.get_or_create_collection(VectorStoreConfig.collection_name)
        except Exception as e:
            logger.error(f"Failed to connect to chroma or Collection not present {VectorStoreConfig.collection_name}")
            raise VectorStoreError() from e
        

    def check_connected(self) -> bool:
        return self.client.heartbeat()
    
    def add_vectors(self, embeddings: List[tuple[Document,List[float]]]):
        docs, vectors = zip(*embeddings)

        self.collection.upsert(
            ids=[doc.id for doc in docs],
            embeddings=list(vectors),
            documents = [doc.content for doc in docs],
            metadatas=[{
                "source":doc.source,
                "page_number": doc.page_number,
                "chunk_index": doc.chunk_index
            } for doc in docs]
        )
        logger.info(f"Upserted {len(docs)} vectors to collection")


    
        
        

