from pathlib import Path
from typing import List
import chromadb
from src.exceptions import VectorStoreError
from src.logger import get_logger
from src.config import RagConfig
from src.models.schemas import Document


logger = get_logger(__name__)

class ChromaVectorStore:

    def __init__(self, config: RagConfig):
        logger.info("Initiating Chroma DB and creating collection")
        self.config = config
        
        try:
            db_path = Path(self.config.vector_store.db_path)
            if not db_path.is_absolute():
                db_path = Path(__file__).resolve().parent.parent.parent / db_path
            
            db_path.mkdir(parents=True, exist_ok=True)
        
        except Exception as e:
            logger.error(f"Unable to create path for vector db: {e}")
            raise VectorStoreError(f"Failed to create db path: {e}") from e
        
        try:
            self.client = chromadb.PersistentClient(path=str(db_path))
            self.collection = self.client.get_or_create_collection(
                name=self.config.vector_store.collection_name
            )
        except Exception as e:
            logger.error(f"Failed to connect to chroma or create collection '{self.config.vector_store.collection_name}': {e}")
            raise VectorStoreError(f"Initialization failed: {e}") from e
        

    def check_connected(self) -> bool:
        return self.client.heartbeat()
    
    def add_vectors(self, embeddings: List[tuple[Document, List[float]]]):
        docs, vectors = zip(*embeddings)

        self.collection.upsert(
            ids=[doc.id for doc in docs],
            embeddings=list(vectors),
            documents=[doc.content for doc in docs],
            metadatas=[{
                "source": doc.source,
                "page_number": doc.page_number,
                "chunk_index": doc.chunk_index
            } for doc in docs]
        )
        logger.info(f"Upserted {len(docs)} vectors to collection")
    
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[tuple[Document, float]]:
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=['distances', 'documents', 'metadatas']
        )
        logger.info(f"Chroma returned: distances={results['distances']}, ids count={len(results['ids'][0]) if results['ids'] else 0}")
        documents = []
        if results['ids'] and len(results['ids']) > 0:
            logger.info(f"Processing {len(results['ids'][0])} documents")
            for i, doc_id in enumerate(results['ids'][0]):
                metadata = results['metadatas'][0][i]
                content = results['documents'][0][i]
                distance = results['distances'][0][i]
                # logger.info(f"Creating Document: id={doc_id}, distance={distance}")
                doc = Document(
                    id=doc_id,
                    content=content,
                    source=metadata['source'],
                    page_number=metadata['page_number'],
                    chunk_index=metadata['chunk_index']
                )
                documents.append((doc, distance))
        
        return documents