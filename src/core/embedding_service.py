from typing import List
from src.config import RagConfig
from langchain_huggingface import HuggingFaceEmbeddings
from src.exceptions import EmbeddingError
from src.logger import get_logger
from src.models.schemas import Document

logger = get_logger(__name__)

class EmbeddingService:

    def __init__(self,config: RagConfig):
        self.config = config
        try:
            self.model = HuggingFaceEmbeddings(model_name=self.config.embedding.model_name)
        except Exception as e:
            logger.error(f"Failed to load embedding model '{self.config.embedding.model_name}': {str(e)}")
            raise EmbeddingError(f"Initialization failed: {e}") from e
            

    def embed_text(self,text: str) -> List[float]:
        logger.info(f"Generating embedding for text snippet (Length: {len(text)} characters).")

        try:
            return self.model.embed_query(text)
        
        except Exception as e:
            logger.error(f"Embedding text failed with exception {e}")
            raise EmbeddingError(f"Failed to embed text: {e}") from e
    
    def embed_batch(self,documents: List[Document]) -> List[List[float]]:
        logger.info(f"Generating batch embeddings for {len(documents)} document(s).")

        try:
            docs_to_embed = [docs.content for docs in documents]
            return self.model.embed_documents(docs_to_embed)
        
        except Exception as e:
            logger.error(f"Batch embedding failed: {e}")
            raise EmbeddingError(f"Failed to embed text: {e}") from e