from dataclasses import dataclass
import json
from pathlib import Path

@dataclass
class EmbeddingConfig:
    """Configuration for embedding model"""
    model_name: str
    embedding_dimension: int

@dataclass
class ChunkingConfig:
    """Configuration for text chunking"""
    chunk_size: int
    chunk_overlap: int
    separator: str

@dataclass
class VectorStoreConfig:
    """Configuration for vector database"""
    db_path: str
    collection_name: str

@dataclass
class RagConfig:
    """Main RAG configuration"""
    embedding: EmbeddingConfig
    chunking: ChunkingConfig
    vector_store: VectorStoreConfig

    @classmethod
    def load(cls, file_path: str = "config.json") -> RagConfig:
        path = Path(file_path)
        
        # If not absolute, assume it's in the project root
        if not path.is_absolute():
            path = Path(__file__).resolve().parent.parent / file_path

        try:
            with open(path, 'r') as f:
                data = json.load(f)

                return cls(
                embedding=EmbeddingConfig(**data["embedding"]),
                chunking=ChunkingConfig(**data["chunking"]),
                vector_store=VectorStoreConfig(**data["vector_store"])
            )
        except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
            print(f"Error loading config: {e}. Using default values.")


if __name__ == "__main__":
    config = RagConfig.load()
    print(config.embedding.model_name)
    print(config.chunking.chunk_size)