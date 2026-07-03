from src.config import RagConfig
from src.core.document_loader import DoclingDocumentLoader
from src.core.embedding_service import EmbeddingService

config = RagConfig.load()
loader = DoclingDocumentLoader("Karthikeyan_Premkumar_Resume-1.pdf", config)
chunks = loader.load()
print(f"Loaded {len(chunks)} chunks")

embedder = EmbeddingService(config)
vectors = embedder.embed_batch(chunks)
print(f"Generated {len(vectors)} embeddings")
print(f"First vector dimension: {len(vectors[0])}")