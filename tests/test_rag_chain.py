# tests/test_vector_store_integrity.py
import pytest
from src.config import RagConfig
from src.core.rag_chain import RAGChain



def test_rag_chain_build():
    """Verify data stored in Chroma is correct"""
    config = RagConfig.load("config.json")
    
    # Initialize RAGChain
    rag = RAGChain(config)
    
    # Initialize vector store if empty
    if rag.vector_store.collection.count() == 0:
        print("Initializing vector store...")
        num_chunks = rag.initialize()
        print(f"✓ Initialized with {num_chunks} chunks\n")
    else:
        print(f"✓ Vector store already has {rag.vector_store.collection.count()} chunks\n")
    
    print(f"Collection count: {rag.vector_store.collection.count()}")

    # Try a search directly
    test_embedding = rag.embedder.embed_text("Python")
    raw_results = rag.vector_store.collection.query(
        query_embeddings=[test_embedding],
        n_results=5,
        include=['distances', 'documents', 'metadatas']
    )

    print(f"Raw search results: {raw_results}")
    # Test queries
    test_queries = [
        "What's Karthikeyan's experience with Python?",
        "Has he built any scalable systems?",
        "What's his education background?"
    ]
    
    for query in test_queries:
        print(f"Query: {query}")
        try:
            response = rag.answer(query)
            print(f"Response: {response.message}\n")
            print(f"References: {len(response.references)} chunks\n")
            print("-" * 80 + "\n")
        except Exception as e:
            print(f"Error: {e}\n")

if __name__ == "__main__":
    test_rag_chain_build()