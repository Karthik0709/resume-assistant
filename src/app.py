# src/app.py
import os
from pathlib import Path
from dotenv import load_dotenv
from src.config import RagConfig
from src.core.rag_chain import RAGChain
from src.logger import get_logger

# Load .env.local before anything else
logger = get_logger(__name__)

config = RagConfig.load("config.json")
rag_chain = RAGChain(config)

# Auto-initialize on startup if vector store is empty
if rag_chain.vector_store.collection.count() == 0:
    logger.info("Vector store empty, initializing...")
    rag_chain.initialize()
else:
    logger.info("Vector store already initialized")