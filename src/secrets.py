# src/secrets.py
import os
from src.logger import get_logger

logger = get_logger(__name__)


class SecretsManager:
    @staticmethod
    def get_huggingface_api_key() -> str:
        key = os.environ.get("HUGGINGFACE_API_KEY")
        if not key:
            logger.error("HUGGINGFACE_API_KEY not set in environment")
            raise ValueError("Missing HUGGINGFACE_API_KEY environment variable")
        return key
    
    @staticmethod
    def get_groq_api_key() -> str:
        key = os.environ.get("GROQ_API_KEY")
        if not key:
            logger.error("GROQ_API_KEY not set in environment")
            raise ValueError("Missing GROQ_API_KEY environment variable")
        return key
