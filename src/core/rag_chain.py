from pathlib import Path
from typing import List

import requests

from src.config import RagConfig
from src.core.document_loader import DoclingDocumentLoader
from src.core.embedding_service import EmbeddingService
from src.core.vector_store import ChromaVectorStore
from src.logger import get_logger
from src.models.schemas import ChatResponse, Document
from src.secrets import SecretsManager
from groq import Groq

logger = get_logger(__name__)


class RAGChain:
    """Orchestrates chunking → embedding → vector store storage"""

    def __init__(self, config: RagConfig):
        self.config = config
        self.embedder = EmbeddingService(config)
        self.vector_store = ChromaVectorStore(config)

    def initialize(self) -> int:
        logger.info("Starting RAG initialization from /data folder")
        
        data_dir = Path(__file__).resolve().parent.parent.parent / "data"
        
        if not data_dir.exists():
            logger.error(f"Data directory not found: {data_dir}")
            raise FileNotFoundError(f"Data directory missing: {data_dir}")
        
        pdf_files = list(data_dir.glob("*.pdf"))
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {data_dir}")
            return 0
        
        logger.info(f"Found {len(pdf_files)} PDF file(s)")
        
        all_documents = []
        
        for pdf_file in pdf_files:
            logger.info(f"Chunking {pdf_file.name}")
            loader = DoclingDocumentLoader(str(pdf_file), self.config)
            documents = loader.load()
            all_documents.extend(documents)
        
        logger.info(f"Total chunks: {len(all_documents)}")
        embeddings = self.embedder.embed_batch(all_documents)
        self.vector_store.add_vectors(embeddings)
        logger.info(f"✓ Stored {len(all_documents)} chunks")
        
        return len(all_documents)

    def answer(self, query: str) -> ChatResponse:
        logger.info(f"Query: {query[:50]}...")
        
        query_embedding = self.embedder.embed_text(query)
        retrieved_docs = self.vector_store.search(query_embedding, top_k=5)
        logger.info(f"answer() received {len(retrieved_docs)} docs from search()")
        MIN_RELEVANCE = 1.5
        relevant_docs = [(doc, dist) for doc, dist in retrieved_docs if dist < MIN_RELEVANCE]

        
        if not relevant_docs:
            return ChatResponse(
                message="No relevant information found.",
                references=[]
            )
        
        context = self._format_context(relevant_docs)
        response_text = self._call_mistral(query, context)
        reference_docs = [doc for doc, _ in relevant_docs]
        
        return ChatResponse(message=response_text, references=reference_docs)

    def _format_context(self, retrieved_docs: List[tuple[Document, float]]) -> str:
        context_parts = []
        for doc, _ in retrieved_docs:
            context_parts.append(f"[{doc.source}, Page {doc.page_number}]\n{doc.content}")
        return "\n\n---\n\n".join(context_parts)

    def _call_mistral(self, query: str, context: str) -> str:
        #api_key = SecretsManager.get_huggingface_api_key()
        api_key = SecretsManager.get_groq_api_key()
        # Now use api_key to call HF API
        logger.info("Calling Qwn with Groq API")
        # url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
        
        # headers = {"Authorization": f"Bearer {api_key}"}

        client = Groq(api_key=api_key)
        
        system_prompt = """You are a helpful assistant answering questions about a professional resume.
Answer based only on the provided resume context. Keep responses concise and natural."""
        
        full_prompt = f"""{system_prompt}

        Resume Context:
        {context}

        ---

        Question: {query}

        Answer the question based on the resume context above."""
        
        payload = {
            "inputs": full_prompt,
            "parameters": {
                "temperature": 0.7,
                "max_new_tokens": 500
            }
        }
        
        try:
            message = client.chat.completions.create(
            model="qwen/qwen3-32b",
            messages=[
                {"role": "user", "content": full_prompt}
            ],
            temperature=0.7,
            max_tokens=500
            )
            return message.choices[0].message.content
    
        except Exception as e:
            logger.error(f"Groq API request failed: {e}")
            raise
        #     logger.info("Calling HuggingFace Inference API")
        #     response = requests.post(url, headers=headers, json=payload, timeout=60)
        #     response.raise_for_status()
            
        #     result = response.json()
            
        #     if isinstance(result, list) and len(result) > 0:
        #         return result[0].get("generated_text", "No response generated")
        #     else:
        #         logger.error(f"Unexpected API response format: {result}")
        #         return "Error: Invalid response from API"
        
        # except requests.exceptions.Timeout:
        #     logger.error("HuggingFace API request timed out")
        #     raise
        # except requests.exceptions.RequestException as e:
        #     logger.error(f"HuggingFace API request failed: {e}")
        #     raise
        # except Exception as e:
        #     logger.error(f"Error processing HuggingFace response: {e}")
        #     raise

    # def __init__(self, config: RagConfig):
    #     self.config = config
    #     self.loader = DoclingDocumentLoader("Karthikeyan_Premkumar_Resume-1.pdf", config)
    #     self.embedder = EmbeddingService(config)
    #     self.vector_store = ChromaVectorStore(config)
    
    # def initialize(self, file_path: str):
    #     # One-time: chunk → embed → store
    #     documents = self.loader.load(file_path)
    #     embeddings = self.embedder.embed_batch(documents)
    #     self.vector_store.add_vectors(embeddings)

    # def build(self, file_path: str) -> int:
    #     """
    #     Chunk a document, embed it, and store in vector DB.
    #     Returns number of chunks stored.
    #     """
    #     logger.info(f"Starting RAG pipeline for {file_path}")
        
    #     try:
    #         # Step 1: Chunk
    #         documents = self.loader.load()
    #         logger.info(f"Chunked document into {len(documents)} chunks")
            
    #         # Step 2: Embed
    #         embeddings = self.embedder.embed_batch(documents)
    #         logger.info(f"Generated {len(embeddings)} embeddings")
            
    #         # Step 3: Store
    #         self.vector_store.add_vectors(embeddings)
    #         logger.info(f"Stored {len(embeddings)} vectors in Chroma")
            
    #         return len(documents)
        
    #     except Exception as e:
    #         logger.error(f"RAG pipeline failed: {e}")
    #         raise