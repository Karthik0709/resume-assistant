from pathlib import Path
from src.core.base import BaseDocumentLoader
from langchain_docling.loader import DoclingLoader
from src.config import RagConfig
from src.models.schemas import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.logger import get_logger

logger = get_logger(__name__)


class DoclingDocumentLoader(BaseDocumentLoader):

    def __init__(self, file_path: str, config: RagConfig):
        self.file_path = file_path
        self.config = config

    def load(self) -> list[Document]:
        path = Path(self.file_path)

        if not path.is_absolute():
            path = Path(__file__).resolve().parent.parent.parent / "data" / self.file_path
        
        loader = DoclingLoader(file_path=path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunking.chunk_size,
            chunk_overlap=self.config.chunking.chunk_overlap
        )
        splitted_docs = splitter.split_documents(docs)

        converted = [Document(
            content= doc.page_content,
            source = str(doc.metadata.get('source')),
            page_number = doc.metadata.get('page',0),
            chunk_index = i
        )
        for i, doc in enumerate(splitted_docs)
        ]
        logger.info(f"Created {len(converted)} chunks")
        return converted
    