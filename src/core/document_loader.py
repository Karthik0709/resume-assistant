from pathlib import Path
from src.core.base import BaseDocumentLoader
from langchain_docling.loader import DoclingLoader
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
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
        
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = False
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        loader = DoclingLoader(file_path=path, converter=converter)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunking.chunk_size,
            chunk_overlap=self.config.chunking.chunk_overlap
        )
        splitted_docs = splitter.split_documents(docs)

        converted = [Document(
            id=f"{path.name}#{doc.metadata.get('page', 0)}#{i}",
            content= doc.page_content,
            source = str(doc.metadata.get('source')),
            page_number = doc.metadata.get('page',0),
            chunk_index = i
        )
        for i, doc in enumerate(splitted_docs)
        ]
        logger.info(f"Created {len(converted)} chunks")
        return converted
    