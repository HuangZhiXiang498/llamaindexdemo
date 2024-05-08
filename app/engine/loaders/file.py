import os
from llama_parse import LlamaParse
from pydantic import BaseModel, validator
from PyPDF2 import PdfReader
import fitz
from llama_index.core.readers.base import BaseReader
from typing import Any, Dict, List, Optional
from fsspec import AbstractFileSystem
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

from llama_index.core.readers.base import BaseReader
from llama_index.core.readers.file.base import get_default_fs, is_default_fs
from llama_index.core.schema import Document

class FileLoaderConfig(BaseModel):
    data_dir: str = "data"
    use_llama_parse: bool = False

    @validator("data_dir")
    def data_dir_must_exist(cls, v):
        if not os.path.isdir(v):
            raise ValueError(f"Directory '{v}' does not exist")
        return v

class PyMuPDFReader(BaseReader):
    """PDF parser using fitz (pymupdf)."""

    def __init__(self, return_full_document: Optional[bool] = False) -> None:
        """
        Initialize PDFReader.
        """
        self.return_full_document = return_full_document

    def load_data(
        self,
        file: Path,
        extra_info: Optional[Dict] = None,
        fs: Optional[AbstractFileSystem] = None,
    ) -> List[Document]:
        """Parse file using fitz."""
        if not isinstance(file, Path):
            file = Path(file)

        fs = fs or get_default_fs()
        with fs.open(file, "rb") as fp:
            pdf_bytes = fp.read()
            # 使用fitz打开PDF文件
            pdf = fitz.open(stream=BytesIO(pdf_bytes))

            # 获取PDF文档的页数
            num_pages = pdf.page_count

            docs = []

            # 如果需要返回整个文档作为一个Document
            if self.return_full_document:
                text = ""
                metadata = {"file_name": file.name}

                # 遍历每一页
                for page_num in range(num_pages):
                    # 获取页面文本
                    page = pdf.load_page(page_num)
                    page_text =  page.get_textpage().extractText()
                    text += page_text

                docs.append(Document(text=text, metadata=metadata))

            # 如果需要将每一页作为一个单独的Document返回
            else:
                for page_num in range(num_pages):
                    # 获取页面文本
                    page = pdf.load_page(page_num)
                    page_text = page.get_textpage().extractText()
                    page_label = fitz.utils.get_label(page)

                    metadata = {"page_label": page_label, "file_name": file.name}
                    if extra_info is not None:
                        metadata.update(extra_info)

                    docs.append(Document(text=page_text, metadata=metadata))

            # 关闭PDF文档
            pdf.close()

            return docs

def llama_parse_parser():
    if os.getenv("LLAMA_CLOUD_API_KEY") is None:
        raise ValueError(
            "LLAMA_CLOUD_API_KEY environment variable is not set. "
            "Please set it in .env file or in your shell environment then run again!"
        )
    parser = LlamaParse(result_type="markdown", verbose=True, language="en")
    return parser

def get_file_documents(config: FileLoaderConfig):
    from llama_index.core.readers import SimpleDirectoryReader

    reader = SimpleDirectoryReader(
        config.data_dir,
        recursive=True,
        file_extractor={
            ".pdf": PyMuPDFReader(),
        },
    )
    if config.use_llama_parse:
        parser = llama_parse_parser()
        reader.file_extractor = {".pdf": parser}
    return reader.load_data()
