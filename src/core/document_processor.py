from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict, Any
import os
import logging
from ..utils.logging_config import setup_logger

logger = setup_logger('document_processor', 'logs/document_processor.log')

class DocumentProcessor:
    def __init__(
        self,
        file_path: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """
        Inicializa o processador de documentos.
        
        Args:
            file_path: Caminho para o arquivo
            chunk_size: Tamanho dos chunks para divisão
            chunk_overlap: Sobreposição entre chunks
        """
        self.file_path = file_path
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        logger.info(f"DocumentProcessor inicializado para: {file_path}")
        
    def process_pdf(self) -> List[Dict[str, Any]]:
        """
        Processa um arquivo PDF.
        
        Returns:
            Lista de documentos processados
        """
        logger.info(f"Processando PDF: {self.file_path}")
        
        if not os.path.exists(self.file_path):
            error_msg = f"Arquivo não encontrado: {self.file_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        try:
            # Carrega o PDF
            loader = PyPDFLoader(self.file_path)
            documents = loader.load()
            logger.info(f"Carregadas {len(documents)} páginas do PDF")
            
            # Divide em chunks
            splits = self.text_splitter.split_documents(documents)
            logger.info(f"Documento dividido em {len(splits)} chunks")
            
            # Prepara os documentos processados
            processed_docs = []
            for i, doc in enumerate(splits):
                processed_doc = {
                    "content": doc.page_content,
                    "metadata": {
                        **doc.metadata,
                        "chunk_id": i
                    }
                }
                processed_docs.append(processed_doc)
            
            logger.info("Processamento concluído com sucesso")
            return processed_docs
            
        except Exception as e:
            logger.error(f"Erro ao processar o documento: {str(e)}")
            raise 