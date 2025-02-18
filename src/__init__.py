# Este arquivo marca o diretório src como um pacote Python
from .utils.logging_config import setup_logger
from .core.document_processor import DocumentProcessor
from .core.qa_chain import QAChain
from .models.groq_handler import GroqHandler

__all__ = [
    'setup_logger',
    'DocumentProcessor',
    'QAChain',
    'GroqHandler'
] 