# Este arquivo marca o diretório core como um módulo Python
from .base_rag import BaseRAG, RAGConfig, RAGDocument, RAGResponse
from .faiss_rag import FAISSRAGSystem, FAISSDocumentStore

__all__ = [
    'BaseRAG',
    'RAGConfig',
    'RAGDocument',
    'RAGResponse',
    'FAISSRAGSystem',
    'FAISSDocumentStore'
] 