from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class RAGConfig:
    """Configuração para o sistema RAG."""
    embeddings_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_documents: int = 3
    similarity_threshold: float = 0.7
    cache_enabled: bool = True
    cache_dir: str = "cache/rag"

@dataclass
class RAGDocument:
    """Representa um documento no sistema RAG."""
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    chunks: Optional[List[Dict[str, Any]]] = None

@dataclass
class RAGResponse:
    """Resposta do sistema RAG."""
    question: str
    answer: str
    sources: List[str]
    metadata: Dict[str, Any]
    confidence: float
    processing_time: float
    status: str = "success"
    error: Optional[str] = None

class BaseRAG(ABC):
    """Classe base abstrata para sistemas RAG."""
    
    @abstractmethod
    def initialize(self, config: RAGConfig) -> None:
        """Inicializa o sistema RAG."""
        pass
        
    @abstractmethod
    def add_documents(self, documents: List[Dict[str, str]]) -> None:
        """Adiciona documentos ao sistema."""
        pass
        
    @abstractmethod
    def query(
        self,
        question: str,
        k: int = 3,
        include_sources: bool = True
    ) -> RAGResponse:
        """Processa uma pergunta e retorna a resposta."""
        pass
        
    @abstractmethod
    def save(self, path: str) -> None:
        """Salva o estado do sistema."""
        pass
        
    @abstractmethod
    def load(self, path: str) -> None:
        """Carrega o estado do sistema."""
        pass
        
    @abstractmethod
    def clear(self) -> None:
        """Limpa todos os documentos e cache."""
        pass
        
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas do sistema."""
        pass 