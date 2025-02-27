from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import json

@dataclass
class RAGConfig:
    """Configuração para o sistema RAG."""
    embeddings_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_documents: int = 3
    similarity_threshold: float = 0.7
    
    # Configurações de cache
    cache_enabled: bool = True
    cache_type: str = "redis"  # "redis" ou "file"
    cache_dir: str = "cache/rag"  # Para cache em arquivo
    
    # Configurações do Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: Optional[str] = None
    redis_db: int = 0
    redis_prefix: str = "rag:"
    redis_ttl: int = 3600  # 1 hora

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
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte a resposta para dicionário."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Converte a resposta para JSON."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RAGResponse':
        """Cria uma instância a partir de um dicionário."""
        return cls(**data)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'RAGResponse':
        """Cria uma instância a partir de uma string JSON."""
        return cls.from_dict(json.loads(json_str))

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