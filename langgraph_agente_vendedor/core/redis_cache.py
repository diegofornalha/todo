from typing import Optional, Any
import json
import redis
from ..utils.logging_config import setup_logger

logger = setup_logger('redis_cache')

class RedisCache:
    """Gerencia o cache usando Redis."""
    
    def __init__(
        self,
        host: str = 'localhost',
        port: int = 6379,
        password: str = None,
        db: int = 0,
        prefix: str = 'rag:',
        ttl: int = 3600  # 1 hora
    ):
        """
        Inicializa o cache Redis.
        
        Args:
            host: Host do Redis
            port: Porta do Redis
            password: Senha do Redis
            db: Número do banco de dados
            prefix: Prefixo para as chaves
            ttl: Tempo de vida dos itens em segundos
        """
        self.redis = redis.Redis(
            host=host,
            port=port,
            password=password,
            db=db,
            decode_responses=True
        )
        self.prefix = prefix
        self.ttl = ttl
        logger.info(f"Cache Redis inicializado em {host}:{port}")
        
    def _get_key(self, key: str) -> str:
        """Gera a chave completa com prefixo."""
        return f"{self.prefix}{key}"
        
    def get(self, key: str) -> Optional[Any]:
        """
        Recupera um item do cache.
        
        Args:
            key: Chave do item
            
        Returns:
            Item do cache ou None se não encontrado
        """
        try:
            data = self.redis.get(self._get_key(key))
            if data:
                logger.info(f"Cache hit para chave: {key}")
                return json.loads(data)
            logger.info(f"Cache miss para chave: {key}")
            return None
        except Exception as e:
            logger.error(f"Erro ao recuperar do cache: {str(e)}")
            return None
            
    def set(self, key: str, value: Any) -> bool:
        """
        Armazena um item no cache.
        
        Args:
            key: Chave do item
            value: Valor a ser armazenado
            
        Returns:
            True se sucesso, False caso contrário
        """
        try:
            full_key = self._get_key(key)
            self.redis.setex(
                full_key,
                self.ttl,
                json.dumps(value)
            )
            logger.info(f"Item armazenado no cache: {key}")
            return True
        except Exception as e:
            logger.error(f"Erro ao armazenar no cache: {str(e)}")
            return False
            
    def delete(self, key: str) -> bool:
        """
        Remove um item do cache.
        
        Args:
            key: Chave do item
            
        Returns:
            True se sucesso, False caso contrário
        """
        try:
            self.redis.delete(self._get_key(key))
            logger.info(f"Item removido do cache: {key}")
            return True
        except Exception as e:
            logger.error(f"Erro ao remover do cache: {str(e)}")
            return False
            
    def clear(self) -> bool:
        """
        Limpa todo o cache com o prefixo definido.
        
        Returns:
            True se sucesso, False caso contrário
        """
        try:
            pattern = f"{self.prefix}*"
            keys = self.redis.keys(pattern)
            if keys:
                self.redis.delete(*keys)
            logger.info("Cache limpo")
            return True
        except Exception as e:
            logger.error(f"Erro ao limpar cache: {str(e)}")
            return False
            
    def ping(self) -> bool:
        """
        Verifica a conexão com o Redis.
        
        Returns:
            True se conectado, False caso contrário
        """
        try:
            return self.redis.ping()
        except Exception as e:
            logger.error(f"Erro ao conectar com Redis: {str(e)}")
            return False 