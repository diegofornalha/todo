import pytest
import time
from langgraph_agente_vendedor.core.redis_cache import RedisCache

@pytest.fixture
def redis_cache():
    """Fixture que fornece uma instância do RedisCache."""
    cache = RedisCache(
        host='localhost',
        port=6379,
        password='redis123',  # Senha definida no redis.conf
        prefix='test:',
        ttl=1  # 1 segundo para testar expiração
    )
    # Limpa o cache antes dos testes
    cache.clear()
    return cache

@pytest.fixture
def invalid_redis_cache():
    """Fixture que fornece uma instância do RedisCache com configuração inválida."""
    return RedisCache(
        host='localhost',
        port=9999,  # Porta inválida
        password='redis123',
        prefix='test:',
        ttl=1
    )

def test_connection(redis_cache):
    """Testa a conexão com o Redis."""
    assert redis_cache.ping() is True

def test_set_get(redis_cache):
    """Testa operações básicas de set/get."""
    # Testa armazenamento e recuperação
    assert redis_cache.set('key1', 'value1') is True
    assert redis_cache.get('key1') == 'value1'
    
    # Testa tipos complexos
    data = {
        'string': 'test',
        'number': 42,
        'list': [1, 2, 3],
        'dict': {'a': 1, 'b': 2}
    }
    assert redis_cache.set('key2', data) is True
    assert redis_cache.get('key2') == data

def test_delete(redis_cache):
    """Testa deleção de chaves."""
    # Armazena e deleta
    redis_cache.set('key1', 'value1')
    assert redis_cache.delete('key1') is True
    assert redis_cache.get('key1') is None
    
    # Tenta deletar chave inexistente
    assert redis_cache.delete('nonexistent') is True

def test_clear(redis_cache):
    """Testa limpeza do cache."""
    # Armazena múltiplos itens
    redis_cache.set('key1', 'value1')
    redis_cache.set('key2', 'value2')
    
    # Limpa o cache
    assert redis_cache.clear() is True
    
    # Verifica se os itens foram removidos
    assert redis_cache.get('key1') is None
    assert redis_cache.get('key2') is None

def test_ttl(redis_cache):
    """Testa expiração de chaves."""
    # Armazena um item
    redis_cache.set('key1', 'value1')
    
    # Verifica se está presente
    assert redis_cache.get('key1') == 'value1'
    
    # Espera a expiração
    time.sleep(1.1)  # Espera um pouco mais que o TTL
    
    # Verifica se expirou
    assert redis_cache.get('key1') is None

def test_prefix(redis_cache):
    """Testa o uso do prefixo nas chaves."""
    # Armazena um item
    redis_cache.set('key1', 'value1')
    
    # Verifica se a chave real tem o prefixo
    assert redis_cache.redis.exists('test:key1') == 1  # Redis.exists() retorna 1 para chave existente

def test_error_handling(invalid_redis_cache):
    """Testa o tratamento de erros."""
    # Verifica se as operações falham graciosamente
    assert invalid_redis_cache.set('key1', 'value1') is False
    assert invalid_redis_cache.get('key1') is None
    assert invalid_redis_cache.delete('key1') is False
    assert invalid_redis_cache.clear() is False
    assert invalid_redis_cache.ping() is False 