import pytest
import time
from unittest.mock import Mock
from langgraph_agente_vendedor.core.redis_cache import RedisCache
from langgraph_agente_vendedor.core.faiss_rag import FAISSRAGSystem
from langgraph_agente_vendedor.core.base_rag import RAGConfig

class MockLLM:
    def invoke(self, messages):
        # Simula respostas baseadas no contexto
        question = messages[-1]["content"].split("Pergunta: ")[-1]
        context = messages[-1]["content"].split("Contexto:\n")[1].split("\n\nPergunta:")[0]
        
        if "qual é meu nome" in question.lower():
            return Mock(content="Oi! Me chamo Eliza, prazer! E você? 😊")
        elif "meu nome é diego" in question.lower():
            return Mock(content="Que legal te conhecer, Diego! Como vai? 🤗")
        elif "como você me chama" in question.lower():
            if "diego" in context.lower():
                return Mock(content="Te chamo de Diego, claro! Prefere algum apelido? 😊")
            else:
                return Mock(content="Oi! Me chamo Eliza! E você, como se chama? 🤗")
        else:
            return Mock(content="Oi! Me chamo Eliza, prazer em te conhecer! Como você se chama? 😊")

@pytest.fixture
def redis_cache():
    """Fixture que fornece uma instância do RedisCache."""
    cache = RedisCache(
        host='localhost',
        port=6379,
        password='redis123',  # Senha definida no redis.conf
        prefix='test:',  # Adicionando prefixo para os testes
        ttl=1  # TTL de 1 segundo para testes
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

@pytest.fixture
def rag_system():
    """Fixture que fornece uma instância do sistema RAG."""
    config = RAGConfig(
        embeddings_model="sentence-transformers/all-MiniLM-L6-v2",
        chunk_size=1000,
        chunk_overlap=200,
        max_documents=3,
        similarity_threshold=0.7,
        cache_enabled=True,
        cache_type="redis",
        redis_password="redis123",
        redis_prefix="test:"  # Usando o mesmo prefixo dos testes
    )
    system = FAISSRAGSystem()
    system.initialize(config)
    system.llm = MockLLM()  # Usa o mock do LLM
    return system

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
    
    # Espera a expiração (2 segundos para garantir)
    time.sleep(2)
    
    # Verifica se expirou
    assert redis_cache.get('key1') is None

def test_prefix(redis_cache):
    """Testa o uso do prefixo nas chaves."""
    # Armazena um item
    redis_cache.set('key1', 'value1')
    
    # Verifica se a chave real tem o prefixo
    assert redis_cache.redis.exists('test:key1') == 1

def test_error_handling(invalid_redis_cache):
    """Testa o tratamento de erros."""
    # Verifica se as operações falham graciosamente
    assert invalid_redis_cache.set('key1', 'value1') is False
    assert invalid_redis_cache.get('key1') is None
    assert invalid_redis_cache.delete('key1') is False
    assert invalid_redis_cache.clear() is False
    assert invalid_redis_cache.ping() is False

def test_conversation_persistence(redis_cache, rag_system):
    """Testa a persistência da memória de conversação entre interações."""
    
    # Primeira interação - Usuário se apresenta
    primeira_pergunta = "Meu nome é Diego"
    primeira_resposta = rag_system.query(primeira_pergunta)
    
    # Verifica se a resposta foi armazenada no cache
    conversation_key = "test:conversation:user_info"  # Ajustando a chave com o prefixo
    cached_data = redis_cache.get(conversation_key)
    assert cached_data is not None
    assert cached_data["name"] == "diego"
    
    # Segunda interação - Pergunta sobre o nome
    segunda_pergunta = "Qual é o meu nome?"
    segunda_resposta = rag_system.query(segunda_pergunta)
    
    # Verifica se a resposta menciona o nome Diego
    assert "diego" in segunda_resposta.answer.lower()
    
    # Terceira interação - Verifica persistência após um tempo
    time.sleep(2)  # Espera um pouco
    terceira_pergunta = "Como você me chama?"
    terceira_resposta = rag_system.query(terceira_pergunta)
    
    # Verifica se ainda lembra o nome
    assert "diego" in terceira_resposta.answer.lower()
    
    # Verifica estatísticas
    assert rag_system.stats["total_queries"] == 3 