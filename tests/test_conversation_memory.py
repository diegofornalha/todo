import pytest
import tempfile
from pathlib import Path
from datetime import datetime, timedelta

from langgraph_project.core.conversation_memory import ConversationMemory

def test_conversation_memory_initialization():
    """Testa a inicialização da memória de conversas."""
    memory = ConversationMemory()
    assert memory.vectorstore is None
    assert len(memory.conversation_history) == 0
    assert memory.max_history == 10
    assert memory.similarity_threshold == 0.7

def test_add_message():
    """Testa a adição de mensagens ao histórico."""
    memory = ConversationMemory()
    
    # Adiciona uma mensagem
    message = {
        'role': 'user',
        'content': 'Como posso ajudar com análise de dados?'
    }
    memory.add_message(message)
    
    assert len(memory.conversation_history) == 1
    assert memory.vectorstore is not None
    
    # Verifica se timestamp foi adicionado
    assert 'timestamp' in memory.conversation_history[0]
    
def test_max_history_limit():
    """Testa o limite máximo do histórico."""
    memory = ConversationMemory(max_history=2)
    
    # Adiciona 3 mensagens
    messages = [
        {'role': 'user', 'content': f'Mensagem {i}'} 
        for i in range(3)
    ]
    
    for msg in messages:
        memory.add_message(msg)
        
    assert len(memory.conversation_history) == 2
    assert memory.conversation_history[0]['content'] == 'Mensagem 1'
    assert memory.conversation_history[1]['content'] == 'Mensagem 2'

def test_get_relevant_history():
    """Testa a recuperação de histórico relevante."""
    memory = ConversationMemory()
    
    # Adiciona mensagens sobre diferentes tópicos
    messages = [
        {
            'role': 'user',
            'content': 'Como faço análise de dados com Python?',
            'timestamp': (datetime.now() - timedelta(minutes=5)).isoformat()
        },
        {
            'role': 'assistant',
            'content': 'Para análise de dados em Python, você pode usar pandas e numpy.',
            'timestamp': (datetime.now() - timedelta(minutes=4)).isoformat()
        },
        {
            'role': 'user',
            'content': 'Qual é a capital da França?',
            'timestamp': (datetime.now() - timedelta(minutes=3)).isoformat()
        }
    ]
    
    for msg in messages:
        memory.add_message(msg)
        
    # Busca mensagens relevantes sobre análise de dados
    relevant = memory.get_relevant_history('Como analisar dados?')
    assert len(relevant) >= 1
    assert any('análise de dados' in msg['content'] for msg in relevant)
    
def test_save_and_load():
    """Testa o salvamento e carregamento da memória."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Cria e popula a memória
        memory = ConversationMemory()
        messages = [
            {'role': 'user', 'content': 'Olá!'},
            {'role': 'assistant', 'content': 'Como posso ajudar?'}
        ]
        
        for msg in messages:
            memory.add_message(msg)
            
        # Salva a memória
        memory.save(tmpdir)
        
        # Cria nova instância e carrega
        new_memory = ConversationMemory()
        new_memory.load(tmpdir)
        
        # Verifica se o histórico foi preservado
        assert len(new_memory.conversation_history) == 2
        assert new_memory.vectorstore is not None
        
def test_clear_memory():
    """Testa a limpeza da memória."""
    memory = ConversationMemory()
    
    # Adiciona algumas mensagens
    messages = [
        {'role': 'user', 'content': 'Olá!'},
        {'role': 'assistant', 'content': 'Como posso ajudar?'}
    ]
    
    for msg in messages:
        memory.add_message(msg)
        
    # Limpa a memória
    memory.clear()
    
    assert len(memory.conversation_history) == 0
    assert memory.vectorstore is None 