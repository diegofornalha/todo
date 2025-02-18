import pytest
import tempfile
import os
from pathlib import Path
import numpy as np
from src.core.vector_store import VectorStore

def test_faiss_initialization():
    """Testa a inicialização do VectorStore com FAISS."""
    store = VectorStore()
    assert store.vectorstore is None
    assert len(store.documents) == 0
    assert store.similarity_threshold == 0.6

def test_faiss_document_addition():
    """Testa a adição de documentos usando FAISS."""
    store = VectorStore()
    
    # Documentos de teste
    documents = [
        {
            "content": "FAISS é uma biblioteca eficiente para busca por similaridade",
            "metadata": {"source": "doc1.txt"}
        },
        {
            "content": "Vetores densos são usados em recuperação de informação",
            "metadata": {"source": "doc2.txt"}
        }
    ]
    
    # Adiciona documentos
    store.add_documents(documents)
    
    assert len(store.documents) == 2
    assert store.vectorstore is not None

def test_faiss_similarity_search():
    """Testa a busca por similaridade usando FAISS."""
    store = VectorStore(similarity_threshold=0.5)  # Menos restritivo para teste
    
    # Documentos de teste
    documents = [
        {
            "content": "FAISS é uma biblioteca para busca por similaridade",
            "metadata": {"source": "doc1.txt"}
        },
        {
            "content": "Python é uma linguagem de programação versátil",
            "metadata": {"source": "doc2.txt"}
        },
        {
            "content": "Busca por similaridade usa vetores densos",
            "metadata": {"source": "doc3.txt"}
        }
    ]
    
    # Adiciona documentos
    store.add_documents(documents)
    
    # Testa busca relacionada a FAISS
    results = store.similarity_search("Como funciona busca por similaridade?")
    assert len(results) > 0
    assert any("FAISS" in result["content"] for result in results)
    
    # Verifica ordenação por similaridade
    for i in range(len(results) - 1):
        assert results[i]["score"] >= results[i + 1]["score"]

def test_faiss_persistence():
    """Testa o salvamento e carregamento do índice FAISS."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = VectorStore(similarity_threshold=0.5)  # Menos restritivo para teste
        
        # Documentos de teste
        documents = [
            {
                "content": "FAISS é uma biblioteca para busca por similaridade",
                "metadata": {"source": "doc1.txt"}
            },
            {
                "content": "Vetores densos são usados em recuperação de informação",
                "metadata": {"source": "doc2.txt"}
            }
        ]
        
        # Adiciona documentos e salva
        store.add_documents(documents)
        save_path = str(Path(tmpdir) / "test_store")
        store.save(save_path)
        
        # Verifica se os arquivos foram criados
        assert os.path.exists(save_path + ".faiss")
        assert os.path.exists(save_path + ".json")
        
        # Carrega em uma nova instância
        new_store = VectorStore(similarity_threshold=0.5)
        new_store.load(save_path)
        
        # Verifica se os documentos foram preservados
        assert len(new_store.documents) == 2
        assert new_store.vectorstore is not None
        
        # Testa busca após carregar
        results = new_store.similarity_search("busca por similaridade")
        assert len(results) > 0
        assert any("FAISS" in result["content"] for result in results)

def test_faiss_empty_search():
    """Testa busca em store vazio."""
    store = VectorStore()
    results = store.similarity_search("teste")
    assert len(results) == 0

def test_faiss_threshold():
    """Testa o limiar de similaridade."""
    store = VectorStore(similarity_threshold=0.5)  # Ajusta para o novo cálculo de similaridade
    
    documents = [
        {
            "content": "FAISS é uma biblioteca para busca por similaridade",
            "metadata": {"source": "doc1.txt"}
        },
        {
            "content": "Tópico completamente não relacionado: receita de bolo",
            "metadata": {"source": "doc2.txt"}
        }
    ]
    
    store.add_documents(documents)
    
    # Busca relacionada
    results = store.similarity_search("Como fazer busca por similaridade?")
    assert len(results) == 1  # Apenas o documento relacionado
    assert "FAISS" in results[0]["content"]
    assert results[0]["score"] >= 0.5
    
    # Busca não relacionada
    results = store.similarity_search("Qual a previsão do tempo?")
    assert len(results) == 0  # Nenhum documento similar o suficiente 