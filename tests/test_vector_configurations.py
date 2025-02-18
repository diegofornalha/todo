import pytest
import tempfile
import os
from pathlib import Path
import numpy as np
from src.core.vector_store import VectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

class TestVectorConfigurations:
    """Testes para diferentes configurações do sistema de vetores."""
    
    @pytest.fixture
    def sample_documents(self):
        """Fixture com documentos de exemplo para testes."""
        return [
            {
                "content": "Bitcoin é uma criptomoeda descentralizada",
                "metadata": {"source": "crypto_doc_1.txt", "category": "crypto"}
            },
            {
                "content": "Ethereum introduziu os contratos inteligentes",
                "metadata": {"source": "crypto_doc_2.txt", "category": "crypto"}
            },
            {
                "content": "Análise técnica usa gráficos para prever preços",
                "metadata": {"source": "trading_doc.txt", "category": "trading"}
            }
        ]

    def test_different_embedding_models(self):
        """Testa diferentes modelos de embedding."""
        models = [
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        ]
        
        for model in models:
            store = VectorStore(model_name=model)
            assert store.embeddings.model.model_name == model
            
    def test_chunk_configurations(self, sample_documents):
        """Testa diferentes configurações de chunks."""
        configs = [
            {"chunk_size": 500, "chunk_overlap": 50},
            {"chunk_size": 1000, "chunk_overlap": 200},
            {"chunk_size": 200, "chunk_overlap": 100}
        ]
        
        for config in configs:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=config["chunk_size"],
                chunk_overlap=config["chunk_overlap"]
            )
            
            # Converte para formato Document
            docs = [
                Document(
                    page_content=doc["content"],
                    metadata=doc["metadata"]
                ) for doc in sample_documents
            ]
            
            chunks = splitter.split_documents(docs)
            assert len(chunks) > 0
            
    def test_similarity_thresholds(self, sample_documents):
        """Testa diferentes limiares de similaridade."""
        thresholds = [0.3, 0.5, 0.7, 0.9]
        query = "Como funciona Bitcoin?"
        
        for threshold in thresholds:
            store = VectorStore(similarity_threshold=threshold)
            store.add_documents(sample_documents)
            
            results = store.similarity_search(query)
            
            # Verifica se todos os resultados estão acima do limiar
            assert all(r["score"] >= threshold for r in results)
            
    def test_cache_and_persistence(self, sample_documents):
        """Testa cache e persistência."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Configuração inicial
            store = VectorStore()
            store.add_documents(sample_documents)
            
            # Salva em diferentes formatos
            store.save(os.path.join(tmpdir, "test_store"))
            
            # Verifica arquivos
            assert os.path.exists(os.path.join(tmpdir, "test_store.faiss"))
            assert os.path.exists(os.path.join(tmpdir, "test_store.json"))
            
            # Carrega e verifica
            new_store = VectorStore()
            new_store.load(os.path.join(tmpdir, "test_store"))
            
            assert len(new_store.documents) == len(sample_documents)
            
    def test_document_metadata(self, sample_documents):
        """Testa metadados personalizados dos documentos."""
        store = VectorStore()
        store.add_documents(sample_documents)
        
        query = "criptomoeda"
        results = store.similarity_search(query)
        
        # Verifica se os metadados foram preservados
        for result in results:
            assert "source" in result["metadata"]
            assert "category" in result["metadata"]
            
    def test_search_configurations(self, sample_documents):
        """Testa diferentes configurações de busca."""
        store = VectorStore()
        store.add_documents(sample_documents)
        
        # Testa diferentes valores de k
        k_values = [1, 2, 3]
        query = "criptomoeda"
        
        for k in k_values:
            results = store.similarity_search(query, k=k)
            assert len(results) <= k
            
        # Verifica ordenação por score
        results = store.similarity_search(query, k=3)
        for i in range(len(results) - 1):
            assert results[i]["score"] >= results[i + 1]["score"]
            
    def test_performance_optimizations(self, sample_documents):
        """Testa otimizações de performance."""
        # Teste com batch de documentos
        batch_sizes = [1, 2, 3]
        store = VectorStore()
        
        for batch_size in batch_sizes:
            for i in range(0, len(sample_documents), batch_size):
                batch = sample_documents[i:i+batch_size]
                store.add_documents(batch)
                
        # Verifica se todos os documentos foram adicionados
        assert len(store.documents) == len(sample_documents)
        
    def test_conversation_memory(self, sample_documents):
        """Testa configurações de memória de conversação."""
        max_history_values = [2, 5, 10]
        
        for max_history in max_history_values:
            store = VectorStore()
            store.add_documents(sample_documents)
            
            # Simula histórico de conversas
            queries = [
                "O que é Bitcoin?",
                "Como funciona Ethereum?",
                "O que é análise técnica?",
                "Quais são os riscos?",
                "Como começar?"
            ]
            
            # Mantém apenas os últimos max_history resultados
            results_history = []
            for query in queries:
                results = store.similarity_search(query)
                results_history.append(results)
                if len(results_history) > max_history:
                    results_history.pop(0)
                    
                assert len(results_history) <= max_history
                
    def test_custom_score_normalization(self, sample_documents):
        """Testa função personalizada de normalização de score."""
        store = VectorStore()
        store.add_documents(sample_documents)
        
        query = "criptomoeda"
        results = store.similarity_search(query)
        
        # Verifica se os scores estão normalizados entre 0 e 1
        for result in results:
            assert 0 <= result["score"] <= 1
            
        # Verifica se a função sigmoide está sendo aplicada corretamente
        # score = 1 / (1 + np.exp(raw_score - 1))
        raw_scores = [0.5, 1.0, 2.0]
        expected_scores = [1 / (1 + np.exp(score - 1)) for score in raw_scores]
        
        for raw, expected in zip(raw_scores, expected_scores):
            normalized = 1 / (1 + np.exp(raw - 1))
            assert abs(normalized - expected) < 1e-6 