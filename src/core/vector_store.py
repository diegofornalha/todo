from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import os
import pickle
import logging
from ..utils.logging_config import setup_logger

logger = setup_logger('vector_store', 'logs/vector_store.log')

class VectorStoreError(Exception):
    """Exceção personalizada para erros do VectorStore."""
    pass

class VectorStore:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        metric: str = "cosine"
    ):
        """
        Inicializa o VectorStore com um modelo de embeddings.
        
        Args:
            model_name: Nome do modelo de embeddings
            metric: Métrica de similaridade para KNN
        """
        logger.info(f"Inicializando VectorStore com modelo: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.metric = metric
        self.vectors = None
        self.documents = []
        self.knn = None
        logger.info("VectorStore inicializado com sucesso")

    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Gera embeddings para uma lista de textos.
        
        Args:
            texts: Lista de textos
            
        Returns:
            Array numpy com os embeddings
        """
        logger.info(f"Gerando embeddings para {len(texts)} textos")
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        logger.info("Embeddings gerados com sucesso")
        return embeddings

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Adiciona documentos ao store.
        
        Args:
            documents: Lista de documentos com conteúdo e metadados
        """
        logger.info(f"Adicionando {len(documents)} documentos ao store")
        
        # Extrai os textos dos documentos
        texts = [doc['content'] for doc in documents]
        embeddings = self._get_embeddings(texts)
        
        if self.vectors is None:
            logger.info("Primeira adição de documentos")
            self.vectors = embeddings
        else:
            logger.info("Concatenando novos vetores aos existentes")
            self.vectors = np.vstack([self.vectors, embeddings])
        
        self.documents.extend(documents)
        
        # Reinicializa o KNN
        logger.info("Reinicializando o KNN")
        self.knn = NearestNeighbors(
            n_neighbors=min(5, len(self.documents)),
            metric=self.metric
        )
        self.knn.fit(self.vectors)
        logger.info("KNN reinicializado com sucesso")

    def similarity_search(
        self,
        query: str,
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Realiza busca por similaridade.
        
        Args:
            query: Texto da consulta
            k: Número de resultados
            
        Returns:
            Lista dos k documentos mais similares
        """
        logger.info(f"Realizando busca por similaridade: '{query}'")
        
        if self.vectors is None or not self.documents:
            logger.warning("Nenhum documento indexado")
            return []
        
        # Gera embedding da query
        query_embedding = self._get_embeddings([query])
        
        # Busca os vizinhos mais próximos
        k = min(k, len(self.documents))
        distances, indices = self.knn.kneighbors(
            query_embedding,
            n_neighbors=k
        )
        
        # Prepara os resultados
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            doc = self.documents[idx]
            results.append({
                **doc,
                "score": 1 - dist  # Converte distância em score
            })
        
        logger.info(f"Encontrados {len(results)} resultados")
        return results
        
    def save(self, path: str) -> None:
        """
        Salva o estado do VectorStore em disco.
        
        Args:
            path: Caminho para salvar
        """
        logger.info(f"Salvando VectorStore em: {path}")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        state = {
            "vectors": self.vectors,
            "documents": self.documents,
            "metric": self.metric
        }
        
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        logger.info("VectorStore salvo com sucesso")
        
    def load(self, path: str) -> None:
        """
        Carrega o estado do VectorStore do disco.
        
        Args:
            path: Caminho para carregar
        """
        logger.info(f"Carregando VectorStore de: {path}")
        
        if not os.path.exists(path):
            error_msg = f"Arquivo não encontrado: {path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        self.vectors = state["vectors"]
        self.documents = state["documents"]
        self.metric = state["metric"]
        
        if self.vectors is not None:
            self.knn = NearestNeighbors(
                n_neighbors=min(5, len(self.documents)),
                metric=self.metric
            )
            self.knn.fit(self.vectors)
            
        logger.info("VectorStore carregado com sucesso") 