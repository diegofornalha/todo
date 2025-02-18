from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
import os
import json
import logging
from ..utils.logging_config import setup_logger

logger = setup_logger('vector_store', 'logs/vector_store.log')

class VectorStoreError(Exception):
    """Exceção personalizada para erros do VectorStore."""
    pass

class SentenceTransformerEmbeddings(Embeddings):
    """Wrapper para SentenceTransformer compatível com LangChain."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Inicializa o wrapper."""
        self.model = SentenceTransformer(model_name)
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Gera embeddings para uma lista de textos."""
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()
        
    def embed_query(self, text: str) -> List[float]:
        """Gera embedding para um texto."""
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

class VectorStore:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        similarity_threshold: float = 0.6
    ):
        """
        Inicializa o VectorStore com FAISS.
        
        Args:
            model_name: Nome do modelo de embeddings
            similarity_threshold: Limiar de similaridade para busca (0.0 a 1.0)
        """
        logger.info(f"Inicializando VectorStore com modelo: {model_name}")
        self.embeddings = SentenceTransformerEmbeddings(model_name)
        self.similarity_threshold = similarity_threshold
        self.vectorstore = None
        self.documents = []
        logger.info("VectorStore inicializado com sucesso")

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Adiciona documentos ao store.
        
        Args:
            documents: Lista de documentos com conteúdo e metadados
        """
        logger.info(f"Adicionando {len(documents)} documentos ao store")
        
        # Converte para o formato Document do LangChain
        docs = [
            Document(
                page_content=doc['content'],
                metadata=doc.get('metadata', {})
            )
            for doc in documents
        ]
        
        # Adiciona ao vectorstore
        if self.vectorstore is None:
            logger.info("Primeira adição de documentos")
            self.vectorstore = FAISS.from_documents(docs, self.embeddings)
        else:
            logger.info("Adicionando novos documentos ao vectorstore existente")
            self.vectorstore.add_documents(docs)
        
        self.documents.extend(documents)
        logger.info("Documentos adicionados com sucesso")

    def similarity_search(
        self,
        query: str,
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Realiza busca por similaridade usando FAISS.
        
        Args:
            query: Texto da consulta
            k: Número de resultados
            
        Returns:
            Lista dos k documentos mais similares
        """
        logger.info(f"Realizando busca por similaridade: '{query}'")
        
        if self.vectorstore is None:
            logger.warning("Nenhum documento indexado")
            return []
        
        # Busca documentos similares
        k = min(k, len(self.documents))
        docs_and_scores = self.vectorstore.similarity_search_with_score(
            query,
            k=k
        )
        
        # Prepara os resultados
        results = []
        for doc, score in docs_and_scores:
            # FAISS retorna distância L2 ao quadrado
            # Convertemos para similaridade cosseno aproximada
            # Normaliza para [0,1] usando uma função sigmoide
            similarity = 1 / (1 + np.exp(score - 1))
            if similarity >= self.similarity_threshold:
                results.append({
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'score': similarity
                })
        
        # Ordena por similaridade
        results.sort(key=lambda x: x['score'], reverse=True)
        
        logger.info(f"Encontrados {len(results)} resultados")
        return results
        
    def save(self, path: str) -> None:
        """
        Salva o estado do VectorStore em disco.
        
        Args:
            path: Caminho para salvar
        """
        logger.info(f"Salvando VectorStore em: {path}")
        save_dir = os.path.dirname(path)
        os.makedirs(save_dir, exist_ok=True)
        
        # Salva vectorstore
        if self.vectorstore:
            self.vectorstore.save_local(path + ".faiss")
            
        # Salva documentos
        with open(path + ".json", "w") as f:
            json.dump(self.documents, f, indent=2)
            
        logger.info("VectorStore salvo com sucesso")
        
    def load(self, path: str) -> None:
        """
        Carrega o estado do VectorStore do disco.
        
        Args:
            path: Caminho para carregar
        """
        logger.info(f"Carregando VectorStore de: {path}")
        
        # Verifica se os arquivos existem
        if not os.path.exists(path + ".faiss"):
            error_msg = f"Arquivo FAISS não encontrado: {path}.faiss"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
            
        if not os.path.exists(path + ".json"):
            error_msg = f"Arquivo de documentos não encontrado: {path}.json"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        # Carrega vectorstore
        self.vectorstore = FAISS.load_local(
            path + ".faiss",
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        
        # Carrega documentos
        with open(path + ".json", "r") as f:
            self.documents = json.load(f)
            
        logger.info("VectorStore carregado com sucesso") 