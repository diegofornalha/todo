from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import os

class VectorStore:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Inicializa o VectorStore com um modelo de embeddings.
        
        Args:
            model_name: Nome do modelo de embeddings a ser usado
        """
        print(f"Inicializando VectorStore com modelo: {model_name}")
        print("Carregando modelo de embeddings...")
        self.model = SentenceTransformer(model_name)
        print("Modelo carregado com sucesso")
        self.vectors = None
        self.documents = []
        self.knn = None

    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Gera embeddings para uma lista de textos.
        
        Args:
            texts: Lista de textos para gerar embeddings
            
        Returns:
            Array numpy com os embeddings
        """
        print(f"Gerando embeddings para {len(texts)} textos...")
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        print("Embeddings gerados com sucesso")
        return embeddings

    def add_documents(self, documents: List[Dict[str, Any]], texts: List[str]):
        """
        Adiciona documentos ao store.
        
        Args:
            documents: Lista de documentos com metadados
            texts: Lista de textos correspondentes aos documentos
        """
        print(f"Adicionando {len(documents)} documentos ao store...")
        embeddings = self._get_embeddings(texts)
        
        if self.vectors is None:
            print("Primeira adição de documentos, inicializando vectors...")
            self.vectors = embeddings
        else:
            print("Concatenando novos vetores aos existentes...")
            self.vectors = np.vstack([self.vectors, embeddings])
        
        self.documents.extend(documents)
        
        print("Reinicializando o KNN com os novos vetores...")
        # Reinicializa o KNN com os novos vetores
        self.knn = NearestNeighbors(n_neighbors=5, metric='cosine')
        self.knn.fit(self.vectors)
        print("KNN reinicializado com sucesso")

    def similarity_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Realiza busca por similaridade.
        
        Args:
            query: Texto da consulta
            k: Número de resultados a retornar
            
        Returns:
            Lista dos k documentos mais similares
        """
        print(f"Realizando busca por similaridade para: '{query}'")
        query_embedding = self._get_embeddings([query])
        
        print(f"Buscando os {k} vizinhos mais próximos...")
        # Encontra os k vizinhos mais próximos
        distances, indices = self.knn.kneighbors(
            query_embedding,
            n_neighbors=min(k, len(self.documents))
        )
        
        print("Preparando resultados...")
        # Retorna os documentos correspondentes
        results = [self.documents[i] for i in indices[0]]
        print(f"Encontrados {len(results)} resultados")
        return results

# Exemplo de uso:
if __name__ == "__main__":
    # Inicializa o store
    store = VectorStore()
    
    # Adiciona alguns documentos
    documents = [
        {"content": "O gato dorme no sofá", "id": 1},
        {"content": "O cachorro brinca no jardim", "id": 2},
        {"content": "A criança lê um livro", "id": 3},
    ]
    texts = [doc["content"] for doc in documents]
    
    store.add_documents(documents, texts)
    
    # Faz uma busca
    results = store.similarity_search("animal dormindo")
    
    # Mostra os resultados
    for doc in results:
        print(f"ID: {doc['id']}, Conteúdo: {doc['content']}") 