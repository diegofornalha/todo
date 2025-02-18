from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import time
from datetime import datetime
import numpy as np

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from ..utils.logging_config import setup_logger

logger = setup_logger('conversation_memory')

class ConversationMemory:
    """Gerencia a memória de conversas usando FAISS."""
    
    def __init__(
        self,
        embeddings_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        max_history: int = 10,
        similarity_threshold: float = 0.7
    ):
        """
        Inicializa a memória de conversas.
        
        Args:
            embeddings_model: Modelo para gerar embeddings
            max_history: Número máximo de mensagens no histórico
            similarity_threshold: Limiar de similaridade para recuperação
        """
        self.embeddings = HuggingFaceEmbeddings(model_name=embeddings_model)
        self.vectorstore = None
        self.max_history = max_history
        self.similarity_threshold = similarity_threshold
        self.conversation_history = []
        
        logger.info(f"ConversationMemory inicializada com modelo: {embeddings_model}")
        
    def add_message(self, message: Dict[str, Any]) -> None:
        """
        Adiciona uma mensagem ao histórico.
        
        Args:
            message: Dicionário com a mensagem (role, content, timestamp)
        """
        # Adiciona timestamp se não existir
        if 'timestamp' not in message:
            message['timestamp'] = datetime.now().isoformat()
            
        # Converte para Document
        doc = Document(
            page_content=message['content'],
            metadata={
                'role': message['role'],
                'timestamp': message['timestamp']
            }
        )
        
        # Atualiza o histórico
        self.conversation_history.append(message)
        if len(self.conversation_history) > self.max_history:
            self.conversation_history.pop(0)
        
        # Adiciona ao vectorstore
        if self.vectorstore is None:
            self.vectorstore = FAISS.from_documents([doc], self.embeddings)
        else:
            self.vectorstore.add_documents([doc])
            
        logger.info(f"Mensagem adicionada ao histórico: {message['role']}")
        
    def get_relevant_history(
        self,
        query: str,
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Recupera mensagens relevantes do histórico.
        
        Args:
            query: Texto para buscar contexto relevante
            k: Número de mensagens a recuperar
            
        Returns:
            Lista de mensagens relevantes
        """
        if not self.vectorstore:
            logger.info("Nenhuma mensagem no histórico")
            return []
            
        # Busca documentos similares
        docs_and_scores = self.vectorstore.similarity_search_with_score(
            query,
            k=min(k, len(self.conversation_history))
        )
        
        # Filtra por similaridade e ordena por timestamp
        relevant_messages = []
        for doc, score in docs_and_scores:
            if score <= self.similarity_threshold:
                message = {
                    'role': doc.metadata['role'],
                    'content': doc.page_content,
                    'timestamp': doc.metadata['timestamp']
                }
                relevant_messages.append(message)
                
        # Ordena por timestamp
        relevant_messages.sort(key=lambda x: x['timestamp'])
        
        logger.info(f"Recuperadas {len(relevant_messages)} mensagens relevantes")
        return relevant_messages
        
    def get_recent_history(self, n: int = 5) -> List[Dict[str, Any]]:
        """
        Retorna as n mensagens mais recentes.
        
        Args:
            n: Número de mensagens a retornar
            
        Returns:
            Lista das últimas n mensagens
        """
        return self.conversation_history[-n:]
        
    def clear(self) -> None:
        """Limpa todo o histórico de conversas."""
        self.vectorstore = None
        self.conversation_history = []
        logger.info("Histórico de conversas limpo")
        
    def save(self, path: str) -> None:
        """
        Salva o estado da memória em disco.
        
        Args:
            path: Caminho para salvar
        """
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Salva vectorstore
        if self.vectorstore:
            self.vectorstore.save_local(str(save_dir / "vectorstore"))
            
        # Salva histórico
        with open(save_dir / "history.json", "w") as f:
            json.dump(self.conversation_history, f, indent=2)
            
        logger.info(f"Memória de conversas salva em: {path}")
        
    def load(self, path: str) -> None:
        """
        Carrega o estado da memória do disco.
        
        Args:
            path: Caminho para carregar
        """
        load_dir = Path(path)
        if not load_dir.exists():
            raise FileNotFoundError(f"Diretório não encontrado: {path}")
            
        # Carrega vectorstore
        vectorstore_path = load_dir / "vectorstore"
        if vectorstore_path.exists():
            self.vectorstore = FAISS.load_local(
                str(vectorstore_path),
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            
        # Carrega histórico
        history_path = load_dir / "history.json"
        if history_path.exists():
            with open(history_path, "r") as f:
                self.conversation_history = json.load(f)
                
        logger.info(f"Memória de conversas carregada de: {path}") 