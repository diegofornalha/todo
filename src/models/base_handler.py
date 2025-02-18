from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class BaseLLMHandler(ABC):
    """Classe base abstrata para handlers de modelos de linguagem."""
    
    @abstractmethod
    def initialize(self) -> None:
        """
        Inicializa o modelo e configura o ambiente.
        
        Raises:
            Exception: Se houver erro na inicialização
        """
        pass
        
    @abstractmethod
    def process_question(
        self,
        question: str,
        context: str = ""
    ) -> Dict[str, Any]:
        """
        Processa uma pergunta e retorna a resposta.
        
        Args:
            question: Pergunta a ser processada
            context: Contexto opcional para a pergunta
            
        Returns:
            Dicionário com a pergunta e resposta
            
        Raises:
            Exception: Se houver erro no processamento
        """
        pass
        
    @abstractmethod
    def process_document(
        self,
        question: str,
        page_content: str,
        source: str
    ) -> Dict[str, Any]:
        """
        Processa uma pergunta sobre um documento específico.
        
        Args:
            question: Pergunta a ser processada
            page_content: Conteúdo do documento
            source: Fonte ou identificação do documento
            
        Returns:
            Dicionário com a pergunta e resposta
            
        Raises:
            Exception: Se houver erro no processamento
        """
        pass 