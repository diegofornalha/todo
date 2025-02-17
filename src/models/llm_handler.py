from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
import logging
from ..utils.logging_config import setup_logger
from groq import Groq

logger = setup_logger('llm_handler', 'logs/llm_handler.log')

class LLMHandler(ABC):
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

class LLMHandler:
    def __init__(self, api_key: str, model: str = "mixtral-8x7b-32768"):
        """
        Inicializa o handler do modelo de linguagem.
        
        Args:
            api_key: Chave da API Groq
            model: Nome do modelo a ser usado
        """
        self.client = Groq(api_key=api_key)
        self.model = model
        logger.info(f"LLMHandler inicializado com modelo: {model}")
        
    def process_question(self, question: str, context: str) -> Dict[str, Any]:
        """
        Processa uma pergunta usando o contexto fornecido.
        
        Args:
            question: Pergunta do usuário
            context: Contexto relevante para a pergunta
            
        Returns:
            Dicionário com a resposta e metadados
        """
        try:
            # Formata o prompt
            prompt = self._format_prompt(question, context)
            
            # Chama a API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Você é um assistente especializado em responder perguntas com base em documentos fornecidos. Suas respostas devem ser precisas, concisas e baseadas apenas no contexto fornecido."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            answer = response.choices[0].message.content
            
            logger.info("Resposta gerada com sucesso")
            return {
                "pergunta": question,
                "resposta": answer,
                "modelo": self.model,
                "tokens_utilizados": response.usage.total_tokens
            }
            
        except Exception as e:
            logger.error(f"Erro ao processar pergunta com LLM: {str(e)}")
            raise
            
    def _format_prompt(self, question: str, context: str) -> str:
        """
        Formata o prompt para o modelo.
        
        Args:
            question: Pergunta do usuário
            context: Contexto relevante
            
        Returns:
            Prompt formatado
        """
        return f"""
Por favor, responda à seguinte pergunta usando apenas as informações fornecidas no contexto abaixo.
Se a resposta não puder ser encontrada no contexto, indique isso claramente.

Contexto:
{context}

Pergunta:
{question}

Resposta:""" 