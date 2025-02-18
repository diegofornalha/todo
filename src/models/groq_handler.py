# Standard library imports
import os
import json
from datetime import datetime
from typing import Dict, Any

# Third-party imports
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
import textwrap

# Local imports
from .base_handler import BaseLLMHandler
from ..utils.logging_config import setup_logger
from ..config.prompt_templates import QA_PROMPT, DOCUMENT_PROMPT

logger = setup_logger('groq_handler')

class GroqAPIError(Exception):
    """Exceção personalizada para erros relacionados à API do Groq."""
    pass

class GroqConfigError(Exception):
    """Exceção personalizada para erros de configuração."""
    pass

class GroqHandler(BaseLLMHandler):
    def __init__(
        self,
        model_name: str = "deepseek-r1-distill-llama-70b",
        temperature: float = 0
    ):
        """
        Inicializa o handler do Groq.
        
        Args:
            model_name: Nome do modelo a ser usado
            temperature: Temperatura para geração de respostas (0-1)
            
        Raises:
            GroqConfigError: Se houver erro na configuração
        """
        self.model_name = model_name
        self.temperature = temperature
        self.handler = None
        
    def initialize(self) -> None:
        """
        Inicializa o handler do Groq e configura o ambiente.
        
        Raises:
            GroqConfigError: Se houver erro na configuração
            GroqAPIError: Se houver erro na inicialização da API
        """
        try:
            logger.info("Carregando variáveis de ambiente...")
            load_dotenv()
            
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise GroqConfigError(textwrap.dedent("""
                    GROQ_API_KEY não encontrada nas variáveis de ambiente.
                    Certifique-se de configurar a chave no arquivo .env
                """).strip())
            
            logger.info(f"Inicializando modelo Groq: {self.model_name}")
            self.handler = ChatGroq(
                groq_api_key=api_key,
                temperature=self.temperature,
                model_name=self.model_name
            )
            logger.info("Modelo Groq inicializado com sucesso")
            
        except GroqConfigError:
            raise
        except Exception as e:
            raise GroqAPIError(f"Erro ao inicializar a API do Groq: {str(e)}")
            
    def process_question(
        self,
        question: str,
        context: str = ""
    ) -> Dict[str, Any]:
        """
        Processa uma pergunta e retorna a resposta do modelo.
        
        Args:
            question: Pergunta a ser processada
            context: Contexto opcional para a pergunta
            
        Returns:
            Dicionário com a pergunta e resposta
            
        Raises:
            GroqAPIError: Se houver erro na comunicação com a API
        """
        if not self.handler:
            raise GroqConfigError("Handler não inicializado. Execute initialize() primeiro")
            
        try:
            logger.info(f"Processando pergunta: {question}")
            
            # Formata o prompt usando o template
            prompt = QA_PROMPT.format(
                context=context if context else "Nenhum contexto fornecido.",
                question=question
            )
            
            messages = [HumanMessage(content=prompt)]
            response = self.handler.invoke(messages)
            
            result = {
                "pergunta": question,
                "contexto": context,
                "resposta": response.content,
                "metadata": {
                    "model": self.model_name,
                    "temperature": self.temperature,
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            logger.info("Resposta processada com sucesso")
            logger.debug(f"Resultado completo: {json.dumps(result, ensure_ascii=False, indent=2)}")
            
            return result
            
        except ConnectionError as e:
            logger.error(f"Erro de conexão com a API: {str(e)}")
            raise GroqAPIError(f"Erro de conexão: {str(e)}")
        except Exception as e:
            logger.error(f"Erro inesperado: {str(e)}")
            raise GroqAPIError(f"Erro inesperado: {str(e)}")
            
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
            GroqAPIError: Se houver erro na comunicação com a API
        """
        if not self.handler:
            raise GroqConfigError("Handler não inicializado. Execute initialize() primeiro")
            
        try:
            logger.info(f"Processando documento - Fonte: {source}")
            logger.info(f"Pergunta: {question}")
            
            # Formata o prompt usando o template de documento
            prompt = DOCUMENT_PROMPT.format(
                page_content=page_content,
                source=source,
                question=question
            )
            
            messages = [HumanMessage(content=prompt)]
            response = self.handler.invoke(messages)
            
            result = {
                "pergunta": question,
                "fonte": source,
                "conteudo": page_content[:200] + "..." if len(page_content) > 200 else page_content,
                "resposta": response.content,
                "metadata": {
                    "model": self.model_name,
                    "temperature": self.temperature,
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            logger.info("Documento processado com sucesso")
            logger.debug(f"Resultado completo: {json.dumps(result, ensure_ascii=False, indent=2)}")
            
            return result
            
        except ConnectionError as e:
            logger.error(f"Erro de conexão com a API: {str(e)}")
            raise GroqAPIError(f"Erro de conexão: {str(e)}")
        except Exception as e:
            logger.error(f"Erro inesperado: {str(e)}")
            raise GroqAPIError(f"Erro inesperado: {str(e)}") 