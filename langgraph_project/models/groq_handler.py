from typing import Dict, Any
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
import os
from dotenv import load_dotenv
import logging

# Configura o logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('groq_handler')

class GroqHandler:
    def __init__(self, model_name: str = "mixtral-8x7b-32768"):
        """
        Inicializa o handler do Groq.
        
        Args:
            model_name: Nome do modelo a ser usado
        """
        load_dotenv()
        
        self.model_name = model_name
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY não encontrada nas variáveis de ambiente")
            
        self.llm = ChatGroq(
            groq_api_key=self.api_key,
            model_name=self.model_name,
            temperature=0.7
        )
        
        logger.info(f"GroqHandler inicializado com modelo: {model_name}")
        
    def invoke(self, messages: list) -> Dict[str, Any]:
        """
        Processa uma lista de mensagens e retorna a resposta.
        
        Args:
            messages: Lista de mensagens no formato do ChatGroq
            
        Returns:
            Resposta do modelo
        """
        try:
            response = self.llm.invoke(messages)
            return response
            
        except Exception as e:
            logger.error(f"Erro ao processar mensagens: {str(e)}")
            raise 