from dotenv import load_dotenv
import os
import logging
from ..utils.logging_config import setup_logger

logger = setup_logger('langsmith_config')

def configure_langsmith():
    """
    Configura o ambiente para LangSmith e LangGraph.
    Carrega a chave de API do arquivo .env e configura as variáveis de ambiente necessárias.
    """
    try:
        logger.info("Carregando variáveis de ambiente para LangSmith...")
        load_dotenv()
        
        api_key = os.getenv("LANGCHAIN_API_KEY")
        if not api_key:
            raise ValueError("LANGCHAIN_API_KEY não encontrada nas variáveis de ambiente")
            
        # Configura as variáveis de ambiente necessárias para o LangSmith
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
        os.environ["LANGCHAIN_PROJECT"] = "todo-app-qa"  # Nome do projeto
        
        logger.info("LangSmith configurado com sucesso")
        return True
        
    except Exception as e:
        logger.error(f"Erro ao configurar LangSmith: {str(e)}")
        return False 