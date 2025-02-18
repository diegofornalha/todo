# Standard library imports
import os
import logging
import textwrap
from typing import Dict

# Third-party imports
from dotenv import load_dotenv

# Local imports
from ..utils.logging_config import setup_logger

logger = setup_logger('langsmith_config')

class LangSmithConfigError(Exception):
    """Exceção personalizada para erros de configuração do LangSmith."""
    pass

class LangSmithConfig:
    """Classe para gerenciar a configuração do LangSmith."""
    
    DEFAULT_CONFIG = {
        "LANGCHAIN_TRACING_V2": "true",
        "LANGCHAIN_ENDPOINT": "https://api.smith.langchain.com",
        "LANGCHAIN_PROJECT": "todo-app-qa"
    }
    
    @staticmethod
    def configure() -> bool:
        """
        Configura o ambiente para LangSmith e LangGraph.
        Carrega a chave de API do arquivo .env e configura as variáveis de ambiente necessárias.
        
        Returns:
            bool: True se a configuração foi bem-sucedida, False caso contrário
            
        Raises:
            LangSmithConfigError: Se houver erro na configuração
        """
        try:
            logger.info("Carregando variáveis de ambiente para LangSmith...")
            load_dotenv()
            
            # Verifica a chave de API
            api_key = os.getenv("LANGCHAIN_API_KEY")
            if not api_key:
                raise LangSmithConfigError(textwrap.dedent("""
                    LANGCHAIN_API_KEY não encontrada nas variáveis de ambiente.
                    Certifique-se de configurar a chave no arquivo .env
                """).strip())
            
            # Configura as variáveis de ambiente
            for key, value in LangSmithConfig.DEFAULT_CONFIG.items():
                os.environ[key] = value
                logger.debug(f"Configurada variável: {key}")
            
            logger.info("LangSmith configurado com sucesso")
            return True
            
        except LangSmithConfigError as e:
            logger.error(str(e))
            raise
        except Exception as e:
            error_msg = f"Erro inesperado ao configurar LangSmith: {str(e)}"
            logger.error(error_msg)
            raise LangSmithConfigError(error_msg) from e
            
    @staticmethod
    def get_config() -> Dict[str, str]:
        """
        Retorna a configuração atual do LangSmith.
        
        Returns:
            Dict[str, str]: Dicionário com as configurações atuais
        """
        return {
            key: os.getenv(key, value)
            for key, value in LangSmithConfig.DEFAULT_CONFIG.items()
        }

# Função de conveniência para manter compatibilidade
def configure_langsmith() -> bool:
    """
    Função de compatibilidade para configurar o LangSmith.
    Recomenda-se usar LangSmithConfig.configure() diretamente.
    
    Returns:
        bool: True se a configuração foi bem-sucedida, False caso contrário
    """
    try:
        return LangSmithConfig.configure()
    except Exception as e:
        logger.error(f"Erro ao configurar LangSmith: {str(e)}")
        return False 