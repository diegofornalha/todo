from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import os
import json
from typing import Optional, Dict, Any
from ..utils.logging_config import setup_logger
from ..config.prompt_templates import QA_PROMPT, DOCUMENT_PROMPT

class GroqAPIError(Exception):
    """Exceção personalizada para erros relacionados à API do Groq."""
    pass

class GroqConfigError(Exception):
    """Exceção personalizada para erros de configuração."""
    pass

class GroqApp:
    def __init__(self, model_name: str = "deepseek-r1-distill-llama-70b", temperature: float = 0):
        """
        Inicializa a aplicação Groq.
        
        Args:
            model_name: Nome do modelo a ser usado
            temperature: Temperatura para geração de respostas (0-1)
            
        Raises:
            GroqConfigError: Se houver erro na configuração
        """
        self.logger = setup_logger('groq_app')
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
            self.logger.info("Carregando variáveis de ambiente...")
            load_dotenv()
            
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise GroqConfigError("GROQ_API_KEY não encontrada nas variáveis de ambiente")
            
            self.logger.info(f"Inicializando modelo Groq: {self.model_name}")
            self.handler = ChatGroq(
                groq_api_key=api_key,
                temperature=self.temperature,
                model_name=self.model_name
            )
            self.logger.info("Modelo Groq inicializado com sucesso")
            
        except GroqConfigError:
            raise
        except Exception as e:
            raise GroqAPIError(f"Erro ao inicializar a API do Groq: {str(e)}")
    
    def process_question(self, question: str, context: str = "") -> Optional[Dict[str, Any]]:
        """
        Processa uma pergunta e retorna a resposta do modelo.
        
        Args:
            question: Pergunta a ser processada
            context: Contexto opcional para a pergunta
            
        Returns:
            Dicionário com a pergunta e resposta, ou None em caso de erro
            
        Raises:
            GroqAPIError: Se houver erro na comunicação com a API
        """
        if not self.handler:
            raise GroqConfigError("Handler não inicializado. Execute initialize() primeiro")
            
        try:
            self.logger.info(f"Processando pergunta: {question}")
            
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
                    "timestamp": self.logger.handlers[0].formatter.formatTime(
                        self.logger.makeRecord('', 0, '', 0, '', (), None)
                    )
                }
            }
            
            self.logger.info("Resposta processada com sucesso")
            self.logger.debug(f"Resultado completo: {json.dumps(result, ensure_ascii=False, indent=2)}")
            
            return result
            
        except ConnectionError as e:
            self.logger.error(f"Erro de conexão com a API: {str(e)}")
            raise GroqAPIError(f"Erro de conexão: {str(e)}")
        except Exception as e:
            self.logger.error(f"Erro inesperado: {str(e)}")
            raise GroqAPIError(f"Erro inesperado: {str(e)}")

    def process_document(self, question: str, page_content: str, source: str) -> Optional[Dict[str, Any]]:
        """
        Processa uma pergunta sobre um documento específico.
        
        Args:
            question: Pergunta a ser processada
            page_content: Conteúdo do documento
            source: Fonte ou identificação do documento
            
        Returns:
            Dicionário com a pergunta e resposta, ou None em caso de erro
            
        Raises:
            GroqAPIError: Se houver erro na comunicação com a API
        """
        if not self.handler:
            raise GroqConfigError("Handler não inicializado. Execute initialize() primeiro")
            
        try:
            self.logger.info(f"Processando documento - Fonte: {source}")
            self.logger.info(f"Pergunta: {question}")
            
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
                    "timestamp": self.logger.handlers[0].formatter.formatTime(
                        self.logger.makeRecord('', 0, '', 0, '', (), None)
                    )
                }
            }
            
            self.logger.info("Documento processado com sucesso")
            self.logger.debug(f"Resultado completo: {json.dumps(result, ensure_ascii=False, indent=2)}")
            
            return result
            
        except ConnectionError as e:
            self.logger.error(f"Erro de conexão com a API: {str(e)}")
            raise GroqAPIError(f"Erro de conexão: {str(e)}")
        except Exception as e:
            self.logger.error(f"Erro inesperado: {str(e)}")
            raise GroqAPIError(f"Erro inesperado: {str(e)}")

def main():
    """
    Função principal que executa o teste da aplicação Groq.
    """
    logger = setup_logger('main')
    app = GroqApp()
    
    try:
        # Inicializa a aplicação
        logger.info("Iniciando aplicação Groq...")
        app.initialize()
        
        # Teste 1: Pergunta geral com contexto
        logger.info("\n=== Teste 1: Pergunta Geral ===")
        question1 = "Quais são as principais características do Python?"
        context = """Python é uma linguagem de programação de alto nível criada por Guido van Rossum. 
        É conhecida por sua sintaxe clara e legível, sendo muito usada em desenvolvimento web, 
        ciência de dados e inteligência artificial. Python tem uma grande comunidade e muitas bibliotecas."""
        
        result1 = app.process_question(question1, context)
        logger.info(f"Pergunta: {result1['pergunta']}")
        logger.info(f"Contexto: {result1['contexto']}")
        logger.info(f"Resposta: {result1['resposta']}")
        
        # Teste 2: Análise de documento
        logger.info("\n=== Teste 2: Análise de Documento ===")
        question2 = "Qual é o principal objetivo do documento?"
        page_content = """Este documento descreve a implementação de um sistema de processamento de linguagem natural
        usando Python e a biblioteca transformers. O sistema é capaz de realizar análise de sentimentos,
        classificação de texto e responder perguntas sobre documentos."""
        source = "documentacao_tecnica.pdf"
        
        result2 = app.process_document(question2, page_content, source)
        logger.info(f"Pergunta: {result2['pergunta']}")
        logger.info(f"Fonte: {result2['fonte']}")
        logger.info(f"Resposta: {result2['resposta']}")
        
    except GroqConfigError as e:
        logger.error(f"Erro de configuração: {str(e)}")
        return 1
    except GroqAPIError as e:
        logger.error(f"Erro na API: {str(e)}")
        return 1
    except Exception as e:
        logger.error(f"Erro inesperado: {str(e)}")
        logger.debug("Detalhes do erro:", exc_info=True)
        return 1
        
    return 0

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code) 