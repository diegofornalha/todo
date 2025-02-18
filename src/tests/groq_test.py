from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import os
import json
from typing import Optional, Dict, Any
from ..utils.logging_config import setup_logger
from ..config.prompt_templates import QA_PROMPT, DOCUMENT_PROMPT
from .base_test import BaseTest
from ..models.groq_handler import GroqHandler, GroqAPIError, GroqConfigError
import textwrap

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

class GroqTest(BaseTest):
    """Testes para o handler do Groq."""
    
    def __init__(self):
        super().__init__('groq_handler')
        self.handler = None
        
    def setup(self) -> None:
        """
        Configura o ambiente para os testes.
        Inicializa o handler do Groq.
        """
        self.handler = GroqHandler()
        self.handler.initialize()
        
    def test_general_question(self) -> bool:
        """
        Testa uma pergunta geral com contexto.
        
        Returns:
            bool: True se o teste passou, False caso contrário
        """
        test_name = "general_question"
        self.log_test_start(test_name)
        
        try:
            # Prepara a pergunta e contexto
            question = "Quais são as principais características do Python?"
            context = textwrap.dedent("""
                Python é uma linguagem de programação de alto nível criada por Guido van Rossum.
                É conhecida por sua sintaxe clara e legível, sendo muito usada em desenvolvimento web,
                ciência de dados e inteligência artificial. Python tem uma grande comunidade e muitas bibliotecas.
            """).strip()
            
            # Processa a pergunta
            result = self.handler.process_question(question, context)
            
            # Verifica o resultado
            if not result or 'resposta' not in result:
                raise ValueError("Resposta não contém os campos esperados")
                
            self.logger.info(f"Pergunta: {result['pergunta']}")
            self.logger.info(f"Contexto: {result['contexto']}")
            self.logger.info(f"Resposta: {result['resposta']}")
            
            self.log_test_result(test_name, True)
            return True
            
        except Exception as e:
            self.log_test_result(test_name, False, e)
            return False
            
    def test_document_analysis(self) -> bool:
        """
        Testa a análise de um documento específico.
        
        Returns:
            bool: True se o teste passou, False caso contrário
        """
        test_name = "document_analysis"
        self.log_test_start(test_name)
        
        try:
            # Prepara os dados do documento
            question = "Qual é o principal objetivo do documento?"
            page_content = textwrap.dedent("""
                Este documento descreve a implementação de um sistema de processamento de linguagem natural
                usando Python e a biblioteca transformers. O sistema é capaz de realizar análise de sentimentos,
                classificação de texto e responder perguntas sobre documentos.
            """).strip()
            source = "documentacao_tecnica.pdf"
            
            # Processa o documento
            result = self.handler.process_document(question, page_content, source)
            
            # Verifica o resultado
            if not result or 'resposta' not in result:
                raise ValueError("Resposta não contém os campos esperados")
                
            self.logger.info(f"Pergunta: {result['pergunta']}")
            self.logger.info(f"Fonte: {result['fonte']}")
            self.logger.info(f"Resposta: {result['resposta']}")
            
            self.log_test_result(test_name, True)
            return True
            
        except Exception as e:
            self.log_test_result(test_name, False, e)
            return False
            
    def run(self) -> bool:
        """
        Executa todos os testes do Groq.
        
        Returns:
            bool: True se todos os testes passaram, False caso contrário
        """
        try:
            self.setup()
            
            # Executa os testes
            tests_passed = all([
                self.test_general_question(),
                self.test_document_analysis()
            ])
            
            return tests_passed
            
        except GroqConfigError as e:
            self.logger.error(f"Erro de configuração: {str(e)}")
            return False
        except GroqAPIError as e:
            self.logger.error(f"Erro na API: {str(e)}")
            return False
        except Exception as e:
            self.logger.error(f"Erro inesperado: {str(e)}")
            self.logger.debug("Detalhes do erro:", exc_info=True)
            return False

def main():
    """Função principal para executar os testes."""
    test = GroqTest()
    success = test.run()
    return 0 if success else 1

if __name__ == "__main__":
    exit(main()) 