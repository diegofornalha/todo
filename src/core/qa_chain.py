# Standard library imports
import logging
from typing import List, Dict, Any
import textwrap

# Local imports
from .vector_store import VectorStore
from ..models.base_handler import BaseLLMHandler
from ..utils.logging_config import setup_logger

logger = setup_logger('qa_chain', 'logs/qa_chain.log')

class QAChain:
    def __init__(
        self,
        llm_handler: BaseLLMHandler,
        embeddings_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        """
        Inicializa o sistema de QA.
        
        Args:
            llm_handler: Handler do modelo de linguagem
            embeddings_model: Nome do modelo de embeddings
        """
        self.llm_handler = llm_handler
        self.vector_store = VectorStore(model_name=embeddings_model)
        logger.info(f"QAChain inicializado com modelo de embeddings: {embeddings_model}")
        
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Adiciona documentos ao sistema.
        
        Args:
            documents: Lista de documentos com conteúdo e metadados
        """
        logger.info(f"Adicionando {len(documents)} documentos ao sistema")
        self.vector_store.add_documents(documents)
        logger.info("Documentos adicionados com sucesso")
        
    def query(
        self,
        question: str,
        k: int = 3,
        include_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Processa uma pergunta e retorna a resposta.
        
        Args:
            question: Pergunta a ser respondida
            k: Número de documentos a recuperar
            include_sources: Se deve incluir as fontes
            
        Returns:
            Dicionário com a pergunta, resposta e metadados
        """
        logger.info(f"Processando pergunta: {question}")
        
        try:
            # Recupera documentos relevantes
            results = self.vector_store.similarity_search(question, k=k)
            
            if not results:
                logger.warning("Nenhum documento relevante encontrado")
                return {
                    "pergunta": question,
                    "resposta": textwrap.dedent("""
                        Não encontrei informações relevantes para responder sua pergunta.
                        Por favor, tente reformular a pergunta ou fornecer mais contexto.
                    """).strip(),
                    "status": "sem_resultados"
                }
            
            # Formata o contexto
            context = self._format_context(results)
            
            # Processa a pergunta com o LLM
            response = self.llm_handler.process_question(question, context)
            
            # Adiciona informações sobre as fontes
            if include_sources:
                sources = []
                for doc in results:
                    if 'metadata' in doc and 'source' in doc['metadata']:
                        sources.append(doc['metadata']['source'])
                response['sources'] = list(set(sources))
            
            response['status'] = 'sucesso'
            logger.info("Resposta gerada com sucesso")
            return response
            
        except Exception as e:
            logger.error(f"Erro ao processar pergunta: {str(e)}")
            return {
                "pergunta": question,
                "resposta": textwrap.dedent("""
                    Desculpe, ocorreu um erro ao processar sua pergunta.
                    Por favor, tente novamente em alguns instantes.
                """).strip(),
                "status": "erro",
                "erro": str(e)
            }
            
    def _format_context(self, results: List[Dict[str, Any]]) -> str:
        """
        Formata os resultados da busca em um contexto.
        
        Args:
            results: Lista de documentos com scores
            
        Returns:
            Texto formatado com o contexto
        """
        formatted_docs = []
        for i, doc in enumerate(results):
            source = doc.get('metadata', {}).get('source', 'Desconhecida')
            score = doc.get('score', 0)
            formatted_docs.append(
                f"Documento {i+1} (Fonte: {source}, Relevância: {score:.2f}):\n{doc['content']}"
            )
        return "\n\n".join(formatted_docs)
        
    def save_index(self, path: str) -> None:
        """
        Salva o índice em disco.
        
        Args:
            path: Caminho para salvar
        """
        logger.info(f"Salvando índice em: {path}")
        self.vector_store.save(path)
        logger.info("Índice salvo com sucesso")
        
    def load_index(self, path: str) -> None:
        """
        Carrega o índice do disco.
        
        Args:
            path: Caminho para carregar
        """
        logger.info(f"Carregando índice de: {path}")
        self.vector_store.load(path)
        logger.info("Índice carregado com sucesso") 