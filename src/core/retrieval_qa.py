from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os
import logging
from ..utils.logging_config import setup_logger
from ..models.llm_handler import LLMHandler
from ..config.prompt_templates import QA_PROMPT, DOCUMENT_PROMPT

logger = setup_logger('retrieval_qa')

class RetrievalQA:
    def __init__(
        self,
        llm_handler: LLMHandler,
        embeddings_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """
        Inicializa o sistema de RetrievalQA.
        
        Args:
            llm_handler: Handler do modelo de linguagem
            embeddings_model: Nome do modelo de embeddings
            chunk_size: Tamanho dos chunks para divisão de documentos
            chunk_overlap: Sobreposição entre chunks
        """
        self.llm_handler = llm_handler
        self.embeddings = HuggingFaceEmbeddings(model_name=embeddings_model)
        self.vectorstore = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        logger.info(f"Sistema RetrievalQA inicializado com modelo de embeddings: {embeddings_model}")
        
    def add_documents(self, documents: List[Dict[str, str]]) -> None:
        """
        Adiciona documentos ao sistema.
        
        Args:
            documents: Lista de documentos com 'content' e 'source'
        """
        logger.info(f"Adicionando {len(documents)} documentos ao sistema")
        
        # Converte para o formato Document do LangChain
        doc_objects = [
            Document(
                page_content=doc['content'],
                metadata={'source': doc['source']}
            ) for doc in documents
        ]
        
        # Divide os documentos em chunks
        splits = self.text_splitter.split_documents(doc_objects)
        logger.info(f"Documentos divididos em {len(splits)} chunks")
        
        # Cria ou atualiza o vectorstore
        if self.vectorstore is None:
            logger.info("Inicializando novo vectorstore")
            self.vectorstore = FAISS.from_documents(splits, self.embeddings)
        else:
            logger.info("Atualizando vectorstore existente")
            self.vectorstore.add_documents(splits)
            
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
            include_sources: Se deve incluir as fontes na resposta
            
        Returns:
            Dicionário com a pergunta, resposta e metadados
        """
        logger.info(f"Processando pergunta: {question}")
        
        try:
            if self.vectorstore is None:
                raise ValueError("Nenhum documento foi adicionado ao sistema")
                
            # Recupera documentos relevantes
            retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": k}
            )
            
            # Formata os documentos recuperados
            docs = retriever.get_relevant_documents(question)
            context = self._format_documents(docs)
            
            # Processa a pergunta com o LLM
            response = self.llm_handler.process_question(question, context)
            
            # Adiciona informações sobre as fontes
            if include_sources:
                sources = [doc.metadata.get('source', 'Desconhecida') for doc in docs]
                response['sources'] = list(set(sources))  # Remove duplicatas
                
            logger.info("Resposta gerada com sucesso")
            return response
            
        except Exception as e:
            logger.error(f"Erro ao processar pergunta: {str(e)}")
            return {
                "pergunta": question,
                "resposta": "Erro ao processar a pergunta",
                "status": "erro",
                "erro": str(e)
            }
            
    def _format_documents(self, docs: List[Document]) -> str:
        """
        Formata uma lista de documentos em texto.
        
        Args:
            docs: Lista de documentos
            
        Returns:
            Texto formatado com os documentos
        """
        formatted_docs = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get('source', 'Desconhecida')
            formatted_docs.append(
                f"Documento {i+1} (Fonte: {source}):\n{doc.page_content}"
            )
        return "\n\n".join(formatted_docs)
        
    def save_vectorstore(self, path: str) -> None:
        """
        Salva o vectorstore em disco.
        
        Args:
            path: Caminho para salvar
        """
        if self.vectorstore is None:
            raise ValueError("Nenhum vectorstore para salvar")
            
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.vectorstore.save_local(path)
        logger.info(f"Vectorstore salvo em: {path}")
        
    def load_vectorstore(self, path: str) -> None:
        """
        Carrega um vectorstore do disco.
        
        Args:
            path: Caminho do vectorstore
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Vectorstore não encontrado em: {path}")
            
        self.vectorstore = FAISS.load_local(path, self.embeddings)
        logger.info(f"Vectorstore carregado de: {path}") 