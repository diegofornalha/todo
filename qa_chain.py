from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List, Dict, Any
import logging
from logging_config import setup_logger
import os

# Configurar tokenizers para evitar warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = setup_logger('qa_chain')

class QAChain:
    def __init__(self, llm, embeddings_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Inicializa o sistema de QA com documentos combinados.
        
        Args:
            llm: Modelo de linguagem a ser usado
            embeddings_model: Modelo de embeddings a ser usado
        """
        self.llm = llm
        self.embeddings = HuggingFaceEmbeddings(model_name=embeddings_model)
        self.vectorstore = None
        
        # Template para combinar documentos
        self.document_prompt = PromptTemplate.from_template("""
Contexto relevante: {context}
Fonte: {source}

Pergunta: {question}

Use o contexto acima para responder à pergunta. Se a informação não estiver no contexto, responda "Não encontrei informação suficiente no contexto".
Responda em português do Brasil e de forma objetiva.

Resposta:""")
        
    def add_documents(self, documents: List[Dict[str, str]]) -> None:
        """
        Adiciona documentos ao vectorstore.
        
        Args:
            documents: Lista de documentos com 'content' e 'source'
        """
        logger.info(f"Adicionando {len(documents)} documentos ao vectorstore")
        
        # Converte para o formato Document do LangChain
        doc_objects = [
            Document(
                page_content=doc['content'],
                metadata={'source': doc['source']}
            ) for doc in documents
        ]
        
        # Divide os documentos em chunks menores
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(doc_objects)
        
        # Cria ou atualiza o vectorstore
        if self.vectorstore is None:
            self.vectorstore = FAISS.from_documents(splits, self.embeddings)
        else:
            self.vectorstore.add_documents(splits)
            
        logger.info("Documentos adicionados com sucesso")
        
    def create_chain(self) -> Any:
        """
        Cria a chain de processamento para QA.
        
        Returns:
            Chain de processamento
        """
        if self.vectorstore is None:
            raise ValueError("Nenhum documento foi adicionado ao vectorstore")
            
        # Configura o retriever
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        
        # Função para formatar o contexto
        def format_docs(docs):
            formatted_docs = []
            for i, doc in enumerate(docs):
                source = doc.metadata.get('source', 'Desconhecida')
                formatted_docs.append(f"Documento {i+1} (Fonte: {source}):\n{doc.page_content}")
            return "\n\n".join(formatted_docs)
        
        # Monta a chain
        chain = (
            {
                "context": retriever | format_docs, 
                "question": RunnablePassthrough(),
                "source": lambda x: "Múltiplas fontes consultadas"
            }
            | self.document_prompt
            | self.llm
            | StrOutputParser()
        )
        
        return chain
        
    def query(self, question: str) -> Dict[str, Any]:
        """
        Processa uma pergunta e retorna a resposta.
        
        Args:
            question: Pergunta a ser respondida
            
        Returns:
            Dicionário com a pergunta e resposta
        """
        logger.info(f"Processando pergunta: {question}")
        
        try:
            chain = self.create_chain()
            response = chain.invoke(question)
            
            result = {
                "pergunta": question,
                "resposta": response,
                "status": "sucesso"
            }
            
            logger.info("Resposta gerada com sucesso")
            return result
            
        except Exception as e:
            logger.error(f"Erro ao processar pergunta: {str(e)}")
            return {
                "pergunta": question,
                "resposta": "Erro ao processar a pergunta",
                "status": "erro",
                "erro": str(e)
            }

def main():
    """
    Função principal para testar o sistema de QA.
    """
    from groq_test import GroqApp
    
    logger.info("Iniciando teste do sistema de QA...")
    
    try:
        # Inicializa o Groq
        app = GroqApp()
        app.initialize()
        
        # Cria o sistema de QA
        qa_system = QAChain(app.handler)
        
        # Adiciona alguns documentos de exemplo
        documents = [
            {
                "content": """A função de ativação Sigmoid é uma função matemática que transforma 
                qualquer número real em um valor entre 0 e 1. É amplamente utilizada em redes neurais, 
                especialmente na camada de saída para problemas de classificação binária. 
                A fórmula da função sigmoid é f(x) = 1 / (1 + e^(-x)).""",
                "source": "neural_networks_basics.pdf"
            },
            {
                "content": """Sigmoid tem algumas limitações importantes: pode sofrer do problema de 
                desvanecimento do gradiente em redes profundas, tem saída não centralizada em zero, 
                e é computacionalmente mais cara que ReLU.""",
                "source": "activation_functions_comparison.pdf"
            }
        ]
        
        qa_system.add_documents(documents)
        
        # Testa algumas perguntas
        questions = [
            "O que é a função Sigmoid?",
            "Quais são as limitações da Sigmoid?",
            "Qual é a fórmula da Sigmoid?"
        ]
        
        logger.info("\n=== Iniciando testes de perguntas ===")
        
        for question in questions:
            logger.info(f"\nTestando pergunta: {question}")
            result = qa_system.query(question)
            
            logger.info("=== Resultado ===")
            logger.info(f"Pergunta: {result['pergunta']}")
            logger.info(f"Resposta: {result['resposta']}")
            logger.info(f"Status: {result['status']}")
            
    except Exception as e:
        logger.error(f"Erro durante o teste: {str(e)}")
        return 1
        
    return 0

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code) 