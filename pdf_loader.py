from langchain_community.document_loaders import PyPDFLoader
from vector_store import VectorStore
from typing import List, Dict
import os

class DocumentProcessor:
    def __init__(self, pdf_path: str):
        """
        Inicializa o processador de documentos.
        
        Args:
            pdf_path: Caminho para o arquivo PDF
        """
        print(f"Inicializando DocumentProcessor com arquivo: {pdf_path}")
        self.pdf_path = pdf_path
        print("Inicializando VectorStore...")
        self.vector_store = VectorStore()
        print("VectorStore inicializado")
        
    def process_pdf(self) -> List[Dict]:
        """
        Processa o PDF e adiciona ao vector store.
        
        Returns:
            Lista de documentos processados
        """
        print(f"Verificando se o arquivo existe em: {os.path.abspath(self.pdf_path)}")
        # Verifica se o arquivo existe
        if not os.path.exists(self.pdf_path):
            raise FileNotFoundError(f"Arquivo não encontrado: {self.pdf_path}")
            
        print("Arquivo encontrado, tentando carregar...")
        # Carrega o PDF
        loader = PyPDFLoader(self.pdf_path)
        print("Loader criado, tentando carregar documentos...")
        documents = loader.load()
        print(f"Carregadas {len(documents)} páginas do PDF")
        
        # Prepara os documentos para o vector store
        processed_docs = []
        texts = []
        
        print("Processando documentos...")
        for i, doc in enumerate(documents):
            processed_doc = {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "id": i
            }
            processed_docs.append(processed_doc)
            texts.append(doc.page_content)
        
        print("Adicionando documentos ao vector store...")
        # Adiciona ao vector store
        self.vector_store.add_documents(processed_docs, texts)
        print("Documentos adicionados com sucesso")
        
        return processed_docs
    
    def search(self, query: str, k: int = 5) -> List[Dict]:
        """
        Realiza busca semântica nos documentos.
        
        Args:
            query: Texto da consulta
            k: Número de resultados a retornar
            
        Returns:
            Lista dos documentos mais relevantes
        """
        print(f"Realizando busca por: '{query}'")
        return self.vector_store.similarity_search(query, k)

# Exemplo de uso
if __name__ == "__main__":
    # Caminho para o PDF na pasta content
    pdf_path = "content/Origem.pdf"
    
    try:
        print("Iniciando processamento do PDF...")
        # Inicializa o processador
        processor = DocumentProcessor(pdf_path)
        
        # Processa o PDF
        docs = processor.process_pdf()
        print(f"Processadas {len(docs)} páginas do PDF")
        
        # Faz uma busca de exemplo
        query = "Qual é a origem do universo?"
        print(f"\nRealizando busca: '{query}'")
        results = processor.search(query)
        
        print(f"\nResultados para a busca: '{query}'")
        for doc in results:
            print(f"\nPágina {doc['metadata']['page']}")
            print(f"Conteúdo: {doc['content'][:200]}...")
    except Exception as e:
        print(f"Erro ao processar o PDF: {e}")
        import traceback
        traceback.print_exc()