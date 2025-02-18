# Standard library imports
from pathlib import Path
import textwrap
from typing import Dict, Any, Optional

# Third-party imports
from langchain_community.vectorstores import FAISS

# Local imports
from .base_test import BaseTest
from ..core.vector_store import VectorStore, VectorStoreError

class VectorStoreTest(BaseTest):
    """Testes para o VectorStore."""
    
    def __init__(self):
        super().__init__('vector_store')
        self.test_data_dir = Path('tests/data')
        
    def setup(self) -> None:
        """
        Configura o ambiente para os testes.
        Cria diretório de dados de teste se não existir.
        """
        self.test_data_dir.mkdir(parents=True, exist_ok=True)
        
    def test_store_creation(self) -> bool:
        """
        Testa a criação do VectorStore.
        
        Returns:
            bool: True se o teste passou, False caso contrário
        """
        test_name = "store_creation"
        self.log_test_start(test_name)
        
        try:
            # Cria um novo store para o teste
            vector_store = VectorStore()
            
            # Prepara dados de teste
            documents = [
                {
                    "content": "Python é uma linguagem de programação versátil",
                    "metadata": {"source": "doc1.txt"}
                },
                {
                    "content": "Machine Learning é um subcampo da Inteligência Artificial",
                    "metadata": {"source": "doc2.txt"}
                },
                {
                    "content": "Processamento de Linguagem Natural trabalha com textos",
                    "metadata": {"source": "doc3.txt"}
                }
            ]
            
            # Adiciona os documentos
            vector_store.add_documents(documents)
            
            # Verifica se os documentos foram adicionados
            if not vector_store.documents or len(vector_store.documents) != 3:
                raise ValueError("Documentos não foram adicionados corretamente")
                
            self.logger.info("VectorStore criado com sucesso")
            self.log_test_result(test_name, True)
            return True
            
        except Exception as e:
            self.log_test_result(test_name, False, e)
            return False
            
    def test_similarity_search(self) -> bool:
        """
        Testa a busca por similaridade.
        
        Returns:
            bool: True se o teste passou, False caso contrário
        """
        test_name = "similarity_search"
        self.log_test_start(test_name)
        
        try:
            # Cria um novo store para o teste
            vector_store = VectorStore()
            
            # Prepara dados de teste
            documents = [
                {
                    "content": "Python é uma linguagem de programação versátil",
                    "metadata": {"source": "doc1.txt"}
                },
                {
                    "content": "Machine Learning é um subcampo da Inteligência Artificial",
                    "metadata": {"source": "doc2.txt"}
                },
                {
                    "content": "Processamento de Linguagem Natural trabalha com textos",
                    "metadata": {"source": "doc3.txt"}
                }
            ]
            query = "Como Python é usado em IA?"
            
            # Adiciona os documentos e realiza a busca
            vector_store.add_documents(documents)
            results = vector_store.similarity_search(query)
            
            # Verifica os resultados
            if not results or len(results) == 0:
                raise ValueError("Busca não retornou resultados")
                
            self.logger.info(f"Query: {query}")
            for i, result in enumerate(results, 1):
                self.logger.info(f"Resultado {i}: {result['content']}")
                
            self.log_test_result(test_name, True)
            return True
            
        except Exception as e:
            self.log_test_result(test_name, False, e)
            return False
            
    def test_store_persistence(self) -> bool:
        """
        Testa a persistência do VectorStore.
        
        Returns:
            bool: True se o teste passou, False caso contrário
        """
        test_name = "store_persistence"
        self.log_test_start(test_name)
        
        try:
            # Cria um novo store para o teste
            vector_store = VectorStore()
            
            # Prepara dados de teste
            documents = [
                {
                    "content": "Python é uma linguagem de programação versátil",
                    "metadata": {"source": "doc1.txt"}
                },
                {
                    "content": "Machine Learning é um subcampo da Inteligência Artificial",
                    "metadata": {"source": "doc2.txt"}
                },
                {
                    "content": "Processamento de Linguagem Natural trabalha com textos",
                    "metadata": {"source": "doc3.txt"}
                }
            ]
            save_path = str(self.test_data_dir / "test_store.pkl")
            
            # Adiciona documentos e salva
            vector_store.add_documents(documents)
            vector_store.save(save_path)
            
            # Cria novo store e carrega
            new_store = VectorStore()
            new_store.load(save_path)
            
            # Verifica se os documentos foram carregados
            if len(new_store.documents) != len(documents):
                raise ValueError(f"Número de documentos carregados ({len(new_store.documents)}) não corresponde ao original ({len(documents)})")
                
            self.logger.info(f"Store salvo e carregado com sucesso em {save_path}")
            self.log_test_result(test_name, True)
            return True
            
        except Exception as e:
            self.log_test_result(test_name, False, e)
            return False
            
    def cleanup(self) -> None:
        """Remove arquivos temporários de teste."""
        if self.test_data_dir.exists():
            for file in self.test_data_dir.glob("*"):
                file.unlink()
            self.test_data_dir.rmdir()
            
    def run(self) -> bool:
        """
        Executa todos os testes do VectorStore.
        
        Returns:
            bool: True se todos os testes passaram, False caso contrário
        """
        try:
            self.setup()
            
            # Executa os testes
            tests_passed = all([
                self.test_store_creation(),
                self.test_similarity_search(),
                self.test_store_persistence()
            ])
            
            self.cleanup()
            return tests_passed
            
        except VectorStoreError as e:
            self.logger.error(f"Erro no VectorStore: {str(e)}")
            return False
        except Exception as e:
            self.logger.error(f"Erro inesperado: {str(e)}")
            self.logger.debug("Detalhes do erro:", exc_info=True)
            return False

def main():
    """Função principal para executar os testes."""
    test = VectorStoreTest()
    success = test.run()
    return 0 if success else 1

if __name__ == "__main__":
    exit(main()) 