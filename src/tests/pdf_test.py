# Standard library imports
from pathlib import Path
import shutil
from typing import Dict, Any, Optional

# Local imports
from .base_test import BaseTest
from ..core.document_processor import DocumentProcessor

class DocumentProcessorTest(BaseTest):
    """Testes para o DocumentProcessor."""
    
    def __init__(self):
        super().__init__('document_processor')
        self.processor = None
        self.test_data_dir = Path('tests/data')
        self.test_pdf_path = self.test_data_dir / 'test.pdf'
        
    def setup(self) -> None:
        """
        Configura o ambiente para os testes.
        Copia o arquivo PDF de teste para o diretório de testes.
        """
        # Cria diretório de teste se não existir
        self.test_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Copia o PDF de teste para o diretório de testes
        source_pdf = Path('/Users/chain/Desktop/miniapp/todo/content/AGENTE IA .pdf')
        if source_pdf.exists():
            shutil.copy2(source_pdf, self.test_pdf_path)
        else:
            raise ValueError("Arquivo PDF de teste não encontrado")
            
        self.processor = DocumentProcessor(str(self.test_pdf_path))
        
    def test_pdf_exists(self) -> bool:
        """
        Testa se o arquivo PDF existe.
        
        Returns:
            bool: True se o teste passou, False caso contrário
        """
        test_name = "pdf_exists"
        self.log_test_start(test_name)
        
        try:
            if not self.test_pdf_path.exists():
                raise ValueError("Arquivo PDF de teste não encontrado")
                
            self.logger.info(f"PDF encontrado em: {self.test_pdf_path}")
            self.log_test_result(test_name, True)
            return True
            
        except Exception as e:
            self.log_test_result(test_name, False, e)
            return False
            
    def test_pdf_loading(self) -> bool:
        """
        Testa o carregamento do PDF.
        
        Returns:
            bool: True se o teste passou, False caso contrário
        """
        test_name = "pdf_loading"
        self.log_test_start(test_name)
        
        try:
            # Carrega o PDF
            pages = self.processor.process_pdf()
            
            # Verifica se há páginas
            if not pages:
                raise ValueError("Nenhuma página encontrada no PDF")
                
            self.logger.info(f"Número de páginas carregadas: {len(pages)}")
            self.logger.info(f"Conteúdo da primeira página: {pages[0]['content'][:200]}...")
            
            self.log_test_result(test_name, True)
            return True
            
        except Exception as e:
            self.log_test_result(test_name, False, e)
            return False
            
    def test_pdf_metadata(self) -> bool:
        """
        Testa a extração de metadados do PDF.
        
        Returns:
            bool: True se o teste passou, False caso contrário
        """
        test_name = "pdf_metadata"
        self.log_test_start(test_name)
        
        try:
            # Carrega o PDF e verifica metadados
            pages = self.processor.process_pdf()
            
            # Verifica se há metadados na primeira página
            if not pages or 'metadata' not in pages[0]:
                raise ValueError("Metadados não encontrados")
                
            metadata = pages[0]['metadata']
            self.logger.info("Metadados encontrados:")
            for key, value in metadata.items():
                self.logger.info(f"{key}: {value}")
                
            self.log_test_result(test_name, True)
            return True
            
        except Exception as e:
            self.log_test_result(test_name, False, e)
            return False
            
    def cleanup(self) -> None:
        """Remove arquivos temporários de teste."""
        if self.test_pdf_path.exists():
            self.test_pdf_path.unlink()
        if self.test_data_dir.exists():
            self.test_data_dir.rmdir()
            
    def run(self) -> bool:
        """
        Executa todos os testes do DocumentProcessor.
        
        Returns:
            bool: True se todos os testes passaram, False caso contrário
        """
        try:
            self.setup()
            
            # Executa os testes
            tests_passed = all([
                self.test_pdf_exists(),
                self.test_pdf_loading(),
                self.test_pdf_metadata()
            ])
            
            self.cleanup()
            return tests_passed
            
        except Exception as e:
            self.logger.error(f"Erro inesperado: {str(e)}")
            self.logger.debug("Detalhes do erro:", exc_info=True)
            return False

def main():
    """Função principal para executar os testes."""
    test = DocumentProcessorTest()
    success = test.run()
    return 0 if success else 1

if __name__ == "__main__":
    exit(main()) 