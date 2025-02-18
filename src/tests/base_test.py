# Standard library imports
import os
import logging
from pathlib import Path
import textwrap
from typing import Optional

# Local imports
from ..utils.logging_config import setup_logger

class BaseTest:
    """Classe base para todos os testes."""
    
    def __init__(self, test_name: str):
        """
        Inicializa o teste com configuração de logging.
        
        Args:
            test_name: Nome do teste para identificação nos logs
        """
        self.logger = setup_logger(f'test_{test_name}')
        self.test_dir = self._setup_test_dir()
        
    def _setup_test_dir(self) -> Path:
        """
        Configura o diretório de testes.
        
        Returns:
            Path: Caminho para o diretório de testes
        """
        # Obtém o diretório do projeto
        project_root = Path(__file__).parent.parent.parent
        
        # Cria diretório de testes se não existir
        test_dir = project_root / 'test_data'
        test_dir.mkdir(exist_ok=True)
        
        return test_dir
        
    def get_test_file_path(self, filename: str) -> Path:
        """
        Retorna o caminho completo para um arquivo de teste.
        
        Args:
            filename: Nome do arquivo
            
        Returns:
            Path: Caminho completo para o arquivo
        """
        return self.test_dir / filename
        
    def log_test_start(self, test_name: str) -> None:
        """
        Registra o início de um teste.
        
        Args:
            test_name: Nome do teste
        """
        self.logger.info(textwrap.dedent(f"""
            {'='*50}
            Iniciando teste: {test_name}
            {'='*50}
        """).strip())
        
    def log_test_result(self, test_name: str, success: bool, error: Optional[Exception] = None) -> None:
        """
        Registra o resultado de um teste.
        
        Args:
            test_name: Nome do teste
            success: Se o teste foi bem-sucedido
            error: Exceção opcional se o teste falhou
        """
        if success:
            self.logger.info(textwrap.dedent(f"""
                {'='*50}
                Teste concluído com sucesso: {test_name}
                {'='*50}
            """).strip())
        else:
            self.logger.error(textwrap.dedent(f"""
                {'='*50}
                Teste falhou: {test_name}
                Erro: {str(error) if error else 'Erro desconhecido'}
                {'='*50}
            """).strip())
            if error:
                self.logger.debug("Detalhes do erro:", exc_info=True)
                
    def run(self) -> bool:
        """
        Método abstrato para executar o teste.
        Deve ser implementado pelas classes filhas.
        
        Returns:
            bool: True se o teste passou, False caso contrário
        """
        raise NotImplementedError("Método run() deve ser implementado pela classe filha") 