import logging
import os
from logging.handlers import RotatingFileHandler

def setup_logger(name: str, log_file: str, level=logging.INFO) -> logging.Logger:
    """
    Configura um logger com handlers para arquivo e console.
    
    Args:
        name: Nome do logger
        log_file: Caminho do arquivo de log
        level: Nível de logging
        
    Returns:
        Logger configurado
    """
    # Cria o diretório de logs se não existir
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Configura o logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove handlers existentes para evitar duplicação
    if logger.handlers:
        logger.handlers.clear()
    
    # Formata as mensagens
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Handler para arquivo com rotação
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Handler para console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger 