import logging
import sys
from logging.handlers import RotatingFileHandler

def setup_logger(name: str, log_file: str = 'app.log', level=logging.INFO) -> logging.Logger:
    """
    Configura um logger com handlers para arquivo e console.
    
    Args:
        name: Nome do logger
        log_file: Caminho do arquivo de log
        level: Nível de logging
        
    Returns:
        Logger configurado
    """
    # Cria o logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Formata as mensagens de log
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Handler para arquivo com rotação
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=1024 * 1024,  # 1MB
        backupCount=5
    )
    file_handler.setFormatter(formatter)
    
    # Handler para console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # Adiciona os handlers ao logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger 