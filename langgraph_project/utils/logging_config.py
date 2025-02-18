import logging
import os

def setup_logger(name: str) -> logging.Logger:
    """
    Configura um logger com formatação padrão.
    
    Args:
        name: Nome do logger
        
    Returns:
        Logger configurado
    """
    # Cria o logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Cria o handler para console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Define o formato
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    
    # Adiciona o handler ao logger
    logger.addHandler(console_handler)
    
    return logger 