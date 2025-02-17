from ..core.retrieval_qa import RetrievalQA
from ..models.groq_handler import GroqHandler
from ..utils.logging_config import setup_logger
import os

logger = setup_logger('qa_example')

def main():
    """
    Exemplo de uso do sistema de RetrievalQA.
    """
    try:
        # Inicializa o handler do Groq
        logger.info("Inicializando handler do Groq...")
        groq_handler = GroqHandler()
        groq_handler.initialize()
        
        # Inicializa o sistema de QA
        logger.info("Inicializando sistema de RetrievalQA...")
        qa_system = RetrievalQA(groq_handler)
        
        # Adiciona alguns documentos de exemplo
        logger.info("Adicionando documentos de exemplo...")
        documents = [
            {
                "content": """Python é uma linguagem de programação de alto nível, interpretada e de propósito geral.
                Foi criada por Guido van Rossum e lançada em 1991. Python é conhecida por sua sintaxe clara e legível,
                que enfatiza a legibilidade do código. É uma linguagem multiparadigma, suportando programação orientada
                a objetos, imperativa e funcional.""",
                "source": "python_overview.pdf"
            },
            {
                "content": """Python é amplamente utilizada em várias áreas, incluindo desenvolvimento web,
                ciência de dados, inteligência artificial, automação, scripting e mais. Possui uma extensa
                biblioteca padrão e um grande ecossistema de pacotes de terceiros. A filosofia da linguagem
                é expressa no "Zen do Python", que inclui princípios como "Simples é melhor que complexo".""",
                "source": "python_applications.pdf"
            },
            {
                "content": """O gerenciamento de pacotes em Python é feito principalmente através do pip,
                que é o instalador de pacotes padrão. Os pacotes são normalmente hospedados no Python
                Package Index (PyPI). Ambientes virtuais são usados para isolar dependências de projetos,
                com ferramentas como venv e virtualenv.""",
                "source": "python_packaging.pdf"
            }
        ]
        
        qa_system.add_documents(documents)
        
        # Salva o vectorstore para uso futuro
        logger.info("Salvando vectorstore...")
        os.makedirs("data", exist_ok=True)
        qa_system.save_vectorstore("data/vectorstore")
        
        # Testa algumas perguntas
        questions = [
            "O que é Python e quais são suas principais características?",
            "Em quais áreas Python é utilizada?",
            "Como é feito o gerenciamento de pacotes em Python?",
            "Quem criou o Python e quando?",
            "O que é o Zen do Python?"
        ]
        
        logger.info("\n=== Iniciando testes de perguntas ===")
        
        for question in questions:
            logger.info(f"\nTestando pergunta: {question}")
            result = qa_system.query(question, include_sources=True)
            
            logger.info("=== Resultado ===")
            logger.info(f"Pergunta: {result['pergunta']}")
            logger.info(f"Resposta: {result['resposta']}")
            if 'sources' in result:
                logger.info(f"Fontes consultadas: {', '.join(result['sources'])}")
            logger.info(f"Status: {result.get('status', 'sucesso')}")
            
    except Exception as e:
        logger.error(f"Erro durante o teste: {str(e)}")
        return 1
        
    return 0

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code) 