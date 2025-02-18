from ..core.retrieval_qa import RetrievalQA
from ..models.groq_handler import GroqHandler
from ..utils.logging_config import setup_logger
from ..todo_graph import create_todo_graph
import os

logger = setup_logger('rag_example')

def main():
    """
    Exemplo de uso do sistema RAG integrado ao LangGraph.
    """
    try:
        # Inicializa o grafo
        logger.info("Inicializando o grafo...")
        graph = create_todo_graph()
        
        # Prepara alguns documentos de exemplo sobre produtividade
        logger.info("Preparando documentos de exemplo...")
        documents = [
            {
                "content": """A técnica Pomodoro é um método de gerenciamento de tempo desenvolvido por Francesco Cirillo.
                Consiste em dividir o trabalho em blocos de 25 minutos, chamados "pomodoros", seguidos por pausas curtas.
                Após 4 pomodoros, faz-se uma pausa mais longa. Esta técnica ajuda a manter o foco e evitar a procrastinação.""",
                "source": "tecnicas_produtividade.pdf"
            },
            {
                "content": """GTD (Getting Things Done) é um método de produtividade criado por David Allen.
                Os cinco passos básicos são: Capturar, Clarificar, Organizar, Refletir e Engajar.
                O objetivo é tirar as tarefas da mente e colocá-las em um sistema confiável.""",
                "source": "gtd_method.pdf"
            },
            {
                "content": """A matriz de Eisenhower é uma ferramenta de priorização que divide as tarefas em quatro quadrantes:
                1. Urgente e Importante
                2. Não Urgente mas Importante
                3. Urgente mas Não Importante
                4. Nem Urgente Nem Importante
                Isso ajuda a identificar o que realmente precisa ser feito primeiro.""",
                "source": "matriz_eisenhower.pdf"
            }
        ]
        
        # Inicializa o estado
        state = {
            "messages": [],
            "next_step": "",
            "task_list": [],
            "current_task": None,
            "retrieval_qa": None
        }
        
        # Adiciona os documentos ao RAG
        logger.info("Adicionando documentos ao sistema RAG...")
        state["retrieval_qa"] = RetrievalQA(GroqHandler())
        state["retrieval_qa"].add_documents(documents)
        
        # Testa algumas perguntas
        perguntas = [
            "Como funciona a técnica Pomodoro?",
            "O que é GTD e como usar?",
            "Como priorizar minhas tarefas?",
            "Quais são as melhores técnicas de produtividade?"
        ]
        
        logger.info("\n=== Iniciando testes de perguntas ===")
        
        for pergunta in perguntas:
            logger.info(f"\nTestando pergunta: {pergunta}")
            
            # Adiciona a pergunta ao estado
            state["messages"] = [{"role": "user", "content": pergunta}]
            
            # Processa com o grafo
            result = graph.invoke(state)
            
            # Mostra a resposta
            logger.info("=== Resposta ===")
            logger.info(f"Pergunta: {pergunta}")
            logger.info(f"Resposta: {result['messages'][-1]['content']}")
            
    except Exception as e:
        logger.error(f"Erro durante o teste: {str(e)}")
        return 1
        
    return 0

if __name__ == "__main__":
    main() 