from pathlib import Path
from typing import List, Dict, Any
import json
import time

from ..core.base_rag import RAGConfig
from ..core.faiss_rag import FAISSRAGSystem
from ..models.groq_handler import GroqHandler
from ..utils.logging_config import setup_logger

logger = setup_logger('rag_example')

def load_sample_documents() -> List[Dict[str, str]]:
    """Carrega documentos de exemplo."""
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
    return documents

def test_rag_system():
    """Testa o sistema RAG."""
    try:
        # Configura o sistema RAG
        config = RAGConfig(
            embeddings_model="sentence-transformers/all-MiniLM-L6-v2",
            chunk_size=1000,
            chunk_overlap=200,
            max_documents=3,
            similarity_threshold=0.7,
            cache_enabled=True,
            cache_dir="cache/rag"
        )
        
        # Inicializa o sistema
        logger.info("Inicializando sistema RAG...")
        rag = FAISSRAGSystem()
        rag.initialize(config)
        
        # Carrega documentos de exemplo
        documents = load_sample_documents()
        logger.info(f"Carregando {len(documents)} documentos de exemplo")
        rag.add_documents(documents)
        
        # Testa algumas perguntas
        perguntas = [
            "Como funciona a técnica Pomodoro?",
            "O que é GTD e como usar?",
            "Como priorizar minhas tarefas?",
            "Quais são as melhores técnicas de produtividade?"
        ]
        
        logger.info("\n=== Iniciando testes de perguntas ===")
        
        resultados = []
        for pergunta in perguntas:
            logger.info(f"\nProcessando pergunta: {pergunta}")
            
            # Processa a pergunta
            response = rag.query(pergunta)
            
            # Registra o resultado
            resultado = {
                "pergunta": pergunta,
                "resposta": response.answer,
                "fontes": response.sources,
                "confianca": response.confidence,
                "tempo": response.processing_time
            }
            resultados.append(resultado)
            
            # Mostra o resultado
            logger.info("=== Resultado ===")
            logger.info(f"Pergunta: {pergunta}")
            logger.info(f"Resposta: {response.answer}")
            logger.info(f"Fontes: {', '.join(response.sources)}")
            logger.info(f"Confiança: {response.confidence:.2f}")
            logger.info(f"Tempo: {response.processing_time:.2f}s")
            
        # Salva os resultados
        output_dir = Path("examples/output")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / "resultados_rag.json", "w") as f:
            json.dump(resultados, f, indent=2, ensure_ascii=False)
            
        # Mostra estatísticas
        stats = rag.get_stats()
        logger.info("\n=== Estatísticas do Sistema ===")
        for key, value in stats.items():
            logger.info(f"{key}: {value}")
            
        # Limpa o sistema
        rag.clear()
        logger.info("\nTeste concluído com sucesso!")
        
        return True
        
    except Exception as e:
        logger.error(f"Erro durante o teste: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_rag_system()
    exit(0 if success else 1) 