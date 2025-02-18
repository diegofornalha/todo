import os
import sys
from typing import Dict, TypedDict, Annotated, Sequence
from typing_extensions import TypedDict

# Adiciona o diretório raiz ao PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph.graph import Graph, StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from src.config.langsmith_config import configure_langsmith
from dotenv import load_dotenv
from src.core.retrieval_qa import RetrievalQA
from src.models.groq_handler import GroqHandler

# Define os tipos para o estado do grafo
class GraphState(TypedDict):
    messages: Sequence[Dict]
    query_type: str
    retrieval_qa: RetrievalQA | None
    stats: Dict[str, Any]

def create_rag_graph():
    """
    Cria um grafo para processar consultas RAG em tempo real.
    Este grafo permite:
    1. Análise semântica de perguntas
    2. Recuperação de documentos relevantes
    3. Geração de respostas contextualizadas
    4. Visualização de métricas em tempo real
    5. Rastreamento de fontes consultadas
    """
    # Carrega variáveis de ambiente
    load_dotenv()
    
    # Configura o LangSmith para rastreamento
    configure_langsmith()
    
    # Configura o modelo Groq
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY não encontrada")
        
    model_name = os.getenv("MODEL_NAME", "mixtral-8x7b-32768")
    llm = GroqHandler(model_name=model_name)
    llm.initialize()
    
    # Cria o grafo de estado
    workflow = StateGraph(GraphState)
    
    # Inicializa o sistema RAG
    retrieval_qa = RetrievalQA(llm=llm)
    
    # Define os nós do grafo
    
    # 1. Nó para analisar o tipo de consulta
    def analyze_query(state: GraphState) -> Dict:
        """Analisa a mensagem do usuário e determina o tipo de consulta."""
        messages = state["messages"]
        last_message = messages[-1]["content"].lower()
        
        system_prompt = """Você é um assistente especializado em análise de consultas.
        
        Analise a mensagem do usuário e retorne APENAS UM dos seguintes tipos (sem explicação adicional):
        - factual_query (para perguntas sobre fatos específicos)
        - conceptual_query (para perguntas sobre conceitos)
        - procedural_query (para perguntas sobre como fazer algo)
        - analytical_query (para perguntas que requerem análise)
        
        IMPORTANTE: Retorne APENAS o tipo, sem nenhum texto adicional."""
        
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=last_message)
        ])
        
        query_type = response.content.strip().lower()
        state["query_type"] = query_type
        
        return {"next": "process_query"}
    
    # 2. Nó para processar a consulta
    def process_query(state: GraphState) -> GraphState:
        """Processa a consulta usando o sistema RAG."""
        messages = state["messages"]
        last_message = messages[-1]["content"]
        query_type = state["query_type"]
        
        # Inicializa o RAG se necessário
        if "retrieval_qa" not in state or state["retrieval_qa"] is None:
            state["retrieval_qa"] = retrieval_qa
            
        try:
            # Processa a pergunta com o RAG
            result = state["retrieval_qa"].query(
                last_message,
                include_sources=True
            )
            
            # Atualiza estatísticas
            if "stats" not in state:
                state["stats"] = {"queries": 0, "successful_queries": 0}
            state["stats"]["queries"] += 1
            
            # Formata a resposta com base no tipo de consulta
            response = f"""<think>
Tipo de consulta: {query_type}
Fontes consultadas: {', '.join(result.get('sources', ['Nenhuma fonte específica']))}
Confiança: {result.get('metadata', {}).get('confidence', 'N/A')}
</think>

{result['resposta']}"""
            
            state["messages"].append({
                "role": "assistant",
                "content": response
            })
            state["stats"]["successful_queries"] += 1
            
        except Exception as e:
            state["messages"].append({
                "role": "assistant",
                "content": f"Erro ao processar a consulta: {str(e)}"
            })
            
        return state
    
    # Adiciona os nós ao grafo
    workflow.add_node("analyze_query", analyze_query)
    workflow.add_node("process_query", process_query)
    
    # Define o ponto de entrada
    workflow.set_entry_point("analyze_query")
    
    # Define as transições
    workflow.add_edge("analyze_query", "process_query")
    workflow.add_edge("process_query", END)
    
    # Compila o grafo
    app = workflow.compile()
    
    return app

if __name__ == "__main__":
    # Testa o grafo
    graph = create_rag_graph()
    
    # Exemplo de conversa
    state = {
        "messages": [
            {"role": "user", "content": "Qual é a capital do Brasil?"}
        ],
        "query_type": "",
        "retrieval_qa": None,
        "stats": {}
    }
    
    result = graph.invoke(state)
    print("\nResposta do assistente:")
    print(result["messages"][-1]["content"]) 