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

# Define os tipos para o estado do grafo
class GraphState(TypedDict):
    messages: Sequence[Dict]
    next_step: str
    task_list: list[str]
    current_task: str | None

def create_todo_graph():
    """
    Cria um grafo para gerenciar tarefas TODO.
    Este grafo permite:
    1. Adicionar novas tarefas
    2. Marcar tarefas como concluídas
    3. Listar tarefas pendentes
    4. Priorizar tarefas
    """
    # Carrega variáveis de ambiente
    load_dotenv()
    
    # Configura o LangSmith para rastreamento
    configure_langsmith()
    
    # Configura o modelo Groq
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY não encontrada")
        
    model_name = os.getenv("MODEL_NAME", "deepseek-r1-distill-llama-70b")
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name=model_name,
        temperature=0
    )
    
    # Cria o grafo de estado
    workflow = StateGraph(GraphState)
    
    # Define os nós do grafo
    
    # 1. Nó para processar comandos
    def process_command(state: GraphState) -> Dict:
        """Processa o comando do usuário e determina o próximo passo."""
        messages = state["messages"]
        last_message = messages[-1]["content"].lower()
        
        if "adicionar" in last_message or "nova tarefa" in last_message:
            return {"next": "add_task"}
        elif "concluir" in last_message or "completar" in last_message:
            return {"next": "complete_task"}
        elif "listar" in last_message or "mostrar" in last_message:
            return {"next": "list_tasks"}
        else:
            return {"next": "list_tasks"}  # Default para listar tarefas
    
    # 2. Nó para adicionar tarefa
    def add_task(state: GraphState) -> GraphState:
        """Adiciona uma nova tarefa à lista."""
        messages = state["messages"]
        task = messages[-1]["content"]
        
        if "task_list" not in state:
            state["task_list"] = []
            
        state["task_list"].append(task)
        state["messages"].append({
            "role": "assistant",
            "content": f"Tarefa adicionada com sucesso: {task}"
        })
        
        return state
    
    # 3. Nó para completar tarefa
    def complete_task(state: GraphState) -> GraphState:
        """Marca uma tarefa como concluída."""
        messages = state["messages"]
        task_index = int(messages[-1]["content"]) - 1
        
        if 0 <= task_index < len(state["task_list"]):
            completed_task = state["task_list"].pop(task_index)
            state["messages"].append({
                "role": "assistant",
                "content": f"Tarefa concluída: {completed_task}"
            })
        else:
            state["messages"].append({
                "role": "assistant",
                "content": "Índice de tarefa inválido"
            })
            
        return state
    
    # 4. Nó para listar tarefas
    def list_tasks(state: GraphState) -> GraphState:
        """Lista todas as tarefas pendentes."""
        tasks = state.get("task_list", [])
        
        if not tasks:
            response = "Não há tarefas pendentes."
        else:
            task_list = "\n".join(f"{i+1}. {task}" for i, task in enumerate(tasks))
            response = f"Tarefas pendentes:\n{task_list}"
            
        state["messages"].append({
            "role": "assistant",
            "content": response
        })
        
        return state
    
    # Adiciona os nós ao grafo
    workflow.add_node("process_command", process_command)
    workflow.add_node("add_task", add_task)
    workflow.add_node("complete_task", complete_task)
    workflow.add_node("list_tasks", list_tasks)
    
    # Define as condições de entrada
    workflow.set_entry_point("process_command")
    
    # Define as transições entre os nós
    workflow.add_conditional_edges(
        "process_command",
        lambda x: x["next"],
        {
            "add_task": "add_task",
            "complete_task": "complete_task",
            "list_tasks": "list_tasks"
        }
    )
    
    # Define os nós finais
    workflow.add_edge("add_task", "list_tasks")
    workflow.add_edge("complete_task", "list_tasks")
    workflow.add_edge("list_tasks", END)
    
    # Compila o grafo
    app = workflow.compile()
    
    return app

def main():
    """Função principal para testar o grafo TODO."""
    try:
        # Cria o grafo
        todo_graph = create_todo_graph()
        print("Grafo TODO criado com sucesso!")
        
        # Teste 1: Adicionar uma tarefa
        print("\n=== Teste 1: Adicionar Tarefa ===")
        state1 = {
            "messages": [{"role": "user", "content": "Adicionar tarefa: Implementar autenticação"}],
            "next_step": "",
            "task_list": [],
            "current_task": None
        }
        
        final_state1 = todo_graph.invoke(state1)
        print("\nResultado - Adicionar Tarefa:")
        for message in final_state1["messages"]:
            if message["role"] == "assistant":
                print(f"Assistente: {message['content']}")
            else:
                print(f"Usuário: {message['content']}")
                
        # Teste 2: Completar uma tarefa
        print("\n=== Teste 2: Completar Tarefa ===")
        state2 = {
            "messages": [{"role": "user", "content": "1"}],  # Completar primeira tarefa
            "next_step": "",
            "task_list": ["Estudar LangGraph", "Implementar TODO app"],
            "current_task": None
        }
        
        final_state2 = todo_graph.invoke(state2)
        print("\nResultado - Completar Tarefa:")
        for message in final_state2["messages"]:
            if message["role"] == "assistant":
                print(f"Assistente: {message['content']}")
            else:
                print(f"Usuário: {message['content']}")
                
        # Teste 3: Listar tarefas
        print("\n=== Teste 3: Listar Tarefas ===")
        state3 = {
            "messages": [{"role": "user", "content": "Listar tarefas"}],
            "next_step": "",
            "task_list": ["Implementar TODO app", "Testar funcionalidades", "Implementar autenticação"],
            "current_task": None
        }
        
        final_state3 = todo_graph.invoke(state3)
        print("\nResultado - Listar Tarefas:")
        for message in final_state3["messages"]:
            if message["role"] == "assistant":
                print(f"Assistente: {message['content']}")
            else:
                print(f"Usuário: {message['content']}")
                
    except Exception as e:
        print(f"Erro durante a execução: {e}")

if __name__ == "__main__":
    main() 