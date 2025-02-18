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
    next_step: str
    task_list: list[str]
    current_task: str | None
    retrieval_qa: RetrievalQA | None  # Novo campo para o sistema RAG

def create_todo_graph():
    """
    Cria um grafo para gerenciar tarefas TODO de forma conversacional.
    Este grafo permite:
    1. Conversar sobre prioridades e objetivos
    2. Adicionar tarefas com contexto
    3. Priorizar tarefas automaticamente
    4. Sugerir próximos passos
    5. Manter o foco nas tarefas importantes
    6. Consultar documentos e conhecimento base (RAG)
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
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name=model_name,
        temperature=0.7
    )
    
    # Cria o grafo de estado
    workflow = StateGraph(GraphState)
    
    # Inicializa o sistema RAG
    retrieval_qa = RetrievalQA(llm)
    
    # Define os nós do grafo
    
    # 1. Nó para processar a intenção do usuário
    def process_intent(state: GraphState) -> Dict:
        """Analisa a mensagem do usuário e determina a próxima ação."""
        messages = state["messages"]
        last_message = messages[-1]["content"].lower()
        
        # Verifica se é uma saudação inicial
        greetings = {"oi", "olá", "ola", "hi", "hello"}
        is_greeting = any(greeting in last_message for greeting in greetings)
        
        # Se for uma saudação e não tivermos outras mensagens, vamos direto para o chat
        if is_greeting and len(messages) <= 2:
            return {"next": "chat"}
        
        system_prompt = """Você é um assistente especializado em gerenciamento de tarefas e prioridades.
        
        Analise a mensagem do usuário e retorne APENAS UMA das seguintes ações (sem explicação adicional):
        - add_task
        - complete_task
        - prioritize
        - rag_query
        - chat
        
        Escolha rag_query se:
        - O usuário faz uma pergunta direta
        - Busca informações específicas
        - Quer saber mais sobre um tópico
        
        Por exemplo:
        - "Como faço X?" -> "rag_query"
        - "O que é Y?" -> "rag_query"
        - "Pode me explicar Z?" -> "rag_query"
        
        IMPORTANTE: Retorne APENAS a ação, sem nenhum texto adicional."""
        
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=last_message)
        ])
        
        # Limpa a resposta para garantir que só temos a ação
        action = response.content.strip().lower()
        # Remove qualquer texto que não seja uma das ações válidas
        valid_actions = {"add_task", "complete_task", "prioritize", "rag_query", "chat"}
        if action not in valid_actions:
            action = "chat"  # Default para chat se a ação não for válida
            
        return {"next": action}
    
    # 2. Nó para adicionar tarefa
    def add_task(state: GraphState) -> GraphState:
        """Adiciona uma nova tarefa com contexto e prioridade."""
        messages = state["messages"]
        last_message = messages[-1]["content"]
        
        system_prompt = """Extraia a tarefa da mensagem do usuário e sugira uma prioridade baseada no contexto.
        Responda de forma natural e conversacional, explicando seu raciocínio."""
        
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=last_message)
        ])
        
        task = response.content
        if "task_list" not in state:
            state["task_list"] = []
            
        state["task_list"].append(task)
        state["messages"].append({
            "role": "assistant",
            "content": f"Entendi! Adicionei esta tarefa à sua lista. Baseado no contexto, sugiro priorizá-la porque {response.content}"
        })
        
        return state
    
    # 3. Nó para completar tarefa
    def complete_task(state: GraphState) -> GraphState:
        """Marca uma tarefa como concluída e sugere próximos passos."""
        messages = state["messages"]
        last_message = messages[-1]["content"]
        
        system_prompt = """Identifique qual tarefa o usuário quer completar e sugira o próximo passo mais apropriado."""
        
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=last_message)
        ])
        
        try:
            task_index = int(response.content) - 1
            if 0 <= task_index < len(state["task_list"]):
                completed_task = state["task_list"].pop(task_index)
                state["messages"].append({
                    "role": "assistant",
                    "content": f"Ótimo trabalho completando '{completed_task}'! Sugiro focar agora em: {response.content}"
                })
            else:
                state["messages"].append({
                    "role": "assistant",
                    "content": "Desculpe, não encontrei essa tarefa na lista. Pode me dizer qual tarefa você quer completar?"
                })
        except:
            state["messages"].append({
                "role": "assistant",
                "content": "Não entendi qual tarefa você quer completar. Pode me dizer o número da tarefa?"
            })
            
        return state
    
    # 4. Nó para listar e priorizar tarefas
    def prioritize(state: GraphState) -> GraphState:
        """Lista tarefas e sugere priorização."""
        tasks = state.get("task_list", [])
        
        if not tasks:
            state["messages"].append({
                "role": "assistant",
                "content": "Você não tem tarefas pendentes. Quer me contar sobre seus objetivos para eu ajudar a criar um plano?"
            })
            return state
            
        system_prompt = """Analise as tarefas e sugira uma ordem de prioridade baseada em:
        - Urgência
        - Importância
        - Dependências
        - Impacto
        Explique seu raciocínio de forma clara e motivadora."""
        
        task_list = "\n".join(f"{i+1}. {task}" for i, task in enumerate(tasks))
        
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Tarefas atuais:\n{task_list}")
        ])
        
        state["messages"].append({
            "role": "assistant",
            "content": response.content
        })
        
        return state
    
    # 5. Nó para chat geral
    def chat(state: GraphState) -> GraphState:
        """Mantém uma conversa natural sobre objetivos e produtividade."""
        messages = state["messages"]
        last_message = messages[-1]["content"].lower()
        
        # Verifica se é uma saudação inicial
        greetings = {"oi", "olá", "ola", "hi", "hello"}
        is_greeting = any(greeting in last_message for greeting in greetings)
        
        # Só mostra a mensagem de boas-vindas se for realmente a primeira interação
        if is_greeting and len(messages) == 1:
            state["messages"].append({
                "role": "assistant",
                "content": "Olá! Que bom ter você aqui! Como posso ajudar com suas tarefas e prioridades hoje? Você pode:\n\n" +
                          "1. Me contar sobre seus objetivos\n" +
                          "2. Adicionar novas tarefas\n" +
                          "3. Pedir ajuda para priorizar suas atividades\n" +
                          "4. Marcar tarefas como concluídas"
            })
            return state
        
        system_prompt = """Você é um assistente amigável e motivador, especializado em produtividade e gestão de tempo.
        
        Ajude o usuário a:
        - Clarificar seus objetivos
        - Desenvolver bons hábitos
        - Manter o foco
        - Superar procrastinação
        
        IMPORTANTE: 
        1. Analise o contexto da conversa e identifique objetivos ou tarefas implícitas
        2. Se o usuário mencionar qualquer atividade, objetivo ou desafio, sugira transformá-lo em uma tarefa concreta
        3. Sempre conclua sua resposta com uma ação clara ou próximo passo
        4. Se não houver tarefas na lista, sugira criar uma baseada no contexto da conversa
        
        Formato da resposta:
        1. Responda de forma empática e natural
        2. Identifique objetivos ou desafios mencionados
        3. Sugira uma tarefa específica (se apropriado)
        4. Conclua com um próximo passo claro
        
        Use <think> tags para mostrar seu processo de pensamento."""
        
        chat_history = [
            SystemMessage(content=system_prompt)
        ] + [
            HumanMessage(content=m["content"]) if m["role"] == "user" else 
            AIMessage(content=m["content"]) 
            for m in messages[-3:]  # Últimas 3 mensagens para contexto
        ]
        
        response = llm.invoke(chat_history)
        
        # Processa a resposta
        content = response.content
        
        # Se não houver tarefas na lista, tenta extrair uma da conversa
        if not state.get("task_list"):
            add_task_prompt = """Analise a conversa e sugira uma tarefa concreta baseada no contexto.
            Se não houver contexto suficiente, sugira uma tarefa inicial para ajudar o usuário a começar.
            Retorne apenas a tarefa, sem explicações adicionais."""
            
            task_response = llm.invoke([
                SystemMessage(content=add_task_prompt),
                HumanMessage(content=str(messages[-3:]))  # Últimas 3 mensagens como contexto
            ])
            
            # Adiciona a tarefa à lista
            task = task_response.content.strip()
            if task:
                state["task_list"].append(task)
                content += f"\n\nCriei uma primeira tarefa para você começar: '{task}'"
        
        state["messages"].append({
            "role": "assistant",
            "content": content
        })
        
        return state
    
    # 6. Nó para consultas RAG
    def rag_query(state: GraphState) -> GraphState:
        """Processa consultas usando o sistema RAG."""
        messages = state["messages"]
        last_message = messages[-1]["content"]
        
        # Inicializa o RAG se necessário
        if "retrieval_qa" not in state or state["retrieval_qa"] is None:
            state["retrieval_qa"] = retrieval_qa
            
        system_prompt = """Você é um assistente especializado em responder perguntas usando uma base de conhecimento.
        
        Analise a pergunta do usuário e:
        1. Use o sistema RAG para buscar informações relevantes
        2. Formule uma resposta clara e objetiva
        3. Cite as fontes consultadas
        4. Se identificar uma tarefa implícita, sugira adicioná-la à lista
        
        Use tags <think> para mostrar seu processo de pensamento."""
        
        try:
            # Processa a pergunta com o RAG
            result = state["retrieval_qa"].query(
                last_message,
                include_sources=True
            )
            
            # Formata a resposta
            response = f"""<think>
Analisando a pergunta usando a base de conhecimento...
Fontes consultadas: {', '.join(result.get('sources', ['Nenhuma fonte específica']))}
</think>

{result['resposta']}"""

            # Se encontrou uma tarefa implícita, sugere adicioná-la
            if "task" in result.get('metadata', {}):
                response += f"\n\nIdentifiquei uma possível tarefa: {result['metadata']['task']}\nGostaria que eu a adicionasse à sua lista?"
                
            state["messages"].append({
                "role": "assistant",
                "content": response
            })
            
        except Exception as e:
            state["messages"].append({
                "role": "assistant",
                "content": f"Desculpe, tive um problema ao consultar a base de conhecimento: {str(e)}"
            })
            
        return state
    
    # Adiciona os nós ao grafo
    workflow.add_node("process_intent", process_intent)
    workflow.add_node("add_task", add_task)
    workflow.add_node("complete_task", complete_task)
    workflow.add_node("prioritize", prioritize)
    workflow.add_node("chat", chat)
    workflow.add_node("rag_query", rag_query)
    
    # Define o ponto de entrada
    workflow.set_entry_point("process_intent")
    
    # Define as transições entre os nós
    workflow.add_conditional_edges(
        "process_intent",
        lambda x: x["next"],
        {
            "add_task": "add_task",
            "complete_task": "complete_task",
            "prioritize": "prioritize",
            "rag_query": "rag_query",
            "chat": "chat"
        }
    )
    
    # Define os nós finais
    workflow.add_edge("add_task", END)
    workflow.add_edge("complete_task", END)
    workflow.add_edge("prioritize", END)
    workflow.add_edge("chat", END)
    workflow.add_edge("rag_query", END)
    
    # Compila o grafo
    app = workflow.compile()
    
    return app

if __name__ == "__main__":
    # Testa o grafo
    graph = create_todo_graph()
    
    # Exemplo de conversa
    state = {
        "messages": [
            {"role": "user", "content": "Preciso organizar melhor meu tempo"}
        ],
        "next_step": "",
        "task_list": [],
        "current_task": None
    }
    
    result = graph.invoke(state)
    print("\nResposta do assistente:")
    print(result["messages"][-1]["content"]) 