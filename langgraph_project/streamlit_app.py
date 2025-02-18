import streamlit as st
from todo_graph import create_todo_graph
import json
import re

st.set_page_config(
    page_title="Assistente de Prioridades TODO",
    page_icon="🎯",
    layout="wide"
)

# Função para extrair pensamento e resposta
def extract_thought_and_answer(text):
    # Procura por conteúdo entre tags <think> e </think>
    thought_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    
    if thought_match:
        thought = thought_match.group(1).strip()
        # Remove a parte do pensamento do texto original para obter apenas a resposta
        answer = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
        return thought, answer
    else:
        return None, text.strip()

# Inicializa o estado da sessão
if 'messages' not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Olá! Eu sou seu assistente de prioridades. Posso ajudar você a:\n\n" + 
         "- Organizar suas tarefas por prioridade\n" +
         "- Sugerir a melhor ordem para executá-las\n" +
         "- Manter você focado no que é importante\n\n" +
         "Como posso ajudar você hoje?"}
    ]
    
if 'task_list' not in st.session_state:
    st.session_state.task_list = []

# Inicializa o grafo
try:
    graph = create_todo_graph()
except Exception as e:
    st.error(f"Erro ao inicializar o assistente: {str(e)}")
    st.stop()

# Layout principal
st.title("🎯 Assistente de Prioridades")

# Sidebar para configurações e lista de tarefas
with st.sidebar:
    st.subheader("📋 Tarefas Prioritárias")
    
    if st.session_state.task_list:
        for i, task in enumerate(st.session_state.task_list):
            col_task, col_done = st.columns([4,1])
            with col_task:
                st.write(f"{i+1}. {task}")
            with col_done:
                if st.button("✓", key=f"complete_{i}", help="Marcar como concluída"):
                    input_data = {
                        "messages": [{"role": "user", "content": f"Marcar tarefa {i+1} como concluída"}],
                        "next_step": "",
                        "task_list": st.session_state.task_list,
                        "current_task": None
                    }
                    try:
                        result = graph.invoke(input_data)
                        st.session_state.task_list = result["task_list"]
                        st.success(f"Tarefa '{task}' concluída!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Erro ao concluir tarefa: {str(e)}")
    else:
        st.info("Nenhuma tarefa pendente")
    
    # Configurações
    st.subheader("⚙️ Configurações")
    show_thoughts = st.toggle("Mostrar processo de pensamento", value=False)

# Container principal para o chat
chat_container = st.container()

# Área de input sempre no final
with st.container():
    # Linha horizontal para separar
    st.markdown("---")
    
    # Input do usuário
    if prompt := st.chat_input("Digite sua mensagem...", key="chat_input"):
        # Adiciona mensagem do usuário
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Prepara input para o grafo
        input_data = {
            "messages": st.session_state.messages,
            "next_step": "",
            "task_list": st.session_state.task_list,
            "current_task": None
        }
        
        # Processa com o grafo
        try:
            result = graph.invoke(input_data)
            st.session_state.task_list = result["task_list"]
            
            # Extrai pensamento e resposta
            thought, answer = extract_thought_and_answer(result.get("messages", [])[-1]["content"])
            
            # Adiciona resposta do assistente
            st.session_state.messages.append({"role": "assistant", "content": answer})
            
            # Se houver pensamento e a opção estiver ativada, adiciona como mensagem especial
            if thought and show_thoughts:
                st.session_state.messages.append({"role": "system", "content": thought})
                
            st.rerun()
        except Exception as e:
            st.error(f"Erro ao processar mensagem: {str(e)}")

# Mostra mensagens no container do chat
with chat_container:
    for message in st.session_state.messages:
        if message["role"] == "system" and show_thoughts:
            # Mostra pensamentos em um expander
            with st.expander("💭 Processo de Pensamento", expanded=True):
                st.markdown(message["content"])
        else:
            # Mostra mensagens normais
            with st.chat_message(message["role"]):
                st.markdown(message["content"]) 