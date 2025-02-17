import streamlit as st
from todo_graph import create_todo_graph
import json

st.set_page_config(page_title="LangGraph TODO App", page_icon="✅")

st.title("LangGraph TODO App")

# Inicializa o estado da sessão
if 'task_list' not in st.session_state:
    st.session_state.task_list = []
    
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Inicializa o grafo
try:
    graph = create_todo_graph()
    st.success("Grafo TODO inicializado com sucesso!")
except Exception as e:
    st.error(f"Erro ao inicializar o grafo: {str(e)}")
    st.stop()

# Interface para adicionar tarefas
with st.form("new_task"):
    task = st.text_input("Nova Tarefa")
    submitted = st.form_submit_button("Adicionar")
    
    if submitted and task:
        # Prepara o input para o grafo
        input_data = {
            "messages": [{"role": "user", "content": f"Adicionar tarefa: {task}"}],
            "next_step": "",
            "task_list": st.session_state.task_list,
            "current_task": None
        }
        
        # Invoca o grafo
        try:
            result = graph.invoke(input_data)
            st.session_state.task_list = result["task_list"]
            st.success("Tarefa adicionada com sucesso!")
        except Exception as e:
            st.error(f"Erro ao adicionar tarefa: {str(e)}")

# Lista de tarefas
if st.session_state.task_list:
    st.subheader("Tarefas Pendentes")
    for i, task in enumerate(st.session_state.task_list):
        col1, col2 = st.columns([4,1])
        with col1:
            st.write(f"{i+1}. {task}")
        with col2:
            if st.button("✅", key=f"complete_{i}"):
                # Prepara o input para completar a tarefa
                input_data = {
                    "messages": [{"role": "user", "content": str(i+1)}],
                    "next_step": "",
                    "task_list": st.session_state.task_list,
                    "current_task": None
                }
                
                # Invoca o grafo
                try:
                    result = graph.invoke(input_data)
                    st.session_state.task_list = result["task_list"]
                    st.success("Tarefa concluída!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Erro ao concluir tarefa: {str(e)}")
else:
    st.info("Nenhuma tarefa pendente!")

# Área de debug
if st.checkbox("Mostrar Debug"):
    st.subheader("Debug")
    st.json({
        "task_list": st.session_state.task_list,
        "messages": st.session_state.messages
    }) 