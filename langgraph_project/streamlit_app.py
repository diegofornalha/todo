import streamlit as st
from rag_graph import create_rag_graph
import json
import re
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(
    page_title="Assistente RAG em Tempo Real",
    page_icon="🔍",
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
        {"role": "assistant", "content": "Olá! Eu sou seu assistente RAG. Posso ajudar você a:\n\n" + 
         "- Responder perguntas usando nossa base de conhecimento\n" +
         "- Analisar documentos em tempo real\n" +
         "- Fornecer respostas contextualizadas\n\n" +
         "Como posso ajudar você hoje?"}
    ]
    
if 'stats' not in st.session_state:
    st.session_state.stats = {
        "queries": 0,
        "successful_queries": 0,
        "query_history": [],
        "response_times": []
    }

# Inicializa o grafo
try:
    graph = create_rag_graph()
except Exception as e:
    st.error(f"Erro ao inicializar o assistente: {str(e)}")
    st.stop()

# Layout principal
st.title("🔍 Assistente RAG em Tempo Real")

# Sidebar para estatísticas e visualizações
with st.sidebar:
    st.header("📊 Estatísticas")
    
    # Métricas principais
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total de Consultas", st.session_state.stats["queries"])
    with col2:
        success_rate = (st.session_state.stats["successful_queries"] / 
                       st.session_state.stats["queries"] * 100 
                       if st.session_state.stats["queries"] > 0 else 0)
        st.metric("Taxa de Sucesso", f"{success_rate:.1f}%")
    
    # Gráfico de tipos de consulta
    if st.session_state.stats["query_history"]:
        st.subheader("Tipos de Consulta")
        query_types = [q["type"] for q in st.session_state.stats["query_history"]]
        type_counts = {t: query_types.count(t) for t in set(query_types)}
        
        fig = go.Figure(data=[
            go.Pie(
                labels=list(type_counts.keys()),
                values=list(type_counts.values()),
                hole=.3
            )
        ])
        fig.update_layout(margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig, use_container_width=True)
    
    # Histórico de consultas
    st.subheader("Histórico de Consultas")
    for query in reversed(st.session_state.stats["query_history"][-5:]):
        with st.expander(f"{query['timestamp']} - {query['type']}"):
            st.write(f"**Pergunta:** {query['question']}")
            st.write(f"**Tempo:** {query['response_time']:.2f}s")
            st.write(f"**Fontes:** {', '.join(query['sources'])}")

# Container principal para o chat
chat_container = st.container()

# Área de chat
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                thought, answer = extract_thought_and_answer(message["content"])
                if thought:
                    with st.expander("💭 Processo de Pensamento"):
                        st.write(thought)
                st.write(answer)
            else:
                st.write(message["content"])

# Input do usuário
if prompt := st.chat_input("Digite sua pergunta..."):
    # Adiciona a mensagem do usuário
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Processa com o grafo
    try:
        start_time = datetime.now()
        result = graph.invoke({
            "messages": st.session_state.messages,
            "query_type": "",
            "stats": st.session_state.stats,
            "retrieval_qa": None
        })
        
        # Atualiza estatísticas
        response_time = (datetime.now() - start_time).total_seconds()
        st.session_state.stats["queries"] += 1
        st.session_state.stats["successful_queries"] += 1
        st.session_state.stats["response_times"].append(response_time)
        
        # Extrai informações da resposta
        thought, answer = extract_thought_and_answer(result["messages"][-1]["content"])
        query_type = result.get("query_type", "unknown")
        sources = []
        if thought:
            source_match = re.search(r'Fontes consultadas: (.*)', thought)
            if source_match:
                sources = [s.strip() for s in source_match.group(1).split(',')]
        
        # Registra no histórico
        st.session_state.stats["query_history"].append({
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "type": query_type,
            "question": prompt,
            "response_time": response_time,
            "sources": sources
        })
        
        # Atualiza a interface
        st.session_state.messages = result["messages"]
        st.rerun()
        
    except Exception as e:
        st.error(f"Erro ao processar sua pergunta: {str(e)}")
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"Desculpe, ocorreu um erro ao processar sua pergunta: {str(e)}"
        }) 