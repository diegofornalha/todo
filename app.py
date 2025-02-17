import streamlit as st
from src.core.document_processor import DocumentProcessor
from src.core.qa_chain import QAChain
from src.models.groq_handler import GroqHandler
import os
from tempfile import NamedTemporaryFile
import logging
from src.utils.logging_config import setup_logger
import re

# Configuração do logger
logger = setup_logger('streamlit_app', 'logs/app.log')

# Configuração da página Streamlit
st.set_page_config(
    page_title="Sistema de Perguntas e Respostas",
    page_icon="🤖",
    layout="wide"
)

# Título e descrição
st.title("Sistema de Perguntas e Respostas 🤖")
st.markdown("""
Este sistema permite que você:
1. Faça upload de documentos (PDFs)
2. Processe e indexe o conteúdo
3. Faça perguntas sobre os documentos
""")

# Inicialização do Groq e QA Chain
@st.cache_resource
def initialize_qa_system():
    try:
        handler = GroqHandler()
        handler.initialize()
        qa_system = QAChain(handler)
        return qa_system
    except Exception as e:
        st.error(f"Erro ao inicializar o sistema: {str(e)}")
        return None

# Função para processar documento
def process_document(uploaded_file):
    try:
        # Cria um arquivo temporário para o PDF
        with NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        # Processa o documento
        processor = DocumentProcessor(tmp_path)
        docs = processor.process_pdf()
        
        # Remove o arquivo temporário
        os.unlink(tmp_path)
        
        return docs
    except Exception as e:
        st.error(f"Erro ao processar o documento: {str(e)}")
        return None

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
        # Se não encontrar tags de pensamento, retorna None para pensamento e o texto original como resposta
        return None, text.strip()

# Inicializa o sistema de QA
qa_system = initialize_qa_system()

# Sidebar para upload de documentos
with st.sidebar:
    st.header("Upload de Documentos")
    uploaded_files = st.file_uploader(
        "Escolha os arquivos PDF",
        type=['pdf'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.write(f"Arquivos carregados: {len(uploaded_files)}")
        
        if st.button("Processar Documentos"):
            with st.spinner("Processando documentos..."):
                for file in uploaded_files:
                    st.write(f"Processando: {file.name}")
                    docs = process_document(file)
                    if docs:
                        # Adiciona os documentos ao QA system
                        qa_system.add_documents(docs)
                        st.success(f"{file.name} processado com sucesso!")
                    else:
                        st.error(f"Erro ao processar {file.name}")

# Área principal para perguntas
st.header("Faça sua Pergunta")

# Checkbox para mostrar/ocultar pensamento
show_thought = st.checkbox("Mostrar processo de pensamento", value=False)

# Input da pergunta
question = st.text_input("Digite sua pergunta:")

if question:
    if not qa_system:
        st.error("Sistema não inicializado corretamente.")
    else:
        with st.spinner("Processando sua pergunta..."):
            try:
                # Processa a pergunta
                result = qa_system.query(question)
                
                # Extrai pensamento e resposta
                thought, answer = extract_thought_and_answer(result['resposta'])
                
                # Exibe a resposta
                st.subheader("Resposta:")
                
                # Mostra o pensamento se a opção estiver marcada e existir pensamento
                if show_thought and thought:
                    with st.expander("Processo de Pensamento", expanded=True):
                        st.markdown(thought)
                
                # Mostra a resposta final
                st.markdown(answer)
                
                # Exibe o status
                if result['status'] == 'sucesso':
                    st.success("Pergunta processada com sucesso!")
                else:
                    st.error("Erro ao processar a pergunta")
                
            except Exception as e:
                st.error(f"Erro ao processar a pergunta: {str(e)}")

# Footer
st.markdown("---")
st.markdown("Desenvolvido com ❤️ usando Streamlit e LangChain") 