import streamlit as st
import os
from src.core.document_processor import DocumentProcessor
from src.core.qa_chain import QAChain
from src.models.llm_handler import LLMHandler
from src.utils.logging_config import setup_logger
from dotenv import load_dotenv

# Carrega variáveis de ambiente
load_dotenv()

# Configura o logger
logger = setup_logger('streamlit_app', 'logs/streamlit_app.log')

# Configuração da página
st.set_page_config(
    page_title="Assistente de Documentos",
    page_icon="📚",
    layout="wide"
)

# Estilo CSS personalizado
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .upload-box {
        border: 2px dashed #4CAF50;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_qa_system():
    """Inicializa o sistema de QA com cache."""
    try:
        api_key = os.getenv('GROQ_API_KEY')
        if not api_key:
            raise ValueError("GROQ_API_KEY não encontrada nas variáveis de ambiente")
            
        llm_handler = LLMHandler(api_key=api_key)
        qa_chain = QAChain(llm_handler=llm_handler)
        return qa_chain
        
    except Exception as e:
        logger.error(f"Erro ao inicializar o sistema de QA: {str(e)}")
        st.error(f"Erro ao inicializar o sistema: {str(e)}")
        return None

def main():
    # Título
    st.title("📚 Assistente de Documentos")
    st.markdown("### Carregue seus documentos e faça perguntas sobre eles")
    
    # Inicializa o sistema
    qa_chain = initialize_qa_system()
    if not qa_chain:
        st.stop()
    
    # Sidebar para upload de documentos
    with st.sidebar:
        st.header("📄 Upload de Documentos")
        
        uploaded_files = st.file_uploader(
            "Arraste ou selecione seus arquivos PDF",
            type=['pdf'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            with st.spinner("Processando documentos..."):
                try:
                    # Cria o diretório temp se não existir
                    temp_dir = "temp"
                    os.makedirs(temp_dir, exist_ok=True)
                    
                    processed_files = []
                    temp_paths = []  # Lista para armazenar caminhos temporários
                    
                    # Primeiro, salva todos os arquivos
                    for file in uploaded_files:
                        try:
                            temp_path = os.path.join(temp_dir, file.name)
                            temp_paths.append(temp_path)
                            
                            with open(temp_path, "wb") as f:
                                f.write(file.getvalue())
                                
                        except Exception as e:
                            logger.error(f"Erro ao salvar {file.name}: {str(e)}")
                            st.error(f"Erro ao salvar {file.name}: {str(e)}")
                            continue
                    
                    # Depois, processa todos os arquivos
                    for temp_path in temp_paths:
                        try:
                            processor = DocumentProcessor(file_path=temp_path)
                            docs = processor.process_pdf()
                            qa_chain.add_documents(docs)
                            
                            processed_files.append(os.path.basename(temp_path))
                            
                        except Exception as e:
                            logger.error(f"Erro ao processar {temp_path}: {str(e)}")
                            st.error(f"Erro ao processar {os.path.basename(temp_path)}: {str(e)}")
                    
                    # Por fim, remove os arquivos temporários
                    for temp_path in temp_paths:
                        try:
                            if os.path.exists(temp_path):
                                os.remove(temp_path)
                        except Exception as e:
                            logger.error(f"Erro ao remover arquivo temporário {temp_path}: {str(e)}")
                    
                    if processed_files:
                        st.success(f"Arquivo(s) processado(s) com sucesso: {', '.join(processed_files)}")
                    
                except Exception as e:
                    logger.error(f"Erro ao processar documentos: {str(e)}")
                    st.error(f"Erro ao processar documentos: {str(e)}")
    
    # Área principal para perguntas
    st.header("❓ Faça sua pergunta")
    
    # Input da pergunta
    question = st.text_input("Digite sua pergunta sobre os documentos")
    
    if st.button("Enviar Pergunta", key="send_question"):
        if not question:
            st.warning("Por favor, digite uma pergunta.")
            return
            
        if not uploaded_files:
            st.warning("Por favor, faça upload de pelo menos um documento primeiro.")
            return
            
        with st.spinner("Processando sua pergunta..."):
            try:
                # Processa a pergunta
                response = qa_chain.query(question)
                
                # Exibe a resposta
                st.markdown("### Resposta:")
                st.write(response['resposta'])
                
                # Exibe as fontes
                if 'sources' in response:
                    st.markdown("### Fontes:")
                    for source in response['sources']:
                        st.markdown(f"- {source}")
                        
            except Exception as e:
                logger.error(f"Erro ao processar pergunta: {str(e)}")
                st.error(f"Erro ao processar sua pergunta: {str(e)}")

if __name__ == "__main__":
    main() 