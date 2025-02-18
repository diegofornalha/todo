import subprocess
import sys
import os
from dotenv import load_dotenv

def start_services():
    """
    Inicia o LangGraph e o Streamlit em portas diferentes.
    LangGraph: 8123
    Streamlit: 8501
    """
    # Carrega variáveis de ambiente
    load_dotenv()
    
    try:
        # Inicia o servidor LangGraph em background
        print("Iniciando servidor LangGraph na porta 8123...")
        langgraph_process = subprocess.Popen(
            [sys.executable, "langgraph_project/server.py"],
            env=os.environ.copy()
        )
        
        # Inicia o Streamlit em uma porta diferente
        print("Iniciando Streamlit na porta 8501...")
        streamlit_process = subprocess.Popen(
            [sys.executable, "app.py", "8501"],
            env=os.environ.copy()
        )
        
        print("\nServiços iniciados:")
        print("- LangGraph: http://localhost:8123")
        print("- Streamlit: http://localhost:8501")
        print("\nPressione Ctrl+C para encerrar os serviços")
        
        # Aguarda os processos
        langgraph_process.wait()
        streamlit_process.wait()
        
    except KeyboardInterrupt:
        print("\nEncerrando serviços...")
        langgraph_process.terminate()
        streamlit_process.terminate()
        sys.exit(0)
    except Exception as e:
        print(f"Erro ao iniciar serviços: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    start_services() 