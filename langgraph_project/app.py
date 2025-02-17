import os
import sys

# Adiciona o diretório raiz ao PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph.graph import Graph
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from src.config.langsmith_config import configure_langsmith
from dotenv import load_dotenv

def test_installation():
    """
    Testa a instalação e configuração do LangGraph e LangSmith.
    """
    print("Iniciando testes de instalação...")
    
    # Carrega variáveis de ambiente
    load_dotenv()
    
    # Configura o LangSmith
    if not configure_langsmith():
        print("Erro ao configurar LangSmith")
        return False
        
    print("LangSmith configurado com sucesso!")
    print("Testando criação de grafo...")
    
    try:
        # Cria um grafo simples para teste
        graph = Graph()
        print("Grafo criado com sucesso!")
        
        # Testa a conexão com o Groq
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY não encontrada nas variáveis de ambiente")
            
        llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name="mixtral-8x7b-32768",
            temperature=0
        )
        
        messages = [
            SystemMessage(content="Você é um assistente útil."),
            HumanMessage(content="Olá! Como você está?")
        ]
        
        response = llm.invoke(messages)
        print("Teste de LLM realizado com sucesso!")
        print(f"Resposta do modelo: {response.content}")
        
        return True
        
    except Exception as e:
        print(f"Erro durante os testes: {e}")
        return False

if __name__ == "__main__":
    test_installation()
