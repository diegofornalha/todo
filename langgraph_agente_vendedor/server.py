import os
from fastapi import FastAPI, HTTPException, APIRouter, Request
from rag_graph import create_rag_graph
from cors_config import configure_cors
import uvicorn
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv

# Carrega variáveis de ambiente
load_dotenv()

app = FastAPI(
    title="LangGraph RAG API",
    description="API para processamento de consultas RAG em tempo real",
    version="1.0.0"
)

# Configura CORS
app = configure_cors(app)

# Cria um router para o prefixo /graphs
graphs_router = APIRouter(prefix="/graphs")

# Inicializa o grafo
try:
    graph = create_rag_graph()
except Exception as e:
    print(f"Erro ao criar grafo: {e}")
    raise

class GraphInput(BaseModel):
    messages: List[Dict]
    query_type: str
    stats: Dict[str, Any]
    retrieval_qa: Optional[Dict] = None

@app.get("/")
async def root():
    return {"message": "LangGraph RAG API está funcionando!"}

@app.post("/query")
async def process_query(input_data: GraphInput):
    """
    Processa uma consulta RAG.
    
    Args:
        input_data: Dados de entrada para o grafo
        
    Returns:
        Resultado do processamento
    """
    try:
        result = graph.invoke(dict(input_data))
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao processar consulta: {str(e)}"
        )

@app.get("/stats")
async def get_stats():
    """Retorna estatísticas do sistema RAG."""
    try:
        return {
            "total_queries": graph.get_stats().get("queries", 0),
            "successful_queries": graph.get_stats().get("successful_queries", 0),
            "success_rate": graph.get_stats().get("success_rate", 0.0),
            "avg_response_time": graph.get_stats().get("avg_response_time", 0.0)
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao obter estatísticas: {str(e)}"
        )

# Rota genérica para lidar com qualquer método HTTP
@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD", "PATCH"])
async def catch_all(path: str, request: Request):
    try:
        # Log da requisição para debug
        print(f"Método: {request.method}, Path: {path}")
        print(f"Headers: {request.headers}")
        body = await request.body()
        print(f"Body: {body}")
        
        # Se for OPTIONS, retorna 200 OK
        if request.method == "OPTIONS":
            return {"ok": True}
            
        # Para outros métodos, tenta processar normalmente
        if path.startswith("graphs/"):
            # Remove o prefixo 'graphs/' e processa
            actual_path = path[7:]
            return {"path": actual_path, "method": request.method}
            
        return {"path": path, "method": request.method}
    except Exception as e:
        print(f"Erro no catch_all: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Inclui o router de graphs na aplicação
app.include_router(graphs_router)

def start():
    """Função para iniciar o servidor"""
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8123,
        reload=True,
        reload_dirs=[".", ".."],
    )

if __name__ == "__main__":
    start() 