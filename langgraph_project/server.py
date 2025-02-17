import os
from fastapi import FastAPI, HTTPException, APIRouter, Request
from todo_graph import create_todo_graph
from cors_config import configure_cors
import uvicorn
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv

# Carrega variáveis de ambiente
load_dotenv()

app = FastAPI(
    title="LangGraph TODO API",
    description="API para gerenciamento de tarefas usando LangGraph",
    version="1.0.0"
)

# Configura CORS
app = configure_cors(app)

# Cria um router para o prefixo /graphs
graphs_router = APIRouter(prefix="/graphs")

# Inicializa o grafo
try:
    graph = create_todo_graph()
except Exception as e:
    print(f"Erro ao criar grafo: {e}")
    raise

class GraphInput(BaseModel):
    messages: List[Dict]
    next_step: str
    task_list: List[str]
    current_task: Optional[str] = None

class Assistant(BaseModel):
    id: str
    name: str
    description: str

@app.get("/")
async def root():
    return {"message": "LangGraph TODO API está funcionando!"}

@app.get("/ok")
async def health_check():
    return {"ok": True}

@app.get("/info")
@graphs_router.get("/info")
async def get_info():
    return {
        "version": "1.0.0",
        "name": "LangGraph TODO API",
        "description": "API para gerenciamento de tarefas usando LangGraph"
    }

@app.get("/graphs")
async def list_graphs():
    return {"graphs": ["todo"]}

@app.post("/assistants/search")
@graphs_router.post("/assistants/search")
async def search_assistants():
    assistants = [
        Assistant(
            id="todo",
            name="TODO Assistant",
            description="Assistente para gerenciamento de tarefas"
        )
    ]
    return {"assistants": assistants}

@app.post("/invoke")
@graphs_router.post("/invoke")
async def invoke_graph(input_data: GraphInput):
    try:
        result = graph.invoke(dict(input_data))
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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