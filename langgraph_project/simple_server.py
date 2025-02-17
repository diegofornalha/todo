from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI()

# Configuração CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "API está funcionando!"}

@app.get("/ok")
async def health_check():
    return {"ok": True}

@app.get("/graphs/info")
async def get_info():
    return {
        "version": "1.0.0",
        "name": "LangGraph TODO API",
        "description": "API para gerenciamento de tarefas usando LangGraph"
    }

@app.post("/graphs/assistants/search")
async def search_assistants():
    return {
        "assistants": [
            {
                "id": "todo",
                "name": "TODO Assistant",
                "description": "Assistente para gerenciamento de tarefas"
            }
        ]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8123) 