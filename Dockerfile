FROM python:3.11-slim

WORKDIR /app

# Instala as dependências do sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copia os arquivos do projeto
COPY requirements.txt .
COPY langgraph.json .
COPY langgraph_project/ langgraph_project/
COPY src/ src/

# Instala as dependências Python
RUN pip install --no-cache-dir -r requirements.txt

# Cria um script para iniciar o servidor
RUN echo '#!/usr/bin/env python3\n\
from langgraph_project.todo_graph import create_todo_graph\n\
from fastapi import FastAPI\n\
from pydantic import BaseModel\n\
import uvicorn\n\
\n\
app = FastAPI()\n\
graph = create_todo_graph()\n\
\n\
class GraphInput(BaseModel):\n\
    messages: list\n\
    next_step: str\n\
    task_list: list\n\
    current_task: str | None\n\
\n\
@app.post("/invoke")\n\
async def invoke_graph(input_data: GraphInput):\n\
    result = graph.invoke(dict(input_data))\n\
    return result\n\
\n\
if __name__ == "__main__":\n\
    uvicorn.run(app, host="0.0.0.0", port=2024)\n\
' > server.py

# Expõe a porta do servidor
EXPOSE 2024

# Comando para iniciar o servidor
CMD ["python", "server.py"] 