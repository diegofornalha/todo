# Sistema de RetrievalQA

Um sistema modular e escalável para consultas em linguagem natural sobre documentos, usando embeddings e modelos de linguagem.

## Características

- Processamento e indexação de documentos
- Busca semântica usando FAISS
- Integração com modelos de linguagem (atualmente suporta Groq)
- Sistema de logging robusto
- Armazenamento persistente de índices
- Respostas em português do Brasil

## Estrutura do Projeto

```
.
├── src/
│   ├── core/
│   │   └── retrieval_qa.py      # Classe principal do sistema
│   ├── models/
│   │   ├── llm_handler.py       # Interface base para modelos
│   │   └── groq_handler.py      # Implementação para Groq
│   ├── utils/
│   │   └── logging_config.py    # Configuração de logging
│   ├── config/
│   │   └── prompt_templates.py  # Templates de prompts
│   ├── api/                     # Futura API REST
│   └── examples/
│       └── qa_example.py        # Exemplo de uso
├── data/                        # Armazenamento de índices
├── logs/                        # Arquivos de log
└── requirements.txt            # Dependências do projeto
```

## Instalação

1. Clone o repositório:
```bash
git clone <url-do-repositorio>
cd retrieval-qa
```

2. Crie e ative um ambiente virtual:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```

4. Configure as variáveis de ambiente:
```bash
cp .env.example .env
# Edite .env e adicione sua GROQ_API_KEY
```

## Uso

### Exemplo Básico

```python
from src.core.retrieval_qa import RetrievalQA
from src.models.groq_handler import GroqHandler

# Inicializa o handler do modelo
groq_handler = GroqHandler()
groq_handler.initialize()

# Cria o sistema de QA
qa_system = RetrievalQA(groq_handler)

# Adiciona documentos
documents = [
    {
        "content": "Texto do documento...",
        "source": "documento1.pdf"
    }
]
qa_system.add_documents(documents)

# Faz uma pergunta
result = qa_system.query(
    "Qual é a sua pergunta?",
    include_sources=True
)

print(f"Resposta: {result['resposta']}")
print(f"Fontes: {result['sources']}")
```

### Salvando e Carregando Índices

```python
# Salva o índice
qa_system.save_vectorstore("data/meu_indice")

# Carrega o índice
qa_system.load_vectorstore("data/meu_indice")
```

## Contribuindo

1. Faça um fork do projeto
2. Crie uma branch para sua feature (`git checkout -b feature/nova-feature`)
3. Commit suas mudanças (`git commit -am 'Adiciona nova feature'`)
4. Push para a branch (`git push origin feature/nova-feature`)
5. Crie um Pull Request

## Licença

Este projeto está licenciado sob a licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

## Roadmap

- [ ] Implementar FASTAPI
- [ ] Implementar cache de respostas
- [ ] Adicionar dashboard de monitoramento
- [ ] Suporte a mais formatos de documento
- [ ] Melhorar a performance do processamento de documentos 

# Chat RAG com Redis

Este projeto implementa um sistema de Retrieval-Augmented Generation (RAG) com cache Redis para melhor performance.

## Requisitos

- Python 3.8+
- Redis 6+
- Dependências Python (ver `requirements.txt`)

## Configuração

1. Instale as dependências:
```bash
pip install -r requirements.txt
```

2. Configure o Redis:

- Copie o arquivo de configuração do Redis:
```bash
cp redis.conf.example redis.conf
```

- Edite o arquivo `redis.conf` conforme necessário:
  - `port`: Porta do Redis (padrão: 6379)
  - `requirepass`: Senha do Redis
  - `maxmemory`: Limite de memória
  - `maxmemory-policy`: Política de expiração

3. Inicie o Redis com a configuração:
```bash
redis-server redis.conf
```

## Configuração do RAG

O sistema RAG pode ser configurado através da classe `RAGConfig`:

```python
from langgraph_agente_vendedor.core.base_rag import RAGConfig

config = RAGConfig(
    # Configurações gerais
    embeddings_model="sentence-transformers/all-MiniLM-L6-v2",
    chunk_size=1000,
    chunk_overlap=200,
    max_documents=3,
    similarity_threshold=0.7,
    
    # Configurações de cache
    cache_enabled=True,
    cache_type="redis",  # "redis" ou "file"
    
    # Configurações do Redis
    redis_host="localhost",
    redis_port=6379,
    redis_password="sua_senha",
    redis_db=0,
    redis_prefix="rag:",
    redis_ttl=3600  # 1 hora
)
```

## Uso

```python
from langgraph_agente_vendedor.core.faiss_rag import FAISSRAGSystem

# Inicializa o sistema
rag = FAISSRAGSystem()
rag.initialize(config)

# Adiciona documentos
documents = [
    {
        "content": "Texto do documento",
        "metadata": {"source": "arquivo.txt"}
    }
]
rag.add_documents(documents)

# Faz uma consulta
response = rag.query("Sua pergunta aqui?")
print(response.answer)
```

## Cache Redis

O sistema usa Redis para cache de respostas, oferecendo:

- Cache distribuído
- Expiração automática
- Alta performance
- Persistência opcional
- Failback para cache em arquivo

Se o Redis não estiver disponível, o sistema automaticamente usa cache em arquivo.

## Testes

Execute os testes com:
```bash
python -m pytest tests/ -v
```

## Contribuindo

1. Fork o projeto
2. Crie sua feature branch (`git checkout -b feature/nome`)
3. Commit suas mudanças (`git commit -am 'Adiciona feature'`)
4. Push para a branch (`git push origin feature/nome`)
5. Crie um Pull Request

## Licença

Este projeto está licenciado sob a licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.
