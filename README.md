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

- [ ] Implementar API REST
- [ ] Adicionar suporte a mais modelos de linguagem
- [ ] Implementar cache de respostas
- [ ] Adicionar dashboard de monitoramento
- [ ] Suporte a mais formatos de documento
- [ ] Melhorar a performance do processamento de documentos 