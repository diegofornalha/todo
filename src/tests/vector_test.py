from ..core.vector_store import VectorStore

try:
    print("Iniciando teste do VectorStore...")
    
    # Inicializa o store
    store = VectorStore()
    
    # Adiciona alguns documentos
    print("\nPreparando documentos de teste...")
    documents = [
        {"content": "O gato dorme no sofá", "id": 1},
        {"content": "O cachorro brinca no jardim", "id": 2},
        {"content": "A criança lê um livro", "id": 3},
    ]
    texts = [doc["content"] for doc in documents]
    
    # Adiciona os documentos
    store.add_documents(documents, texts)
    
    # Faz uma busca
    query = "animal dormindo"
    print(f"\nRealizando busca por: '{query}'")
    results = store.similarity_search(query)
    
    # Mostra os resultados
    print("\nResultados:")
    for doc in results:
        print(f"ID: {doc['id']}, Conteúdo: {doc['content']}")
except Exception as e:
    print(f"Erro durante o teste: {e}")
    import traceback
    traceback.print_exc() 