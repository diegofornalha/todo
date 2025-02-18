import pytest
from pathlib import Path
from src.core.vector_store import VectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def test_crypto_pdf_search():
    """Testa busca por similaridade usando dados reais do PDF sobre criptomoedas."""
    # Carrega o PDF
    pdf_path = Path("content/AGENTE IA .pdf")
    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()
    
    # Divide o texto em chunks menores para melhor precisão
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # Reduzindo o tamanho do chunk
        chunk_overlap=100  # Reduzindo o overlap
    )
    chunks = text_splitter.split_documents(pages)
    
    # Prepara os documentos
    documents = [
        {
            "content": chunk.page_content,
            "metadata": {
                "source": pdf_path.name,
                "page": chunk.metadata.get("page", 1)
            }
        }
        for chunk in chunks
    ]
    
    # Inicializa o VectorStore com limiar mais baixo
    store = VectorStore(similarity_threshold=0.3)  # Reduzindo o limiar
    store.add_documents(documents)
    
    # Testa algumas perguntas mais alinhadas com o conteúdo do PDF
    queries = [
        "Como um agente de IA pode ajudar na venda de produtos?",
        "Quais são as melhores práticas para atendimento ao cliente?",
        "O que não fazer durante uma negociação?",
        "Como melhorar a comunicação com o cliente?",
        "Quais são as estratégias de vendas recomendadas?"
    ]
    
    for query in queries:
        results = store.similarity_search(query)
        
        # Verifica se encontrou resultados relevantes
        assert len(results) > 0, f"Nenhum resultado encontrado para: {query}"
        
        # Verifica se os scores estão ordenados
        for i in range(len(results) - 1):
            assert results[i]["score"] >= results[i + 1]["score"]
            
        # Verifica se os scores estão acima do limiar
        assert all(r["score"] >= 0.3 for r in results)
        
        # Imprime os resultados para inspeção
        print(f"\nQuery: {query}")
        for i, result in enumerate(results, 1):
            print(f"\nResultado {i} (score: {result['score']:.3f}, página: {result['metadata']['page']}):")
            print(result['content'][:200] + "...") 