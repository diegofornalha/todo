from langchain_community.document_loaders import PyPDFLoader
import os

# Caminho para o PDF
pdf_path = "content/Origem.pdf"

try:
    # Verifica se o arquivo existe
    print(f"Verificando se o arquivo existe em: {os.path.abspath(pdf_path)}")
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {pdf_path}")
    
    print("Arquivo encontrado, tentando carregar...")
    
    # Carrega o PDF
    loader = PyPDFLoader(pdf_path)
    print("Loader criado, tentando carregar documentos...")
    
    documents = loader.load()
    print(f"\nCarregadas {len(documents)} páginas do PDF")
    
    # Mostra o conteúdo da primeira página
    if documents:
        print("\nConteúdo da primeira página:")
        print(documents[0].page_content[:500])
except Exception as e:
    print(f"Erro ao carregar o PDF: {e}")
    import traceback
    traceback.print_exc() 