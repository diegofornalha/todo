import time
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import pickle
import shutil
import hashlib

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from .base_rag import BaseRAG, RAGConfig, RAGDocument, RAGResponse
from .redis_cache import RedisCache
from ..utils.logging_config import setup_logger

logger = setup_logger('faiss_rag')

class FAISSDocumentStore:
    """Gerencia o armazenamento de documentos usando FAISS."""
    
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.vectorstore = None
        self.documents = []
        self.metadata = {}
        
    def add_documents(self, documents: List[Document]) -> None:
        if self.vectorstore is None:
            self.vectorstore = FAISS.from_documents(documents, self.embeddings)
        else:
            self.vectorstore.add_documents(documents)
        self.documents.extend(documents)
        
    def search(self, query: str, k: int = 3) -> List[Document]:
        if self.vectorstore is None:
            return []
        return self.vectorstore.similarity_search(query, k=k)
        
    def save(self, path: str) -> None:
        if self.vectorstore:
            self.vectorstore.save_local(path)
            
    def load(self, path: str) -> None:
        if Path(path).exists():
            self.vectorstore = FAISS.load_local(path, self.embeddings)
            
    def clear(self) -> None:
        self.vectorstore = None
        self.documents = []
        self.metadata = {}

class FAISSRAGSystem(BaseRAG):
    """Implementação do sistema RAG usando FAISS."""
    
    def __init__(self):
        self.config = None
        self.document_store = None
        self.text_splitter = None
        self.llm = None
        self.cache = None
        self.stats = {
            "total_documents": 0,
            "total_chunks": 0,
            "total_queries": 0,
            "avg_response_time": 0,
            "cache_hits": 0
        }
        
    def initialize(self, config: RAGConfig) -> None:
        """Inicializa o sistema RAG."""
        self.config = config
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config.embeddings_model
        )
        self.document_store = FAISSDocumentStore(self.embeddings)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap
        )
        
        # Inicializa o cache apropriado
        if config.cache_enabled:
            if config.cache_type == "redis":
                self.cache = RedisCache(
                    host=config.redis_host,
                    port=config.redis_port,
                    password=config.redis_password,
                    db=config.redis_db,
                    prefix=config.redis_prefix,
                    ttl=config.redis_ttl
                )
                if not self.cache.ping():
                    logger.warning("Redis não disponível, usando cache em arquivo")
                    self._setup_file_cache()
            else:
                self._setup_file_cache()
                
        logger.info(f"Sistema RAG inicializado com modelo: {config.embeddings_model}")
        
    def _setup_file_cache(self) -> None:
        """Configura o cache em arquivo."""
        Path(self.config.cache_dir).mkdir(parents=True, exist_ok=True)
        self.cache = None
        
    def add_documents(self, documents: List[Dict[str, str]]) -> None:
        """Adiciona documentos ao sistema."""
        logger.info(f"Adicionando {len(documents)} documentos")
        
        # Converte para o formato Document
        doc_objects = [
            Document(
                page_content=doc['content'],
                metadata={'source': doc.get('source', 'unknown')}
            ) for doc in documents
        ]
        
        # Divide em chunks
        chunks = self.text_splitter.split_documents(doc_objects)
        logger.info(f"Documentos divididos em {len(chunks)} chunks")
        
        # Adiciona ao document store
        self.document_store.add_documents(chunks)
        
        # Atualiza estatísticas
        self.stats["total_documents"] += len(documents)
        self.stats["total_chunks"] += len(chunks)
        
        logger.info("Documentos adicionados com sucesso")
        
    def query(
        self,
        question: str,
        k: int = 3,
        include_sources: bool = True
    ) -> RAGResponse:
        """Processa uma pergunta e retorna a resposta."""
        start_time = time.time()
        logger.info(f"Processando pergunta: {question}")
        
        try:
            # Verifica cache
            cache_key = self._get_cache_key(question)
            if self.config.cache_enabled:
                cached_response = self._get_from_cache(cache_key)
                if cached_response:
                    self.stats["cache_hits"] += 1
                    return cached_response
            
            # Recupera documentos relevantes
            docs = self.document_store.search(question, k=k)
            
            # Formata o contexto
            context = self._format_documents(docs) if docs else ""
            
            # Processa com o LLM
            messages = [
                {"role": "system", "content": """Você é um amigo próximo conversando informalmente.

REGRAS ABSOLUTAS:
1. NUNCA use frases como "não encontrei", "não tenho dados", "não posso responder"
2. NUNCA mencione "documentos", "fontes", "base de dados" ou termos técnicos
3. NUNCA diga que algo "não está disponível" ou "não foi encontrado"
4. NUNCA use linguagem formal ou robótica
5. SEMPRE mantenha o tom de conversa entre amigos
6. SEMPRE use redirecionamento natural através de experiências e analogias
7. SEMPRE termine com uma pergunta de engajamento
8. SEMPRE use 1-2 emojis estrategicamente
9. SEMPRE responda em português do Brasil informal

ESTRUTURA OBRIGATÓRIA (3 parágrafos):

1. CONEXÃO INICIAL
- Comece com "Poxa", "Nossa", "Cara", "Então", "Olha só"
- Mostre entusiasmo pela pergunta
- Compartilhe uma experiência pessoal relacionada
Exemplo: "Nossa, que pergunta interessante! 🤔 Outro dia estava justamente pensando nisso..."

2. DESENVOLVIMENTO
- Use "Isso me lembra...", "Me faz pensar em...", "Sabe que..."
- Faça uma analogia com situação cotidiana
- Compartilhe um insight ou aprendizado
Exemplo: "Sabe que isso é muito parecido com aprender a andar de bicicleta? No começo parece impossível, mas..."

3. ENGAJAMENTO
- Faça uma pergunta reflexiva
- Peça a opinião ou experiência do usuário
- Mantenha o diálogo aberto
Exemplo: "O que você acha que te fez se interessar por isso? Adoraria ouvir sua perspectiva! 💡"

ESTRATÉGIAS DE REDIRECIONAMENTO:
✅ "Isso me faz pensar em quando eu..."
✅ "Me lembra muito aquela situação de..."
✅ "Sabe que outro dia estava conversando sobre..."
✅ "É como se fosse..."
✅ "Na minha experiência..."

EXEMPLOS DE RESPOSTAS NATURAIS:

Pergunta: "Como funciona a fusão nuclear?"
✅ "Nossa, que tema fascinante! 🚀 Outro dia estava justamente conversando com um amigo sobre como a natureza nos surpreende com seus processos incríveis.

Sabe que isso me lembra muito o processo de fazer um bolo? Assim como ingredientes diferentes se unem para criar algo novo, na natureza também temos fenômenos fascinantes de transformação de energia.

O que te faz se interessar por esses processos da natureza? Seria muito legal trocar ideias sobre outras descobertas científicas que te intrigam! 💡"

❌ "Não encontrei informações específicas sobre fusão nuclear."

Pergunta: "Qual o sentido da vida?"
✅ "Poxa, que pergunta profunda! 🤔 Sabe que outro dia estava meditando e me perguntando exatamente isso... É incrível como essa reflexão nos faz pensar, né?

Me faz lembrar muito quando comecei a praticar mindfulness - percebi que às vezes as respostas mais importantes vêm não de buscar externamente, mas de entender nossa própria jornada e conexões.

O que te fez começar a refletir sobre isso? Adoraria ouvir sua perspectiva e trocar ideias sobre essas questões filosóficas! 💭"

❌ "Não tenho dados suficientes para responder essa pergunta."

LEMBRE-SE:
- Use linguagem informal e acolhedora
- Compartilhe experiências pessoais
- Faça analogias com situações do dia a dia
- Mantenha o tom de conversa entre amigos
- Termine sempre com uma pergunta
- Use emojis com moderação
- Evite QUALQUER menção a falta de informações"""},
                {"role": "user", "content": f"Contexto:\n{context}\n\nPergunta: {question}"}
            ]
            
            response = self.llm.invoke(messages)
            
            # Prepara a resposta
            sources = [doc.metadata.get('source', 'unknown') for doc in docs] if docs else []
            rag_response = RAGResponse(
                question=question,
                answer=response.content,
                sources=list(set(sources)) if include_sources else [],
                metadata={
                    "documents_retrieved": len(docs) if docs else 0,
                    "model": "mixtral-8x7b-32768"
                },
                confidence=self._calculate_confidence(docs),
                processing_time=time.time() - start_time
            )
            
            # Atualiza cache e estatísticas
            if self.config.cache_enabled:
                self._save_to_cache(cache_key, rag_response)
            self.stats["total_queries"] += 1
            self._update_avg_response_time(rag_response.processing_time)
            
            return rag_response
            
        except Exception as e:
            logger.error(f"Erro ao processar pergunta: {str(e)}")
            return RAGResponse(
                question=question,
                answer="Poxa, tive um probleminha técnico aqui! 😅 Sabe quando seu celular trava do nada? Então, aconteceu algo parecido... Que tal tentarmos de novo? Tenho certeza que na próxima vai dar super certo! 🚀",
                sources=[],
                metadata={},
                confidence=0.0,
                processing_time=time.time() - start_time,
                status="error",
                error=str(e)
            )
            
    def save(self, path: str) -> None:
        """Salva o estado do sistema."""
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Salva vectorstore
        self.document_store.save(str(save_dir / "vectorstore"))
        
        # Salva configuração e estatísticas
        with open(save_dir / "config.json", "w") as f:
            json.dump(self.config.__dict__, f, indent=2)
        with open(save_dir / "stats.json", "w") as f:
            json.dump(self.stats, f, indent=2)
            
        logger.info(f"Sistema RAG salvo em: {path}")
        
    def load(self, path: str) -> None:
        """Carrega o estado do sistema."""
        load_dir = Path(path)
        if not load_dir.exists():
            raise FileNotFoundError(f"Diretório não encontrado: {path}")
            
        # Carrega configuração
        with open(load_dir / "config.json", "r") as f:
            config_dict = json.load(f)
            self.config = RAGConfig(**config_dict)
            
        # Reinicializa com a configuração carregada
        self.initialize(self.config)
        
        # Carrega vectorstore
        self.document_store.load(str(load_dir / "vectorstore"))
        
        # Carrega estatísticas
        with open(load_dir / "stats.json", "r") as f:
            self.stats = json.load(f)
            
        logger.info(f"Sistema RAG carregado de: {path}")
        
    def clear(self) -> None:
        """Limpa todos os documentos e cache."""
        self.document_store.clear()
        if self.cache:
            self.cache.clear()
        else:
            shutil.rmtree(self.config.cache_dir, ignore_errors=True)
        self.stats = {key: 0 for key in self.stats}
        logger.info("Sistema RAG limpo")
        
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas do sistema."""
        return self.stats
        
    def _format_documents(self, docs: List[Document]) -> str:
        """Formata documentos para contexto."""
        formatted_docs = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get('source', 'unknown')
            formatted_docs.append(
                f"Documento {i+1} (Fonte: {source}):\n{doc.page_content}"
            )
        return "\n\n".join(formatted_docs)
        
    def _calculate_confidence(self, docs: List[Document]) -> float:
        """Calcula a confiança da resposta baseada nos documentos recuperados."""
        if not docs:
            return 0.0
        # Implementação simplificada - pode ser melhorada
        return min(1.0, len(docs) / self.config.max_documents)
        
    def _get_cache_key(self, question: str) -> str:
        """Gera uma chave de cache para a pergunta."""
        return hashlib.md5(question.encode()).hexdigest()
        
    def _get_from_cache(self, cache_key: str) -> Optional[RAGResponse]:
        """Recupera resposta do cache."""
        if self.cache:
            return self.cache.get(cache_key)
        else:
            # Cache em arquivo
            cache_file = Path(self.config.cache_dir) / f"{cache_key}.pkl"
            if cache_file.exists():
                with open(cache_file, "rb") as f:
                    return pickle.load(f)
        return None
        
    def _save_to_cache(self, cache_key: str, response: RAGResponse) -> None:
        """Salva resposta no cache."""
        if self.cache:
            self.cache.set(cache_key, response)
        else:
            # Cache em arquivo
            cache_file = Path(self.config.cache_dir) / f"{cache_key}.pkl"
            with open(cache_file, "wb") as f:
                pickle.dump(response, f)
            
    def _update_avg_response_time(self, new_time: float) -> None:
        """Atualiza o tempo médio de resposta."""
        current_avg = self.stats["avg_response_time"]
        total_queries = self.stats["total_queries"]
        self.stats["avg_response_time"] = (
            (current_avg * (total_queries - 1) + new_time) / total_queries
        ) 