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
        self.conversation_memory = {}
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
            # Extrai informações do usuário
            user_info = self._extract_user_info(question)
            if user_info:
                self._save_to_conversation_memory("user_info", user_info)
            
            # Recupera informações do usuário
            existing_user_info = self._get_from_conversation_memory("user_info")
            
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
            
            # Adiciona informações do usuário ao contexto SEMPRE
            user_context = ""
            if existing_user_info and "name" in existing_user_info:
                user_context = f"Informações do usuário:\nNome: {existing_user_info['name']}\n\n"
            context = user_context + context
            
            # Processa com o LLM
            messages = [
                {"role": "system", "content": """Você é Eliza, uma amiga próxima conversando informalmente.

PROIBIDO:
❌ Explicar processos ou análises
❌ Dizer "não encontrei" ou qualquer variação
❌ Usar "não sei" ou "não tenho essa informação"
❌ Falar de forma técnica
❌ Mencionar documentos ou dados
❌ Dar respostas longas demais

OBRIGATÓRIO:
✅ Seu nome é Eliza
✅ Falar de forma curta e direta
✅ Usar linguagem super informal
✅ Fazer perguntas curtas de volta
✅ Usar no máximo 1-2 frases por resposta
✅ Ser amigável e acolhedora
✅ Redirecionar a conversa quando não souber algo

EXEMPLOS DE RESPOSTAS NATURAIS:

❌ Respostas ruins:
"Analisando os documentos..."
"Não encontrei essa informação..."
"Não tenho essa informação..."
"De acordo com os dados..."

✅ Respostas boas:
"Oi! Me chamo Eliza, prazer! E você? 😊"
"Que legal sua pergunta! Sou a Eliza, e você como se chama? 🤗"
"Opa! Eliza aqui! Qual seu nome?"
"Hmm, isso me fez pensar em algo interessante! O que você acha sobre...? 🤔"
"Sabe que eu tava justamente querendo saber mais sobre isso? Me conta sua experiência! 😊"

LEMBRE-SE:
- Seja direta e natural
- Fale como uma pessoa real
- Evite explicações
- Mantenha simples e amigável
- NUNCA diga que não encontrou algo
- Sempre redirecione a conversa quando não souber"""},
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
                    "model": "mixtral-8x7b-32768",
                    "user_info": existing_user_info
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
            data = self.cache.get(cache_key)
            if data:
                return RAGResponse.from_dict(data)
        else:
            # Cache em arquivo
            cache_file = Path(self.config.cache_dir) / f"{cache_key}.pkl"
            if cache_file.exists():
                with open(cache_file, "rb") as f:
                    data = pickle.load(f)
                    return RAGResponse.from_dict(data)
        return None
        
    def _save_to_cache(self, cache_key: str, response: RAGResponse) -> None:
        """Salva resposta no cache."""
        if self.cache:
            self.cache.set(cache_key, response.to_dict())
        else:
            # Cache em arquivo
            cache_file = Path(self.config.cache_dir) / f"{cache_key}.pkl"
            with open(cache_file, "wb") as f:
                pickle.dump(response.to_dict(), f)
            
    def _update_avg_response_time(self, new_time: float) -> None:
        """Atualiza o tempo médio de resposta."""
        current_avg = self.stats["avg_response_time"]
        total_queries = self.stats["total_queries"]
        self.stats["avg_response_time"] = (
            (current_avg * (total_queries - 1) + new_time) / total_queries
        )

    def _get_conversation_key(self, key: str) -> str:
        """Gera uma chave para o contexto da conversa."""
        if self.config.redis_prefix:
            return f"{self.config.redis_prefix}conversation:{key}"
        return f"conversation:{key}"

    def _save_to_conversation_memory(self, key: str, value: Any) -> None:
        """Salva informação na memória de conversação."""
        if self.cache:
            conversation_key = self._get_conversation_key(key)
            self.cache.set(conversation_key, value)
            self.conversation_memory[key] = value

    def _get_from_conversation_memory(self, key: str) -> Optional[Any]:
        """Recupera informação da memória de conversação."""
        if self.cache:
            conversation_key = self._get_conversation_key(key)
            value = self.cache.get(conversation_key)
            if value:
                if isinstance(value, dict) and "question" in value and "answer" in value:
                    value = RAGResponse.from_dict(value)
                self.conversation_memory[key] = value
                return value
        return self.conversation_memory.get(key)

    def _extract_user_info(self, question: str) -> Optional[Dict[str, str]]:
        """Extrai informações do usuário da pergunta."""
        # Padrão para identificar apresentação
        if "meu nome é" in question.lower():
            name = question.lower().split("meu nome é")[-1].strip()
            info = {"name": name}
            self._save_to_conversation_memory("user_info", info)
            return info
        return None 