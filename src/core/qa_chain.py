# Standard library imports
import logging
from typing import List, Dict, Any
import textwrap
import random

# Local imports
from .vector_store import VectorStore
from ..models.base_handler import BaseLLMHandler
from ..utils.logging_config import setup_logger

logger = setup_logger('qa_chain', 'logs/qa_chain.log')

class QAChain:
    def __init__(
        self,
        llm_handler: BaseLLMHandler,
        embeddings_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        """
        Inicializa o sistema de QA.
        
        Args:
            llm_handler: Handler do modelo de linguagem
            embeddings_model: Nome do modelo de embeddings
        """
        self.llm_handler = llm_handler
        self.vector_store = VectorStore(model_name=embeddings_model)
        logger.info(f"QAChain inicializado com modelo de embeddings: {embeddings_model}")
        
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Adiciona documentos ao sistema.
        
        Args:
            documents: Lista de documentos com conteúdo e metadados
        """
        logger.info(f"Adicionando {len(documents)} documentos ao sistema")
        self.vector_store.add_documents(documents)
        logger.info("Documentos adicionados com sucesso")
        
    def query(
        self,
        question: str,
        k: int = 3,
        include_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Processa uma pergunta e retorna a resposta.
        
        Args:
            question: Pergunta a ser respondida
            k: Número de documentos a recuperar
            include_sources: Se deve incluir as fontes
            
        Returns:
            Dicionário com a pergunta, resposta e metadados
        """
        logger.info(f"Processando pergunta: {question}")
        
        try:
            # Recupera documentos relevantes
            results = self.vector_store.similarity_search(question, k=k)
            
            if not results:
                logger.warning("Nenhum documento relevante encontrado")
                
                # Identifica o tom emocional da pergunta
                emotional_tone = self._identify_emotional_tone(question)
                
                # Seleciona uma resposta base de acordo com o tom
                base_response = self._get_base_response(emotional_tone, question)
                
                # Adiciona variação natural
                response = self._add_natural_variation(base_response)
                
                return {
                    "pergunta": question,
                    "resposta": response,
                    "status": "redirecionamento"
                }
            
            # Formata o contexto
            context = self._format_context(results)
            
            # Processa a pergunta com o LLM
            response = self.llm_handler.process_question(question, context)
            
            # Adiciona informações sobre as fontes
            if include_sources:
                sources = []
                for doc in results:
                    if 'metadata' in doc and 'source' in doc['metadata']:
                        sources.append(doc['metadata']['source'])
                response['sources'] = list(set(sources))
            
            response['status'] = 'sucesso'
            logger.info("Resposta gerada com sucesso")
            return response
            
        except Exception as e:
            logger.error(f"Erro ao processar pergunta: {str(e)}")
            return {
                "pergunta": question,
                "resposta": textwrap.dedent(f"""
                    Poxa, sabe que agora deu um pequeno probleminha técnico aqui? 🤔
                    Mas não se preocupa! Que tal a gente tentar de novo? Tenho certeza que vamos ter um papo super legal sobre {question.lower()}! 😊
                """).strip(),
                "status": "erro",
                "erro": str(e)
            }
            
    def _format_context(self, results: List[Dict[str, Any]]) -> str:
        """
        Formata os resultados da busca em um contexto.
        
        Args:
            results: Lista de documentos com scores
            
        Returns:
            Texto formatado com o contexto
        """
        formatted_docs = []
        for i, doc in enumerate(results):
            source = doc.get('metadata', {}).get('source', 'Desconhecida')
            score = doc.get('score', 0)
            formatted_docs.append(
                f"Documento {i+1} (Fonte: {source}, Relevância: {score:.2f}):\n{doc['content']}"
            )
        return "\n\n".join(formatted_docs)
        
    def save_index(self, path: str) -> None:
        """
        Salva o índice em disco.
        
        Args:
            path: Caminho para salvar
        """
        logger.info(f"Salvando índice em: {path}")
        self.vector_store.save(path)
        logger.info("Índice salvo com sucesso")
        
    def load_index(self, path: str) -> None:
        """
        Carrega o índice do disco.
        
        Args:
            path: Caminho para carregar
        """
        logger.info(f"Carregando índice de: {path}")
        self.vector_store.load(path)
        logger.info("Índice carregado com sucesso")

    def _identify_emotional_tone(self, question: str) -> str:
        """
        Identifica o tom emocional da pergunta.
        
        Args:
            question: Pergunta do usuário
            
        Returns:
            Tom emocional identificado
        """
        patterns = {
            'frustração': ['perdi', 'ruim', 'problema', 'difícil', 'triste', '😢', '😔'],
            'conquista': ['consegui', 'vitória', 'ganhei', 'sucesso', '🚀', '💪'],
            'dúvida': ['confuso', 'dúvida', 'não sei', 'como', 'ajuda', '🤔'],
            'insegurança': ['medo', 'receio', 'será', 'incerto', 'risco', '😰'],
            'neutro': []
        }
        
        for tone, keywords in patterns.items():
            if any(keyword.lower() in question.lower() for keyword in keywords):
                return tone
        return 'neutro'
        
    def _get_base_response(self, emotional_tone: str, question: str) -> str:
        """
        Retorna uma resposta base de acordo com o tom emocional.
        
        Args:
            emotional_tone: Tom emocional identificado
            question: Pergunta original
            
        Returns:
            Resposta base formatada
        """
        experiences = {
            'frustração': [
                "Poxa, sei exatamente como é isso! 😢 Cara, outro dia eu também perdi uma grana e foi bem difícil mesmo. Já parou pra pensar que todo mundo passa por isso? 🤔",
                "Nossa, sei exatamente como é isso! 😢 Então, uma vez passei por isso também e foi muito desafiador. Imagina só que legal poder trocar essa experiência! 💡",
                "Poxa, sei exatamente como é isso! 😢 Olha, já aconteceu comigo também e foi bem complicado. Pensa só que interessante você trazer esse tema! 😊"
            ],
            'conquista': [
                "Caramba, que demais! 🚀 Cara, outro dia eu também consegui uma conquista assim! Já parou pra pensar no quanto você evoluiu? 😊",
                "Nossa, que demais! 🚀 Então, uma vez consegui algo parecido e foi incrível! Imagina só que bacana ver você crescendo! 💡",
                "Caramba, que demais! 🚀 Olha, já aconteceu comigo também e foi super especial! Pensa só que interessante sua jornada! 😄"
            ],
            'dúvida': [
                "Calma, vamos por partes! 🤔 Cara, outro dia eu também tava assim! Já parou pra pensar como todo mundo começa do zero? 💡",
                "Então, vamos com calma! 🤔 Olha, uma vez passei por isso também! Imagina só que legal sua curiosidade! 😊",
                "Nossa, vamos devagar! 🤔 Sabe, já aconteceu comigo também! Pensa só que interessante sua dúvida! 🚀"
            ],
            'insegurança': [
                "Todo mundo começa assim! 💡 Cara, outro dia eu também me sentia assim! Já parou pra pensar que isso é super normal? 😊",
                "Nossa, todo mundo passa por isso! 💡 Então, uma vez passei por isso também! Imagina só que bacana essa troca! 🤔",
                "Olha, todo mundo tem esse momento! 💡 Sabe, já aconteceu comigo também! Pensa só que interessante sua jornada! 🚀"
            ],
            'neutro': [
                "Olha, que interessante sua pergunta! 🤔 Cara, outro dia tava pensando nisso! Já parou pra pensar como esse tema é legal? 💡",
                "Nossa, que bacana seu questionamento! 🤔 Então, uma vez refleti bastante sobre isso! Imagina só que interessante sua visão! 😊",
                "Poxa, que legal essa dúvida! 🤔 Sabe, já aconteceu comigo também! Pensa só que demais poder conversar sobre isso! 🚀"
            ]
        }
        
        reflections = {
            'frustração': [
                "É como se fosse aprender a andar de bicicleta, sabe? 🤔 Cara, me lembra quando caí várias vezes! Já parou pra pensar como depois tudo faz sentido? 💡",
                "Então, me faz pensar em quando montei meu primeiro quebra-cabeça - 😊 Nossa, outro dia lembrei disso! Imagina só que bacana essa comparação! 🚀",
                "Olha, é parecido com uma vez que tentei uma receita nova - 🤔 Sabe, já aconteceu comigo! Pensa só que interessante essa reflexão! 💡"
            ],
            'conquista': [
                "É como se fosse ganhar uma medalha, sabe? 🚀 Cara, me lembra quando consegui minha primeira vitória! Já parou pra pensar como esse momento é especial? 😊",
                "Então, me faz pensar em quando plantei minha primeira semente - 💡 Nossa, outro dia refleti sobre isso! Imagina só que bacana essa conquista! 🤔",
                "Olha, é parecido com uma vez que cheguei ao topo da montanha - 😄 Sabe, já aconteceu comigo! Pensa só que interessante essa jornada! 🚀"
            ],
            'dúvida': [
                "É como se fosse explorar um lugar novo, sabe? 🤔 Cara, me lembra quando me perdi pela primeira vez! Já parou pra pensar como cada descoberta é especial? 💡",
                "Então, me faz pensar em quando aprendi um jogo novo - 😊 Nossa, outro dia passei por isso! Imagina só que bacana esse processo! 🚀",
                "Olha, é parecido com uma vez que conheci uma cidade nova - 🤔 Sabe, já aconteceu comigo! Pensa só que interessante essa experiência! 💡"
            ],
            'insegurança': [
                "É como se fosse aprender a nadar, sabe? 🤔 Cara, me lembra quando tive medo da água! Já parou pra pensar como é bom superar isso? 💡",
                "Então, me faz pensar em quando tentei algo novo - 😊 Nossa, outro dia refleti sobre isso! Imagina só que bacana essa coragem! 🚀",
                "Olha, é parecido com uma vez que andei de patins - 🤔 Sabe, já aconteceu comigo! Pensa só que interessante esse processo! 💡"
            ],
            'neutro': [
                "É como se fosse ler um livro novo, sabe? 🤔 Cara, me lembra quando descobri minha história favorita! Já parou pra pensar como esse momento é especial? 💡",
                "Então, me faz pensar em quando aprendi algo novo - 😊 Nossa, outro dia pensei nisso! Imagina só que bacana essa reflexão! 🚀",
                "Olha, é parecido com uma vez que tive uma ótima conversa - 🤔 Sabe, já aconteceu comigo! Pensa só que interessante esse tema! 💡"
            ]
        }
        
        follow_ups = {
            'frustração': [
                "Cara, o que você acha de compartilharmos mais sobre como você tá lidando com isso? Já parou pra pensar como é bom ter apoio nesses momentos? 💡",
                "Então, que tal a gente conversar mais sobre o que você tá sentindo? Nossa, outro dia isso me ajudou muito! Imagina só que bacana essa troca! 😊",
                "Olha, como você vê essa situação agora? Sabe, uma vez passei por algo parecido! Pensa só que interessante sua perspectiva! 🤔"
            ],
            'conquista': [
                "Cara, o que você acha que te ajudou mais nessa conquista? Já parou pra pensar no seu progresso? Me lembra quando consegui algo parecido! 🚀",
                "Então, que tal a gente explorar seus próximos objetivos? Nossa, outro dia fiz isso e foi incrível! Imagina só que bacana planejar! 😊",
                "Olha, como você vê os próximos passos? Sabe, uma vez tracei metas assim! Pensa só que interessante sua visão! 💡"
            ],
            'dúvida': [
                "Cara, o que você acha da gente explorar isso juntos? Já parou pra pensar como é bom ter alguém pra ajudar? Me lembra quando tive essas dúvidas! 🤔",
                "Então, que tal a gente começar pelo básico? Nossa, outro dia isso me ajudou muito! Imagina só que bacana esse processo! 💡",
                "Olha, como você vê esse aprendizado? Sabe, uma vez comecei assim também! Pensa só que interessante sua jornada! 😊"
            ],
            'insegurança': [
                "Cara, o que você acha da gente conversar sobre esses receios? Já parou pra pensar como é normal se sentir assim? Me lembra quando precisei de apoio! 💡",
                "Então, que tal a gente pensar em pequenos passos? Nossa, outro dia isso me ajudou muito! Imagina só que bacana esse processo! 🤔",
                "Olha, como você vê o começo dessa jornada? Sabe, uma vez comecei assim também! Pensa só que interessante seu caminho! 😊"
            ],
            'neutro': [
                "Cara, o que você acha da gente explorar mais esse tema? Já parou pra pensar como é legal trocar ideias? Me lembra quando comecei a estudar isso! 🤔",
                "Então, que tal a gente trocar mais ideias sobre isso? Nossa, outro dia tive ótimas reflexões! Imagina só que bacana essa conversa! 💡",
                "Olha, como você vê esse assunto? Sabe, uma vez me aprofundei nisso! Pensa só que interessante sua perspectiva! 😊"
            ]
        }
        
        # Seleciona aleatoriamente uma resposta de cada categoria
        experience = random.choice(experiences[emotional_tone])
        reflection = random.choice(reflections[emotional_tone])
        follow_up = random.choice(follow_ups[emotional_tone])
        
        # Formata a resposta em três parágrafos
        response = f"{experience}\n\n{reflection}\n\n{follow_up}"
        
        return response
        
    def _add_natural_variation(self, response: str) -> str:
        """
        Adiciona variações naturais à resposta.
        
        Args:
            response: Resposta base
            
        Returns:
            Resposta com variações naturais
        """
        # Adiciona variações sutis
        variations = {
            'exatamente': ['justamente', 'precisamente', 'realmente'],
            'interessante': ['fascinante', 'curioso', 'incrível'],
            'legal': ['bacana', 'massa', 'maneiro'],
            'muito': ['super', 'bastante', 'bem']
        }
        
        varied_response = response
        for word, alternatives in variations.items():
            if word in varied_response.lower():
                varied_response = varied_response.replace(
                    word,
                    random.choice(alternatives),
                    1  # Substitui apenas a primeira ocorrência
                )
                
        return varied_response 