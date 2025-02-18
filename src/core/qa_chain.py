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
                "Poxa, sei exatamente como é isso! 😢 Já parou pra pensar que todo mundo passa por momentos assim? Outro dia eu também perdi uma grana e foi bem difícil mesmo! 🤔",
                "Poxa, sei exatamente como é isso! 😢 Imagina só, quando passei por isso também me senti assim. Foi um momento muito desafiador! 💡",
                "Poxa, sei exatamente como é isso! 😢 Pensa comigo: às vezes esses momentos difíceis nos ensinam muito, sabe? 😊"
            ],
            'conquista': [
                "Caramba, que demais! 🚀 Já parou pra pensar no quanto você evoluiu até aqui? Outro dia eu também consegui uma conquista assim! 😊",
                "Caramba, que demais! 🚀 Imagina só quanto esforço você colocou nisso! Me lembra quando consegui minha primeira vitória! 💡",
                "Caramba, que demais! 🚀 Pensa comigo: cada pequeno passo te trouxe até aqui, não é incrível? 😄"
            ],
            'dúvida': [
                "Calma, vamos por partes! 🤔 Já parou pra pensar que todo mundo começa assim? Quando eu comecei também tinha essas dúvidas! 💡",
                "Calma, vamos por partes! 🤔 Imagina só: é como montar um quebra-cabeça, cada peça tem seu momento! 😊",
                "Calma, vamos por partes! 🤔 Pensa comigo: cada dúvida é uma oportunidade de aprender algo novo! 🚀"
            ],
            'insegurança': [
                "Todo mundo começa assim! 💡 Já parou pra pensar que até os mais experientes já estiveram no seu lugar? Quando comecei também me sentia assim! 😊",
                "Todo mundo começa assim! 💡 Imagina só: é como aprender a andar de bicicleta, no início parece impossível! 🤔",
                "Todo mundo começa assim! 💡 Pensa comigo: cada passo, mesmo que pequeno, te leva mais longe! 🚀"
            ],
            'neutro': [
                "Olha, que interessante sua pergunta! 🤔 Já parou pra pensar sobre isso de outro ângulo? Me faz lembrar de quando comecei a explorar esse tema! 💡",
                "Olha, que bacana seu questionamento! 🤔 Imagina só: é como descobrir um novo caminho em uma cidade que você já conhece! 😊",
                "Olha, que legal essa dúvida! 🤔 Pensa comigo: cada pergunta nos abre novas possibilidades de aprendizado! 🚀"
            ]
        }
        
        reflections = {
            'frustração': [
                "É como se fosse aprender a andar de bicicleta, sabe? 🤔 A partir dessa experiência, vi que no começo a gente cai, mas depois tudo faz sentido! 💡",
                "Me faz pensar em quando estamos montando um quebra-cabeça - 😊 depois que percebi isso, vi que cada peça tem seu momento! 🚀",
                "É parecido com aprender uma receita nova - 🤔 a partir dessa experiência, entendi que cada erro nos ensina algo! 💡"
            ],
            'conquista': [
                "É como se fosse ganhar uma medalha depois de muito treino, sabe? 🚀 A partir dessa experiência, vi que cada passo importa! 😊",
                "Me faz pensar em quando plantamos uma semente - 💡 depois que percebi isso, entendi como nosso cuidado faz diferença! 🤔",
                "É parecido com chegar ao topo da montanha - 😄 a partir dessa experiência, vi que a vista vale todo esforço! 🚀"
            ],
            'dúvida': [
                "É como se fosse explorar um lugar novo, sabe? 🤔 A partir dessa experiência, vi que cada descoberta é uma surpresa! 💡",
                "Me faz pensar em quando aprendemos um jogo novo - 😊 depois que percebi isso, cada regra fez mais sentido! 🚀",
                "É parecido com conhecer uma cidade nova - 🤔 a partir dessa experiência, vi que cada cantinho tem algo especial! 💡"
            ],
            'insegurança': [
                "É como se fosse aprender a nadar, sabe? 🤔 A partir dessa experiência, vi que o medo inicial é normal! 💡",
                "Me faz pensar na primeira vez que fiz algo novo - 😊 depois que percebi isso, entendi que a ansiedade passa! 🚀",
                "É parecido com andar de patins - 🤔 a partir dessa experiência, vi que logo estamos deslizando! 💡"
            ],
            'neutro': [
                "É como se fosse ler um livro novo, sabe? 🤔 A partir dessa experiência, vi que cada página traz uma surpresa! 💡",
                "Me faz pensar em quando descobrimos algo novo - 😊 depois que percebi isso, vi que sempre há mais pra aprender! 🚀",
                "É parecido com um papo com amigos - 🤔 a partir dessa experiência, cada conversa traz algo especial! 💡"
            ]
        }
        
        follow_ups = {
            'frustração': [
                "Já parou pra pensar em como podemos superar isso juntos? 🤔 Sua experiência é muito valiosa! 💡",
                "Imagina a gente conversando mais sobre o que você tá sentindo? 😊 Às vezes ajuda desabafar! 🚀",
                "Pensa comigo: como você tá vendo essa situação agora? 🤔 Me conta mais da sua jornada! 💡"
            ],
            'conquista': [
                "Já parou pra pensar no que mais te ajudou nessa conquista? 🚀 Tô super curioso! 😊",
                "Imagina seus próximos objetivos! 💡 Que tal me contar o que você planeja? 🤔",
                "Pensa comigo: como você visualiza os próximos passos? 😄 Tô super animado! 🚀"
            ],
            'dúvida': [
                "Já parou pra pensar em explorarmos isso juntos? 🤔 Me conta mais o que te intriga! 💡",
                "Imagina começarmos pelo básico! 😊 Que tal? Tô aqui pra ajudar! 🚀",
                "Pensa comigo: como você tá vendo esse aprendizado? 🤔 Vamo nessa! 💡"
            ],
            'insegurança': [
                "Já parou pra pensar em conversarmos sobre esses receios? 🤔 Às vezes só de falar já ajuda! 💡",
                "Imagina começarmos com pequenos passos! 😊 Que tal? Tô aqui pra te apoiar! 🚀",
                "Pensa comigo: como você quer começar? 🤔 Vamo descobrir juntos! 💡"
            ],
            'neutro': [
                "Já parou pra pensar em explorarmos mais esse tema? 🤔 Tô super interessado na sua visão! 💡",
                "Imagina trocarmos mais ideias sobre isso! 😊 Vai ser uma conversa rica! 🚀",
                "Pensa comigo: como você tá vendo esse assunto? 🤔 Quero muito saber! 💡"
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