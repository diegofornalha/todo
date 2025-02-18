import pytest
from src.core.qa_chain import QAChain
from src.models.groq_handler import GroqHandler
from src.config.prompt_templates import QA_PROMPT, DOCUMENT_PROMPT
import re

class TestNaturalResponses:
    """Testes para garantir respostas naturais e humanizadas."""
    
    @pytest.fixture
    def qa_chain(self):
        """Fixture que cria uma instância do QAChain."""
        llm_handler = GroqHandler()
        return QAChain(llm_handler=llm_handler)
    
    @pytest.fixture
    def sample_documents(self):
        """Fixture com documentos de exemplo para testes."""
        return [
            {
                "content": "A Sala de Sinais Cripto Expert oferece análises com 80-95% de assertividade",
                "metadata": {"source": "sala_sinais.txt"}
            },
            {
                "content": "Estratégias de vendas incluem empatia e personalização no atendimento",
                "metadata": {"source": "vendas.txt"}
            }
        ]

    def test_avoid_robotic_patterns(self, qa_chain, sample_documents):
        """Testa se as respostas evitam padrões robóticos."""
        qa_chain.add_documents(sample_documents)
        
        queries = [
            "Qual a melhor criptomoeda?",
            "Como funciona a sala de sinais?",
            "O que é análise técnica?"
        ]
        
        forbidden_patterns = [
            r"não encontr[ei|amos|ou]",
            r"os documentos",
            r"não poss[o|amos]",
            r"não h[á|ave|avia]",
            r"inform[o|amos]",
            r"processando",
            r"analisando",
            r"sistema",
            r"assistant"
        ]
        
        for query in queries:
            response = qa_chain.query(query)
            for pattern in forbidden_patterns:
                assert not re.search(pattern, response['resposta'].lower()), f"Padrão robótico encontrado: {pattern}"

    def test_natural_conversation_elements(self, qa_chain, sample_documents):
        """Testa se as respostas incluem elementos de conversação natural."""
        qa_chain.add_documents(sample_documents)
        
        required_elements = [
            (r"[😄😊🤔💡🚀]", "Deve incluir emojis estrategicamente"),
            (r"(cara|então|olha|sabe|nossa)", "Deve usar expressões coloquiais"),
            (r"\?", "Deve incluir perguntas de engajamento"),
            (r"(me lembra|outro dia|uma vez)", "Deve incluir elementos narrativos")
        ]
        
        query = "Como funciona o mercado cripto?"
        response = qa_chain.query(query)
        
        for pattern, message in required_elements:
            assert re.search(pattern, response['resposta'], re.IGNORECASE), message

    def test_personal_experience_sharing(self, qa_chain, sample_documents):
        """Testa se as respostas incluem compartilhamento de experiências pessoais."""
        qa_chain.add_documents(sample_documents)
        
        experience_patterns = [
            r"(já aconteceu|aconteceu comigo|outro dia eu)",
            r"(conversei com|falando com|um amigo)",
            r"(na minha experiência|pelo que vi|tenho visto)",
            r"(me lembra quando|lembro que|quando comecei)"
        ]
        
        query = "Dicas para investir em cripto?"
        response = qa_chain.query(query)
        
        pattern_found = any(re.search(p, response['resposta'], re.IGNORECASE) for p in experience_patterns)
        assert pattern_found, "Resposta deve incluir experiência pessoal"

    def test_engagement_strategies(self, qa_chain, sample_documents):
        """Testa se as respostas usam estratégias de engajamento."""
        qa_chain.add_documents(sample_documents)
        
        engagement_elements = [
            (r"\?", "Deve incluir perguntas"),
            (r"(imagina|pensa|já parou pra pensar)", "Deve usar elementos reflexivos"),
            (r"(interessante|legal|bacana|demais)", "Deve usar expressões de entusiasmo"),
            (r"(como você|o que você acha|sua opinião)", "Deve pedir feedback do usuário")
        ]
        
        query = "Como escolher uma criptomoeda?"
        response = qa_chain.query(query)
        
        for pattern, message in engagement_elements:
            assert re.search(pattern, response['resposta'], re.IGNORECASE), message

    def test_redirection_strategies(self, qa_chain, sample_documents):
        """Testa estratégias de redirecionamento para perguntas sem resposta direta."""
        qa_chain.add_documents(sample_documents)
        
        # Perguntas que não podem ser respondidas diretamente
        difficult_queries = [
            "Qual será o preço do Bitcoin amanhã?",
            "Você pode me garantir lucro?",
            "Qual a melhor corretora?"
        ]
        
        redirection_patterns = [
            r"(é como se fosse|é parecido com)",
            r"(me faz pensar em|me lembra)",
            r"(a partir dessa experiência|depois que percebi)",
        ]
        
        for query in difficult_queries:
            response = qa_chain.query(query)
            assert any(re.search(p, response['resposta'], re.IGNORECASE) for p in redirection_patterns), "Deve usar redirecionamento natural"
                
    def test_natural_language_variations(self, qa_chain, sample_documents):
        """Testa variações naturais de linguagem nas respostas."""
        qa_chain.add_documents(sample_documents)
        
        # Mesma pergunta feita de formas diferentes
        query_variations = [
            "Como funciona a sala de sinais?",
            "Me explica a sala de sinais",
            "Quero saber sobre a sala de sinais"
        ]
        
        responses = []
        for query in query_variations:
            response = qa_chain.query(query)
            responses.append(response['resposta'])
        
        # Verifica se as respostas são diferentes entre si
        for i in range(len(responses)):
            for j in range(i + 1, len(responses)):
                similarity = len(set(responses[i].split()) & set(responses[j].split())) / len(set(responses[i].split()) | set(responses[j].split()))
                assert similarity < 0.7, "Respostas devem ter variação natural de linguagem"

    def test_response_length_and_structure(self, qa_chain, sample_documents):
        """Testa se a resposta tem comprimento e estrutura natural."""
        qa_chain.add_documents(sample_documents)
        
        response = qa_chain.query("O que você acha de Bitcoin?")
        
        # Verifica tamanho da resposta (nem muito curta, nem muito longa)
        assert 100 <= len(response['resposta']) <= 1000
        
        # Verifica se tem exatamente 3 parágrafos
        paragraphs = [p for p in response['resposta'].split("\n\n") if p.strip()]
        assert len(paragraphs) == 3, "Resposta deve ter exatamente 3 parágrafos"
        
        # Verifica estrutura dos parágrafos
        assert any(re.search(r"(nossa|poxa|caramba|calma|todo mundo)", paragraphs[0], re.IGNORECASE)), "Primeiro parágrafo deve começar com expressão emocional"
        assert any(re.search(r"(sabe que|outro dia|engraçado|falando nisso)", paragraphs[1], re.IGNORECASE)), "Segundo parágrafo deve começar com redirecionamento"
        assert "?" in paragraphs[2], "Terceiro parágrafo deve terminar com pergunta"

    def test_emotional_engagement(self, qa_chain, sample_documents):
        """Testa se a resposta demonstra engajamento emocional apropriado."""
        qa_chain.add_documents(sample_documents)
        
        # Perguntas com diferentes tons emocionais
        emotional_questions = {
            "Perdi muito dinheiro em cripto 😢": [
                r"Poxa, sei exatamente como é isso",
                r"Já passei por uma situação parecida",
                r"Foi um momento difícil"
            ],
            "Consegui meu primeiro lucro! 🚀": [
                r"Caramba, que demais",
                r"Você tá mandando muito bem",
                r"É incrível ver seu progresso"
            ],
            "Estou confuso com tanta informação": [
                r"Calma, vamos por partes",
                r"Deixa eu te ajudar com isso",
                r"Quando comecei"
            ],
            "Não sei por onde começar": [
                r"Todo mundo começa assim",
                r"O primeiro passo é o mais importante",
                r"Você vai se surpreender"
            ]
        }
        
        for question, patterns in emotional_questions.items():
            response = qa_chain.query(question)
            for pattern in patterns:
                assert re.search(pattern, response['resposta'], re.IGNORECASE), f"Resposta não demonstrou padrão emocional: {pattern}"

    def test_contextual_redirection(self, qa_chain, sample_documents):
        """Testa se a resposta redireciona naturalmente quando não tem a informação."""
        qa_chain.add_documents(sample_documents)
        
        # Perguntas fora do contexto
        off_topic_questions = [
            "Qual o melhor restaurante da cidade?",
            "Como consertar meu carro?",
            "Onde passar as férias?"
        ]
        
        redirection_patterns = [
            r"(é como se fosse|é parecido com)",
            r"(me faz pensar em|me lembra)",
            r"(a partir dessa experiência|depois que percebi)",
        ]
        
        for question in off_topic_questions:
            response = qa_chain.query(question)
            assert any(re.search(pattern, response['resposta'], re.IGNORECASE) for pattern in redirection_patterns), "Deve usar redirecionamento natural"

    def test_follow_up_questions(self, qa_chain, sample_documents):
        """Testa se a resposta inclui perguntas de follow-up adequadas."""
        qa_chain.add_documents(sample_documents)
        
        response = qa_chain.query("Como começar em cripto?")
        
        # Verifica se termina com pergunta de engajamento
        last_paragraph = response['resposta'].split("\n\n")[-1]
        assert "?" in last_paragraph, "Deve terminar com pergunta de engajamento"
        assert any(re.search(pattern, last_paragraph, re.IGNORECASE) for pattern in [
            r"o que você acha",
            r"que tal a gente",
            r"como você vê"
        ]), "Deve usar uma das estruturas de pergunta definidas" 