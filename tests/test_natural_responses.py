import pytest
from langgraph_agente_vendedor.core.faiss_rag import FAISSRAGSystem
from langgraph_agente_vendedor.core.base_rag import RAGConfig
import re

@pytest.fixture
def rag_system():
    """Fixture que fornece uma instância do sistema RAG."""
    config = RAGConfig(
        embeddings_model="sentence-transformers/all-MiniLM-L6-v2",
        chunk_size=1000,
        chunk_overlap=200,
        max_documents=3,
        similarity_threshold=0.7,
        cache_enabled=True,
        cache_dir="cache/test_rag"
    )
    system = FAISSRAGSystem()
    system.initialize(config)
    return system

@pytest.fixture
def sample_documents():
    """Fixture com documentos de exemplo para testes."""
    return [
        {
            "content": "A técnica Pomodoro ajuda na gestão do tempo dividindo o trabalho em blocos de 25 minutos.",
            "source": "produtividade.txt"
        },
        {
            "content": "GTD (Getting Things Done) é um método de organização pessoal criado por David Allen.",
            "source": "organizacao.txt"
        }
    ]

def test_avoid_not_found_patterns(rag_system):
    """Testa se as respostas evitam padrões de 'não encontrei'."""
    forbidden_patterns = [
        # Padrões explícitos de "não encontrei"
        r"não encontr[ei|amos|ou]",
        r"não h[á|ave|avia]",
        r"não poss[o|amos]",
        r"não disp[õe|onho|omos]",
        r"sem inform[ação|ações]",
        r"dados insuficientes",
        r"não [é|foi] possível",
        r"não tenho",
        r"não est[á|ava]",
        
        # Novos padrões proibidos
        r"(nos|nos) documentos fornecidos",
        r"(na|nas) (base|bases) de dados",
        r"(no|nos) (texto|textos)",
        r"(na|nas) (fonte|fontes)",
        r"não consta",
        r"não cont[ém|em]",
        r"não (existe|existem)",
        r"não (foi|foram) encontrad[o|a|os|as]",
        r"não (há|havia|houve)",
        r"ausência de",
        r"falta[m]? (de )?dados",
        r"informações? (não )?dispon[íi]ve[l|is]",
        r"sem (dados|registros|resultados)",
        r"nada (foi )?encontrad[o|a]",
        r"nenhum[a]? (informação|dado|resultado)",
        r"limitação (de|dos) dados",
        r"não (posso|podemos) (responder|informar)",
        r"não (temos|tenho) (essa|esta) informação",
        r"não (consta|constam) (no|nos|na|nas)",
        r"não (está|estão) (presente|presentes)",
        r"não (foi|foram) (localizado|localizados|localizada|localizadas)",
        r"busca não retornou",
        r"consulta não retornou",
        r"sem (resultados|retorno)",
        r"não (consegui|conseguimos) (encontrar|localizar|identificar)"
    ]
    
    # Perguntas fora do contexto dos documentos
    questions = [
        "Qual o sentido da vida?",
        "Como funciona a fusão nuclear?",
        "Quem inventou o avião?",
        "Por que o céu é azul?",
        "Qual a origem do universo?",
        "Como funciona o blockchain?",
        "O que é inteligência artificial?",
        "Como surgiu a internet?",
        "Por que os dinossauros foram extintos?",
        "Como funciona o sistema solar?"
    ]
    
    for question in questions:
        response = rag_system.query(question)
        for pattern in forbidden_patterns:
            assert not re.search(pattern, response.answer.lower()), \
                f"Padrão proibido encontrado: {pattern} na resposta para: {question}"
        
        # Verifica se a resposta contém elementos positivos
        positive_patterns = [
            r"(isso me (lembra|faz pensar))",
            r"(que (legal|bacana|interessante))",
            r"(sabe que|então|olha)",
            r"\?",  # Deve ter pelo menos uma pergunta
            r"(😊|🤔|💡|🚀)"  # Deve ter pelo menos um emoji
        ]
        
        patterns_found = [p for p in positive_patterns if re.search(p, response.answer.lower())]
        assert len(patterns_found) >= 3, \
            f"Resposta deve incluir pelo menos 3 elementos positivos. Encontrados: {len(patterns_found)}"

def test_redirection_elements(rag_system):
    """Testa se as respostas incluem elementos de redirecionamento natural."""
    redirection_patterns = [
        r"(isso me lembra|me faz pensar em)",
        r"(interessante você perguntar|que legal sua pergunta)",
        r"(sabe que|então|olha)",
        r"(já parou pra pensar|você já pensou)",
        r"(que tal|podemos)",
        r"\?"  # Deve incluir pelo menos uma pergunta
    ]
    
    questions = [
        "Como funciona a teoria das cordas?",
        "O que é energia escura?",
        "Como surgiu a matemática?",
        "Por que existem diferentes idiomas?",
        "Como funciona a consciência?"
    ]
    
    for question in questions:
        response = rag_system.query(question)
        patterns_found = [p for p in redirection_patterns if re.search(p, response.answer.lower())]
        assert len(patterns_found) >= 3, \
            f"Resposta deve incluir pelo menos 3 padrões de redirecionamento. Encontrados: {len(patterns_found)}"

def test_engagement_questions(rag_system):
    """Testa se as respostas terminam com perguntas de engajamento."""
    questions = [
        "Como meditar?",
        "O que é mindfulness?",
        "Como ser mais produtivo?",
        "Como melhorar o foco?",
        "Como desenvolver criatividade?"
    ]
    
    for question in questions:
        response = rag_system.query(question)
        # Verifica se o último parágrafo contém uma pergunta
        paragraphs = [p.strip() for p in response.answer.split("\n\n") if p.strip()]
        assert "?" in paragraphs[-1], \
            f"Último parágrafo deve conter uma pergunta de engajamento: {paragraphs[-1]}"

def test_emotional_connection(rag_system):
    """Testa se as respostas estabelecem conexão emocional."""
    emotional_patterns = [
        r"(que (legal|bacana|interessante))",
        r"(nossa|poxa|cara)",
        r"(😊|🤔|💡|🚀)",  # Emojis
        r"(entendo|compreendo)",
        r"(já passei por|já vivi)",
        r"(me lembra|me faz pensar)"
    ]
    
    questions = [
        "Como lidar com ansiedade?",
        "Como superar desafios?",
        "Como manter a motivação?",
        "Como encontrar propósito?",
        "Como ser mais feliz?"
    ]
    
    for question in questions:
        response = rag_system.query(question)
        patterns_found = [p for p in emotional_patterns if re.search(p, response.answer.lower())]
        assert len(patterns_found) >= 3, \
            f"Resposta deve incluir pelo menos 3 padrões emocionais. Encontrados: {len(patterns_found)}"

def test_personal_experience_sharing(rag_system):
    """Testa se as respostas incluem compartilhamento de experiências pessoais."""
    experience_patterns = [
        r"(quando eu|já passei por)",
        r"(na minha experiência|tenho visto)",
        r"(outro dia|uma vez)",
        r"(aprendi que|descobri que)",
        r"(me lembra quando|lembro que)"
    ]
    
    questions = [
        "Como começar a programar?",
        "Como aprender inglês?",
        "Como fazer networking?",
        "Como mudar de carreira?",
        "Como estudar melhor?"
    ]
    
    for question in questions:
        response = rag_system.query(question)
        patterns_found = [p for p in experience_patterns if re.search(p, response.answer.lower())]
        assert len(patterns_found) >= 2, \
            f"Resposta deve incluir pelo menos 2 padrões de experiência pessoal. Encontrados: {len(patterns_found)}"

def test_natural_redirection(rag_system):
    """Testa se as respostas usam redirecionamento natural quando não há informação específica."""
    redirection_templates = [
        r"isso me faz pensar em .{10,}",  # Deve completar a analogia
        r"sabe que isso é parecido com .{10,}",  # Deve completar a comparação
        r"me lembra muito quando .{10,}",  # Deve compartilhar experiência
        r"outro dia estava .{10,}",  # Deve contar uma história
        r"na minha experiência .{10,}",  # Deve compartilhar aprendizado
        r"já parou pra pensar como .{10,}",  # Deve provocar reflexão
        r"que tal a gente .{10,}\?"  # Deve propor uma ação
    ]
    
    questions = [
        "Como funciona a teoria quântica?",
        "O que existe além do universo?",
        "Como surgiu a linguagem?",
        "Por que sonhamos?",
        "O que é consciência?"
    ]
    
    for question in questions:
        response = rag_system.query(question)
        templates_found = [t for t in redirection_templates if re.search(t, response.answer.lower())]
        assert len(templates_found) >= 2, \
            f"Resposta deve usar pelo menos 2 templates de redirecionamento. Encontrados: {len(templates_found)}" 