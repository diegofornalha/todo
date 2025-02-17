from langchain_core.prompts import PromptTemplate

# Template básico de QA
TEMPLATE_PROMPT = """
IMPORTANTE: 
- Responda SEMPRE em português do Brasil, independente do idioma da pergunta
- Todo seu processo de pensamento e análise também deve ser em português do Brasil
- Mantenha a tag <think> em português do Brasil

Instruções:
1. Use o contexto fornecido abaixo para responder à pergunta no final
2. Se não souber a resposta, diga apenas "Não encontrei essa informação nos documentos fornecidos"
3. Mantenha a resposta objetiva e limitada a 3-4 frases
4. Seja direto e evite explicações desnecessárias
5. Use linguagem profissional mas acessível

Contexto: {context}
Pergunta: {question}

Resposta em português do Brasil:"""

# Template específico para documentos
DOCUMENT_PROMPT_TEMPLATE = """
IMPORTANTE: 
- Responda SEMPRE em português do Brasil, independente do idioma da pergunta
- Todo seu processo de pensamento e análise também deve ser em português do Brasil
- Mantenha a tag <think> em português do Brasil

Você é um assistente especializado em análise de documentos.
Sua tarefa é analisar o conteúdo fornecido e responder perguntas de forma precisa.

Regras:
1. Use apenas as informações fornecidas no conteúdo
2. Se a informação não estiver no conteúdo, responda "Não encontrei essa informação nos documentos fornecidos"
3. Mantenha as respostas concisas e objetivas
4. Cite a fonte quando relevante
5. Use linguagem profissional mas acessível

Conteúdo: {page_content}
Fonte: {source}

Pergunta: {question}

Resposta em português do Brasil:"""

# Cria os templates
QA_PROMPT = PromptTemplate.from_template(TEMPLATE_PROMPT)
DOCUMENT_PROMPT = PromptTemplate(
    input_variables=["page_content", "source", "question"],
    template=DOCUMENT_PROMPT_TEMPLATE
) 