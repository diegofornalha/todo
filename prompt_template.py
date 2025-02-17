from langchain_core.prompts import PromptTemplate

# Template básico de QA
TEMPLATE_PROMPT = """
1. Use o contexto fornecido abaixo para responder à pergunta no final.
2. Se você não souber a resposta, diga apenas "Não sei" sem tentar inventar uma resposta.
3. Mantenha a resposta objetiva e limitada a 3-4 frases.
4. Responda sempre em português do Brasil.
5. Seja direto e evite explicações desnecessárias.

Contexto: {context}
Pergunta: {question}

Resposta Objetiva:"""

# Template específico para documentos
DOCUMENT_PROMPT = """
Você é um assistente especializado em análise de documentos.
Sua tarefa é analisar o conteúdo fornecido e responder perguntas de forma precisa.

Regras:
1. Use apenas as informações fornecidas no conteúdo
2. Se a informação não estiver no conteúdo, responda "Não encontrei essa informação no documento"
3. Mantenha as respostas concisas e objetivas
4. Cite a fonte quando relevante
5. Responda sempre em português do Brasil

Conteúdo: {page_content}
Fonte: {source}

Pergunta: {question}

Resposta:"""

# Cria os templates
QA_PROMPT = PromptTemplate.from_template(TEMPLATE_PROMPT)
DOCUMENT_PROMPT = PromptTemplate(
    input_variables=["page_content", "source", "question"],
    template=DOCUMENT_PROMPT
) 