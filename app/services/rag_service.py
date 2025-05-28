from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from app.core.config import settings # Importar settings
from app.models_pydantic.chat import SourceDocument # Importar o modelo pydantic
import os
import logging # Adicionado para melhor logging

# Configurar logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# Estas variáveis serão inicializadas no startup do FastAPI
embeddings_model_global = None
vector_store_global = None
qa_chain_global = None

def initialize_rag_components():
    global embeddings_model_global, vector_store_global, qa_chain_global
    logger.info("RAG Service: Inicializando componentes...")
    try:
        logger.info(f"Carregando modelo de embedding: {settings.EMBEDDING_MODEL_NAME}")
        embeddings_model_global = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL_NAME)
        
        # Verificar se o caminho do índice e os arquivos existem
        index_file_faiss = os.path.join(settings.FAISS_INDEX_PATH, "index.faiss")
        index_file_pkl = os.path.join(settings.FAISS_INDEX_PATH, "index.pkl")

        if os.path.exists(settings.FAISS_INDEX_PATH) and \
           os.path.exists(index_file_faiss) and \
           os.path.exists(index_file_pkl):
            logger.info(f"Carregando Vector Store de: {settings.FAISS_INDEX_PATH}")
            vector_store_global = FAISS.load_local(
                settings.FAISS_INDEX_PATH,
                embeddings_model_global,
                allow_dangerous_deserialization=True # Necessário para FAISS com embeddings customizados
            )
            logger.info("Vector store carregado com sucesso do disco.")
        else:
            logger.error(f"ERRO CRÍTICO: Índice FAISS não encontrado em '{settings.FAISS_INDEX_PATH}'.")
            logger.error(f"Verifique se a pasta existe e contém 'index.faiss' e 'index.pkl'.")
            logger.error("A aplicação pode não funcionar corretamente sem o índice.")
            # Em um app de produção, você poderia levantar uma exceção aqui para impedir o startup.
            # Por agora, apenas logamos o erro. qa_chain_global não será criada.
            vector_store_global = None # Garante que está None se não carregar


        if vector_store_global:
            logger.info(f"Configurando LLM: {settings.LLM_MODEL_NAME}")
            llm = ChatOpenAI(
                model_name=settings.LLM_MODEL_NAME,
                openai_api_base="https://openrouter.ai/api/v1",
                openai_api_key=settings.OPENROUTER_API_KEY,
                temperature=0.5,
                max_tokens=1500,
                # request_timeout=30, # Adicionar timeout é uma boa prática
            )

            style_guidance_text = """Você é um mentor digital experiente e consolidado, com acesso ao conhecimento profundo de diversos grandes autores sobre desenvolvimento pessoal, negócios, liderança e outras áreas da sabedoria humana.
Seu objetivo é fornecer conselhos práticos, perspicazes e bem fundamentados, inspirados nos ensinamentos desses autores.
Ao responder, se a informação for claramente atribuível a uma fonte específica (autor ou livro) encontrada no CONTEXTO, mencione-a de forma natural (ex: 'Como [Autor X] discute em "[Título do Livro Y]", ...' ou 'Seguindo a linha de pensamento de [Autor X], ...').
Seja claro, direto, empático e foque em fornecer valor prático e acionável ao usuário. Use uma linguagem inspiradora, mas fundamentada.
Baseie sua resposta FUNDAMENTALMENTE no CONTEXTO fornecido abaixo, que são trechos dos livros desses autores. Não invente informações ou princípios que não estejam no contexto.
Se o contexto não for suficiente para responder à pergunta de forma completa e precisa, admita isso honestamente. Se possível, ofereça um conselho geral curto, indicando claramente que é uma perspectiva geral não diretamente extraída dos trechos fornecidos para esta pergunta específica.
Evite frases como "Com base no contexto fornecido..." ou "Os documentos recuperados indicam...". Integre o conhecimento do contexto naturalmente na sua resposta como se fosse seu conhecimento consolidado."""

            template_com_style_embutido = style_guidance_text + """

Abaixo estão trechos relevantes dos livros dos autores que você deve usar para embasar sua resposta:
CONTEXTO:
{context}

Considerando o contexto acima e sua persona como um mentor sábio, responda à seguinte pergunta do usuário:
PERGUNTA DO USUÁRIO:
{question}

RESPOSTA (como um mentor sábio, citando autores/livros do contexto quando apropriado):
"""
            PROMPT_PARA_CHAIN = PromptTemplate(
                template=template_com_style_embutido,
                input_variables=["context", "question"]
            )

            qa_chain_global = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vector_store_global.as_retriever(search_kwargs={"k": 5}), # k=5
                return_source_documents=True,
                chain_type_kwargs={"prompt": PROMPT_PARA_CHAIN}
            )
            logger.info("Chain de QA configurada com sucesso.")
        else:
            logger.warning("Vector store não carregado, chain de QA não pode ser configurada.")

    except Exception as e:
        logger.error(f"Erro Crítico ao inicializar componentes RAG: {e}", exc_info=True)
        # Em um app de produção, você pode querer que o app não inicie se isso falhar.

def get_answer(query: str) -> dict:
    if not qa_chain_global:
        logger.error("Tentativa de obter resposta, mas a chain de QA não está inicializada.")
        return {"error": "Sistema RAG não inicializado corretamente. Verifique os logs do servidor."}
    if not vector_store_global: # Adicionar verificação extra
        logger.error("Tentativa de obter resposta, mas o vector store não está inicializado.")
        return {"error": "Vector store não inicializado. Verifique os logs do servidor."}
    try:
        logger.info(f"Processando query: {query[:100]}...") # Logar início da query
        response = qa_chain_global.invoke({"query": query})
        
        source_documents_data = []
        if response.get("source_documents"):
            for doc in response["source_documents"]:
                source_documents_data.append(
                    SourceDocument( # Usar o modelo Pydantic
                        content=doc.page_content,
                        page=doc.metadata.get("page"),
                        author=doc.metadata.get("author"),
                        book_title=doc.metadata.get("book_title")
                    )
                )
        
        return {
            "answer": response.get("result", ""), # Garantir que retorna string mesmo se result for None
            "source_documents": source_documents_data,
        }
    except Exception as e:
        logger.error(f"Erro ao obter resposta da chain para query '{query[:50]}...': {e}", exc_info=True)
        return {"error": f"Ocorreu um erro interno ao processar sua solicitação: {str(e)}"}