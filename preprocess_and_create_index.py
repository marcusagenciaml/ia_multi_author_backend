import os
import json
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter # Atualizado para o novo local
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import time # Para medir o tempo

# --- CONFIGURAÇÕES ---
PDF_SOURCES_DIR = "pdf_sources"
PDF_METADATA_FILE = "pdf_metadata.json"
# Modelo de embedding que planejamos usar no backend também
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
# Nome da pasta onde o índice FAISS será salvo
FAISS_INDEX_PATH = "faiss_index_multi_author" 

def load_pdf_metadata():
    if os.path.exists(PDF_METADATA_FILE):
        with open(PDF_METADATA_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    print(f"AVISO: Arquivo de metadados '{PDF_METADATA_FILE}' não encontrado. Os metadados de autor/livro usarão padrões.")
    return {}

def create_index():
    start_time = time.time()
    print("Iniciando processo de criação do índice FAISS...")
    pdf_metadata = load_pdf_metadata()
    all_docs_split = [] # Lista para guardar todos os chunks de todos os livros

    if not os.path.exists(PDF_SOURCES_DIR) or not os.listdir(PDF_SOURCES_DIR):
        print(f"ERRO: Diretório de PDFs '{PDF_SOURCES_DIR}' não encontrado ou está vazio.")
        print("Por favor, adicione seus arquivos PDF a esta pasta.")
        return

    print(f"Procurando PDFs em: {os.path.abspath(PDF_SOURCES_DIR)}")

    for pdf_filename in os.listdir(PDF_SOURCES_DIR):
        if pdf_filename.lower().endswith(".pdf"):
            file_path = os.path.join(PDF_SOURCES_DIR, pdf_filename)
            print(f"\nProcessando PDF: {pdf_filename}...")
            
            try:
                loader = PyMuPDFLoader(file_path)
                documents_from_pdf = loader.load() # Lista de Document, um por página

                # Adicionar metadados de autor e título
                # Se o nome do arquivo não estiver no JSON, usa valores padrão
                metadata_for_file = pdf_metadata.get(pdf_filename, {})
                author = metadata_for_file.get("author", "Autor Desconhecido")
                book_title = metadata_for_file.get("book_title", pdf_filename.replace('.pdf', '').replace('_', ' ').title())

                print(f"  Autor: {author}, Título: {book_title}")

                for doc_page in documents_from_pdf:
                    doc_page.metadata["author"] = author
                    doc_page.metadata["book_title"] = book_title
                    # doc_page.metadata["source_pdf"] = pdf_filename # Adicionar nome do arquivo de origem também

                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=CHUNK_SIZE,
                    chunk_overlap=CHUNK_OVERLAP,
                    length_function=len,
                    is_separator_regex=False, # Default é False, bom para texto geral
                )
                docs_split_from_file = text_splitter.split_documents(documents_from_pdf)
                all_docs_split.extend(docs_split_from_file)
                print(f"  '{pdf_filename}' dividido em {len(docs_split_from_file)} chunks.")
            except Exception as e:
                print(f"  ERRO ao processar '{pdf_filename}': {e}")
                continue # Pula para o próximo arquivo se houver erro em um

    if not all_docs_split:
        print("Nenhum documento foi processado com sucesso. O índice não será criado.")
        return

    print(f"\nTotal de chunks de todos os PDFs a serem indexados: {len(all_docs_split)}")

    print(f"Carregando modelo de embedding: {EMBEDDING_MODEL_NAME}...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    print("Criando índice FAISS a partir dos documentos... Isso pode levar vários minutos dependendo do número de chunks.")
    vector_store = FAISS.from_documents(all_docs_split, embeddings)

    # Criar a pasta do índice se não existir
    if not os.path.exists(FAISS_INDEX_PATH):
        os.makedirs(FAISS_INDEX_PATH)
        print(f"Pasta do índice '{FAISS_INDEX_PATH}' criada.")

    print(f"Salvando índice FAISS em: {FAISS_INDEX_PATH}")
    vector_store.save_local(FAISS_INDEX_PATH)
    
    end_time = time.time()
    print("\n--- RESUMO DA CRIAÇÃO DO ÍNDICE ---")
    print(f"Índice FAISS criado e salvo com sucesso em '{FAISS_INDEX_PATH}'.")
    print(f"Tempo total de processamento: {end_time - start_time:.2f} segundos.")
    print(f"Número total de chunks indexados: {len(all_docs_split)}")
    
    # Verificar o tamanho da pasta do índice (aproximado)
    try:
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(FAISS_INDEX_PATH):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total_size += os.path.getsize(fp)
        print(f"Tamanho aproximado da pasta do índice: {total_size / (1024 * 1024):.2f} MB")
    except Exception as e:
        print(f"Não foi possível calcular o tamanho da pasta do índice: {e}")


if __name__ == "__main__":
    print("Certifique-se de ter criado o arquivo 'pdf_metadata.json' e colocado os PDFs na pasta 'pdf_sources'.")
    create_index()