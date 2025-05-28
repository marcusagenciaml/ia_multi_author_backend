# Use uma imagem base oficial do Python
FROM python:3.11-slim

# Defina o diretório de trabalho dentro do contêiner
WORKDIR /app

# Copie o arquivo de dependências primeiro para aproveitar o cache do Docker
COPY requirements.txt .

# Instale as dependências
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copie todo o diretório da aplicação (incluindo a pasta do índice FAISS)
COPY . .
# A pasta 'faiss_index_multi_author/' será copiada aqui se estiver na raiz do contexto do build.

# Exponha a porta que o Uvicorn usará (Render define $PORT, mas Uvicorn pode usar um fixo)
EXPOSE 8000 

# Comando para rodar a aplicação quando o contêiner iniciar
# Render.com geralmente define a variável de ambiente PORT.
# Uvicorn precisa de --host 0.0.0.0 para ser acessível de fora do contêiner.
# O Render vai mapear a porta externa para a porta 8000 do seu contêiner.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]