import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
import os # Importar para gerenciar o diretório do ChromaDB

# --- Configurações ---
# Nome do arquivo CSV de entrada
input_csv_file = 'sucupira.csv'
# Diretório onde o ChromaDB será salvo
chroma_db_dir = "./sucupira_chroma_db"
# Modelo de embedding
embedding_model_name = 'paraphrase-MiniLM-L6-v2'

# 1. Carregar o arquivo CSV
try:
    df = pd.read_csv(input_csv_file)
    print(f"Arquivo '{input_csv_file}' carregado com sucesso.")
except FileNotFoundError:
    print(f"Erro: O arquivo '{input_csv_file}' não foi encontrado. Por favor, verifique o caminho.")
    exit() # Encerrar o script se o arquivo não for encontrado

# 2. Combinar as duas colunas em uma única string
df['texto_combinado'] = df['Título'] + " " + df['Área de Avaliação']

# Opcional: Remover linhas onde o texto combinado possa ser vazio ou nulo
initial_rows = len(df)
df.dropna(subset=['texto_combinado'], inplace=True)
if len(df) < initial_rows:
    print(f"Foram removidas {initial_rows - len(df)} linhas com 'texto_combinado' vazio ou nulo.")
else:
    print("Nenhuma linha com 'texto_combinado' vazio ou nulo encontrada.")

# 3. Carregar um modelo de embedding pré-treinado
print(f"\nCarregando o modelo de embedding '{embedding_model_name}'. Isso pode levar um momento na primeira vez...")
model = SentenceTransformer(embedding_model_name)
print("Modelo carregado com sucesso!")

# 4. Gerar os embeddings
print("Gerando os embeddings. Aguarde...")
embeddings = model.encode(df['texto_combinado'].tolist(), show_progress_bar=True)
print("Embeddings gerados!")

# 5. Preparar os dados para o ChromaDB
# O ChromaDB precisa dos textos e dos metadados (informações adicionais sobre cada texto)
documents = df['texto_combinado'].tolist()
# Criamos metadados a partir de outras colunas do DataFrame, úteis para recuperação futura
metadatas = df[['Título', 'Área de Avaliação', 'ISSN', 'Estrato']].to_dict(orient='records')

# 6. Inicializar a função de embedding para o ChromaDB (usando o mesmo modelo)
# É fundamental que a função de embedding usada para criar e consultar o ChromaDB seja a mesma.
embedding_function_chroma = SentenceTransformerEmbeddings(model_name=embedding_model_name)

# 7. Salvar os embeddings diretamente no ChromaDB
print(f"\nSalvando os embeddings no ChromaDB em '{chroma_db_dir}'. Isso pode levar um tempo...")

# Remover o diretório existente do ChromaDB, se houver, para evitar conflitos ou dados antigos
if os.path.exists(chroma_db_dir):
    import shutil
    shutil.rmtree(chroma_db_dir)
    print(f"Diretório existente '{chroma_db_dir}' removido para recriação.")

# Criar e persistir o VectorStore
vectorstore = Chroma.from_texts(
    texts=documents,
    embedding=embedding_function_chroma,
    metadatas=metadatas,
    persist_directory=chroma_db_dir
)
print("Embeddings salvos no ChromaDB com sucesso!")

# Para verificar se foi salvo, você pode carregar e fazer uma busca simples (opcional)
print("\nVerificando o ChromaDB (carregando e fazendo uma busca de teste)...")
loaded_vectorstore = Chroma(
    persist_directory=chroma_db_dir,
    embedding_function=embedding_function_chroma
)

query = "cursos de medicina"
results = loaded_vectorstore.similarity_search_with_score(query, k=3) # Buscar 3 resultados mais similares
print(f"\nResultados da busca por '{query}':")
for doc, score in results:
    print(f"- Título: {doc.metadata.get('Título', 'N/A')}")
    print(f"  Área: {doc.metadata.get('Área de Avaliação', 'N/A')}")
    print(f"  Conteúdo: {doc.page_content[:100]}...")
    print(f"  Score de similaridade: {score:.4f}")
    print("---")

print("\nProcesso concluído. O ChromaDB está pronto para uso!")