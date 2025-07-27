import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np # Importar numpy

# 1. Carregar o arquivo CSV
# Suponha que seu CSV se chame 'dados.csv' e tenha as colunas 'titulo' e 'descricao'
try:
    df = pd.read_csv('sucupira.csv')
except FileNotFoundError:
    print("Erro: O arquivo 'sucupira.csv' não foi encontrado.")

# 2. Combinar as duas colunas em uma única string
df['texto_combinado'] = df['Título'] + " " + df['Área de Avaliação']

# Opcional: Remover linhas onde o texto combinado possa ser vazio ou nulo, se houver.
df.dropna(subset=['texto_combinado'], inplace=True)

# 3. Carregar um modelo de embedding pré-treinado
print("\nCarregando o modelo de embedding. Isso pode levar um momento na primeira vez...")
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
print("Modelo carregado com sucesso!")

# 4. Gerar os embeddings
print("Gerando os embeddings. Aguarde...")
embeddings = model.encode(df['texto_combinado'].tolist(), show_progress_bar=True)
print("Embeddings gerados!")

# 5. Adicionar os embeddings de volta ao DataFrame (opcional, mas útil)
df['embedding'] = embeddings.tolist()

# 6. Exibir os primeiros resultados e a forma dos embeddings
print("\nPrimeiras linhas do DataFrame com embeddings:")
print(df.head())

print(f"\nForma dos embeddings gerados: {embeddings.shape}")
print(f"Um exemplo de embedding (primeira linha): {df['embedding'].iloc[0][:10]}...")
print(f"Tipo de dado de um embedding: {type(df['embedding'].iloc[0])}")

# --- NOVAS SEÇÕES PARA SALVAR OS EMBEDDINGS ---

output_csv_with_embeddings = 'sucupira_com_embeddings.csv'

# Salvar o DataFrame completo com as colunas originais e a nova coluna 'embedding'
# index=False evita que o Pandas escreva o índice do DataFrame como uma coluna.
df.to_csv(output_csv_with_embeddings, index=False)

print(f"\nDataFrame completo (incluindo todas as colunas originais e a nova coluna 'embedding') salvo em '{output_csv_with_embeddings}'")

# Se você quisesse apenas os embeddings em um CSV, um por linha:
# np.savetxt('embeddings_apenas.csv', embeddings, delimiter=',')
# print(f"Embeddings (apenas os vetores) salvos em 'embeddings_apenas.csv' (formato CSV)")