import pandas as pd
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
import os

# --- Configurações ---
chroma_db_dir = "./sucupira_chroma_db"  # Deve ser o mesmo diretório usado no script anterior
embedding_model_name = 'paraphrase-MiniLM-L6-v2'  # Deve ser o mesmo modelo usado no script anterior

# 1. Carregar o ChromaDB
print(f"Carregando o ChromaDB de '{chroma_db_dir}'...")
if not os.path.exists(chroma_db_dir):
    print(f"Erro: O diretório '{chroma_db_dir}' não foi encontrado.")
    exit()

# Carregar a função de embedding (deve ser a mesma usada para criar o ChromaDB)
embedding_function = SentenceTransformerEmbeddings(model_name=embedding_model_name)

# Carregar o VectorStore
vectorstore = Chroma(
    persist_directory=chroma_db_dir,
    embedding_function=embedding_function
)
print("ChromaDB carregado com sucesso!")

# 2. Função para buscar a linha mais similar
def buscar_mais_similar(consulta, k=10):
    """
    Busca as k linhas mais similares à consulta no ChromaDB
    
    Args:
        consulta (str): Texto para buscar similaridade
        k (int): Número de resultados a retornar
    
    Returns:
        Lista de dicionários com os resultados
    """
    results = vectorstore.similarity_search_with_score(consulta, k=k)
    
    output = []
    for doc, score in results:
        output.append({
            "Título": doc.metadata.get("Título", "N/A"),
            "Área de Avaliação": doc.metadata.get("Área de Avaliação", "N/A"),
            "ISSN": doc.metadata.get("ISSN", "N/A"),  # <--- ADD THIS LINE
            "Estrato": doc.metadata.get("Estrato", "N/A"), # <--- AND THIS LINE IF YOU WANT ESTRATO
            "Texto Combinado": doc.page_content,
            "Score de Similaridade": float(score)
        })
    
    return output

# 3. Interface para o usuário
print("\nBem-vindo ao buscador de similaridade do dataset Sucupira!")
print("Digite sua consulta (ou 'sair' para terminar):")

while True:
    consulta = input("\nConsulta: ").strip()
    
    if consulta.lower() in ['sair', 'exit', 'quit']:
        print("Encerrando o programa...")
        break
    
    if not consulta:
        print("Por favor, digite uma consulta válida.")
        continue
    
    try:
        resultados = buscar_mais_similar(consulta)
        
        if resultados:
            print(f"\nOs resultados mais similares para '{consulta}':")
            for i, resultado in enumerate(resultados):
                print(f"\n--- Resultado {i+1} ---")
                print(f"- ISSN: {resultado.get('ISSN', 'N/A')}")
                print(f"- Título: {resultado.get('Título', 'N/A')}")
                print(f"- Área de Avaliação: {resultado.get('Área de Avaliação', 'N/A')}")
                print(f"- Estrato: {resultado.get('Estrato', 'N/A')}") # Imprime o Estrato
                print(f"- Similaridade: {resultado.get('Score de Similaridade', 0.0):.4f}")
                print(f"- Texto completo: {resultado.get('Texto Combinado', 'N/A')[:200]}...")
        else:
            print("Nenhum resultado encontrado para sua consulta.")
        
    except Exception as e:
        print(f"Ocorreu um erro ao processar a consulta: {e}")

print("\nAté logo!")