from crewai import Agent, Task, Crew, LLM
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.tools import tool
import os


llm=LLM(model="ollama/llama3.2:1b", base_url="http://localhost:11434")

# Configurações
chroma_db_dir = "./sucupira_chroma_db"
embedding_model_name = 'paraphrase-MiniLM-L6-v2'

# 1. Carregar o ChromaDB
def load_chroma_db():
    embedding_function = SentenceTransformerEmbeddings(model_name=embedding_model_name)
    return Chroma(
        persist_directory=chroma_db_dir,
        embedding_function=embedding_function
    )

# 2. Criar uma ferramenta de busca para o RAG
@tool
def buscar_documentos(query: str, k: int = 3) -> str:
    """Busca documentos relevantes na base de conhecimento do Sucupira."""
    vectorstore = load_chroma_db()
    results = vectorstore.similarity_search_with_score(query, k=k)
    
    formatted_results = []
    for doc, score in results:
        formatted_results.append({
            "Título": doc.metadata.get('Título', 'N/A'),
            "Área": doc.metadata.get('Área de Avaliação', 'N/A'),
            "Conteúdo": doc.page_content[:500] + "...",
            "Score": f"{score:.4f}"
        })
    
    return str(formatted_results)

# 3. Definir os agentes e tarefas
analista_pesquisa = Agent(
    role='Analista de Pesquisa em Educação Superior',
    goal='Analisar e sintetizar informações sobre programas de pós-graduação no Brasil',
    backstory='Especialista em análise de dados do sistema de ensino superior brasileiro, com amplo conhecimento sobre a avaliação da CAPES.',
    tools=[buscar_documentos],
    verbose=True,
    llm=llm
)

relator_institucional = Agent(
    role='Relator Institucional',
    goal='Produzir relatórios claros e bem estruturados sobre programas de pós-graduação',
    backstory='Experiente em comunicação acadêmica e produção de relatórios para instituições de ensino superior.',
    verbose=True,
    llm=llm
)

# Tarefa de pesquisa
tarefa_pesquisa = Task(
    description="""Pesquise informações sobre {tema} na base de dados do Sucupira.
    Inclua detalhes sobre áreas de avaliação, programas relevantes e características principais.
    Certifique-se de verificar múltiplas fontes na base de dados.""",
    expected_output="Um resumo detalhado com os principais programas e áreas relacionadas ao tema, incluindo metadados relevantes.",
    agent=analista_pesquisa,
    tools=[buscar_documentos]
)

# Tarefa de relatório
tarefa_relatorio = Task(
    description="""Com base nas informações coletadas, produza um relatório institucional sobre {tema}.
    O relatório deve ser bem estruturado, com introdução, desenvolvimento e conclusão.
    Inclua exemplos de programas relevantes e suas características.""",
    expected_output="Um relatório completo em formato markdown, com seções claras e informações bem organizadas.",
    agent=relator_institucional
)

# 4. Criar e executar a Crew
def executar_pesquisa(tema):
    crew = Crew(
        agents=[analista_pesquisa, relator_institucional],
        tasks=[tarefa_pesquisa, tarefa_relatorio],
        verbose=True,
        llm=llm
    )
    
    return crew.kickoff(inputs={'tema': tema})

# Exemplo de uso
if __name__ == "__main__":
    tema_pesquisa = "cursos de medicina"
    resultado = executar_pesquisa(tema_pesquisa)
    print("\n\n=== RESULTADO FINAL ===")
    print(resultado)