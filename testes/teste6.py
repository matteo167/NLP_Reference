from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool
from typing import Optional
import requests
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
import os

llm = LLM(model="ollama/llama3.2:3b", base_url="http://localhost:11434")

class JournalInfoTool(BaseTool):
    name: str = "Journal Information"
    description: str = "Retrieves detailed information about academic journals using their ISSN. Returns title, publisher, total articles, and active articles from Crossref API."
    
    def _run(self, issn: str) -> str:
        """
        Retrieves journal information from Crossref API using ISSN
        
        Args:
            issn: The ISSN of the journal
            
        Returns:
            Formatted string with journal information
        """
        url = f"https://api.crossref.org/journals/{issn}"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            message = data.get("message", {})
            journal_info = {
                "title": message.get("title", "Não disponível"),
                "publisher": message.get("publisher", "Não disponível"),
                "ISSN": message.get("ISSN", []),
                "total_articles": message.get("counts", {}).get("total-dois", 0),
                "active_articles": message.get("counts", {}).get("current-dois", 0)
            }
            
            formatted_output = (
                f"Informações do Periódico:\n"
                f"Título: {journal_info['title']}\n"
                f"Editora: {journal_info['publisher']}\n"
                f"ISSN: {', '.join(journal_info['ISSN'])}\n"
                f"Total de Artigos: {journal_info['total_articles']}\n"
                f"Artigos Ativos: {journal_info['active_articles']}"
            )
            
            return formatted_output
            
        except requests.exceptions.RequestException as e:
            return f"Erro ao buscar informações do periódico: {str(e)}"

class JournalSearchTool(BaseTool):
    name: str = "Journal Search"
    description: str = "Searches for academic journals in the Sucupira database based on similarity to the query. Returns journal titles, evaluation areas, ISSN, Qualis rating, and similarity scores."
    
    chroma_db_dir: str = "../sucupira_chroma_db"
    embedding_model_name: str = 'paraphrase-MiniLM-L6-v2'
    embedding_function: Optional[SentenceTransformerEmbeddings] = None
    vectorstore: Optional[Chroma] = None
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not os.path.exists(self.chroma_db_dir):
            raise ValueError(f"Database directory '{self.chroma_db_dir}' not found")
            
        self.embedding_function = SentenceTransformerEmbeddings(model_name=self.embedding_model_name)
        self.vectorstore = Chroma(
            persist_directory=self.chroma_db_dir,
            embedding_function=self.embedding_function
        )
    
    def _run(self, query: str, k: Optional[int] = 5) -> str:
        """
        Searches for journals similar to the query
        
        Args:
            query: The search query (journal name, area, etc.)
            k: Number of results to return (default 5)
            
        Returns:
            Formatted string with search results
        """
        try:
            k = int(k) if k else 5
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            
            if not results:
                return "Nenhum periódico encontrado para sua busca."
                
            output = []
            for doc, score in results:
                output.append({
                    "Title": doc.metadata.get("Título", "N/A"),
                    "Evaluation Area": doc.metadata.get("Área de Avaliação", "N/A"),
                    "ISSN": doc.metadata.get("ISSN", "N/A"),
                    "Qualis Rating": doc.metadata.get("Estrato", "N/A"),
                    "Similarity Score": float(score)
                })
            
            formatted_results = []
            for i, res in enumerate(output, 1):
                formatted = f"\n{i}. {res['Title']}\n"
                formatted += f"   Área: {res['Evaluation Area']}\n"
                formatted += f"   ISSN: {res['ISSN']}\n"
                formatted += f"   Qualis: {res['Qualis Rating']}\n"
                formatted += f"   Similaridade: {res['Similarity Score']:.3f}\n"
                formatted_results.append(formatted)
            
            return "Resultados da busca de periódicos:\n" + "\n".join(formatted_results)
            
        except Exception as e:
            return f"Erro ao buscar periódicos: {str(e)}"

# Create researcher agent with both tools
researcher = Agent(
    role='Especialista em Periódicos Científicos',
    goal='Identificar e detalhar informações sobre revistas científicas relevantes nas áreas de Computação e Medicina.',
    backstory='Um pesquisador experiente com profundo conhecimento em bases de dados acadêmicas, focado em encontrar periódicos de alta qualidade para publicação e análise de dados.',
    tools=[JournalSearchTool(), JournalInfoTool()],
    llm=llm,
    verbose=True
)

# Create tasks
search_task = Task(
    description='Listar os 10 periódicos mais relevantes nas áreas de Computação e Medicina, incluindo seus ISSNs.',
    agent=researcher,
    expected_output='Uma lista detalhada com os 10 periódicos mais relevantes para as áreas especificadas, contendo Título, Área de Avaliação, ISSN e Qualis Rating de cada um.'
)

info_task = Task(
    description='Para cada ISSN identificado na tarefa anterior, obter informações detalhadas do periódico.',
    agent=researcher,
    expected_output='Informações completas de cada periódico, incluindo a Editora, o Total de Artigos publicados e o número de Artigos Ativos.'
)

# Create and run crew
crew = Crew(
    agents=[researcher],
    tasks=[search_task, info_task],
    process=Process.sequential,
    verbose=True,
    llm=llm
)

result = crew.kickoff()
print("\nResultado Final:", result)