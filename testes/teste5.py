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
                f"Journal Information:\n"
                f"Title: {journal_info['title']}\n"
                f"Publisher: {journal_info['publisher']}\n"
                f"ISSN: {', '.join(journal_info['ISSN'])}\n"
                f"Total Articles: {journal_info['total_articles']}\n"
                f"Active Articles: {journal_info['active_articles']}"
            )
            
            return formatted_output
            
        except requests.exceptions.RequestException as e:
            return f"Error fetching journal information: {str(e)}"

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
                return "No journals found matching your query."
                
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
                formatted += f"   Area: {res['Evaluation Area']}\n"
                formatted += f"   ISSN: {res['ISSN']}\n"
                formatted += f"   Qualis: {res['Qualis Rating']}\n"
                formatted += f"   Similarity: {res['Similarity Score']:.3f}\n"
                formatted_results.append(formatted)
            
            return "Journal search results:\n" + "\n".join(formatted_results)
            
        except Exception as e:
            return f"Error searching journals: {str(e)}"

# Create researcher agent with both tools
researcher = Agent(
    role='Pesquisador Acadêmico',
    goal='Encontrar revistas científicas relevantes e obter informações detalhadas sobre elas',
    backstory='Especialista em identificar periódicos de qualidade para publicação e analisar seus dados',
    tools=[JournalSearchTool(), JournalInfoTool()],
    llm=llm,
    verbose=True
)

# Create tasks
search_task = Task(
    description='Liste os 10 periódicos mais relevantes de computation e medicine e seus ISSNs',
    agent=researcher,
    expected_output='Uma lista formatada com as 10 revistas mais similares, incluindo título, área de avaliação, ISSN e qualis rating'
)

info_task = Task(
    description='Para cada ISSN encontrado na tarefa anterior, obtenha informações detalhadas sobre o periódico',
    agent=researcher,
    expected_output='Informações detalhadas de cada periódico incluindo editora, total de artigos e artigos ativos',
    context=[search_task]
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
print("\nFinal Result:", result)