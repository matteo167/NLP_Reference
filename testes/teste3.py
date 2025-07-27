from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool
from typing import Optional
import pandas as pd
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
import os

llm=LLM(model="ollama/llama3.2:3b", base_url="http://localhost:11434")


class JournalSearchTool(BaseTool):
    name: str = "Journal Search"
    description: str = "Searches for academic journals in the Sucupira database based on similarity to the query. Returns journal titles, evaluation areas, ISSN, Qualis rating, and similarity scores."
    
    # Declare all fields as class attributes
    chroma_db_dir: str = "../sucupira_chroma_db"
    embedding_model_name: str = 'paraphrase-MiniLM-L6-v2'
    embedding_function: Optional[SentenceTransformerEmbeddings] = None
    vectorstore: Optional[Chroma] = None
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Now you can initialize these fields
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
            
            # Format the results nicely
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


# Exemplo de uso em um agente
researcher = Agent(
    role='Pesquisador Acadêmico',
    goal='Encontrar revistas científicas relevantes',
    backstory='Especialista em identificar periódicos de qualidade para publicação',
    tools=[JournalSearchTool()],
    llm=llm
)

# Create task
research_task = Task(
    description='liste os 10 periódicos mais relevantes de computation e medicine e seus issn',
    agent=researcher,
    expected_output='lista com as 10 revistas mais similares com o issn'
)

# Create and run crew
crew = Crew(
    agents=[researcher],
    tasks=[research_task],
    process=Process.sequential,
    verbose=True,
    llm=llm
)

result = crew.kickoff()
print("\nResult:", result)