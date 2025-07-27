
from crewai import Agent, Task, Process, Crew, LLM

#LLM Object from crewai package
llm=LLM(model="ollama/llama3.2:1b", base_url="http://localhost:11434")

info_agent = Agent(
    role="Information Agent",
    goal="Give compelling information about a certain topic",
    backstory="""
        You love to know information.  People love you for it.
    """,
    llm=llm
)

task1 = Task(
    description="Tell me all about the eagles.",
    expected_output="Give me a quick summary and then also give me 7 bullet points describing it.",
    agent=info_agent
)

crew = Crew(
    agents=[info_agent],
    tasks=[task1],
    verbose=True
)

result = crew.kickoff()

print("############")
print(result)