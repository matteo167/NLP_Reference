from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool  

llm=LLM(model="ollama/llama3.2:3b", base_url="http://localhost:11434")


class CalculatorTool(BaseTool):
    name: str = "Calculator"
    description: str = "Performs basic arithmetic calculations"

    def _run(self, expression: str) -> str:
        try:
            allowed_chars = set("0123456789+-*/. ()")
            if not all(c in allowed_chars for c in expression):
                return "Error: Only basic arithmetic operations allowed"
            
            result = eval(expression)
            return f"The result of {expression} is {result}"
        except Exception as e:
            return f"Error calculating: {str(e)}"

# Create agent
math_agent = Agent(
    role='Math Specialist',
    goal='Perform accurate calculations',
    backstory='Expert in arithmetic operations',
    tools=[CalculatorTool()],
    verbose=True,
    llm=llm
)

# Create task
calculation_task = Task(
    description='qual é o resultado do cálculo (15 * 3) + (20 / 4) - 10',
    agent=math_agent,
    expected_output='The exact numerical result'
)

# Create and run crew
crew = Crew(
    agents=[math_agent],
    tasks=[calculation_task],
    process=Process.sequential,
    verbose=True,
    llm=llm
)

result = crew.kickoff()
print("\nResult:", result)