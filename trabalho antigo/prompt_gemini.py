import os
from dotenv import load_dotenv
import google.generativeai as genai

# Carrega variáveis de ambiente do arquivo .env
load_dotenv()
api_key = os.getenv("Chave_secreta")

print(api_key)
if api_key:
    print("Chave de API do Gemini carregada do arquivo .env.")
    os.environ["GOOGLE_API_KEY"] = api_key
else:
    print("Erro: A chave 'GOOGLE_API_KEY' não foi encontrada no .env.")


# Configura o client do Gemini
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))



def chamada_LLM(prompt, model_name='gemini-2.0-flash', temperature=0.7, max_output_tokens=500):
  try:
    model = genai.GenerativeModel(model_name)
    response = model.generate_content(
        prompt,
        generation_config={
            "temperature": temperature,
            "max_output_tokens": max_output_tokens
        }
    )
    return response.text
  except Exception as e:
    print(f"Erro ao gerar resposta com o modelo {model_name}: {e}")
    return None
  


# Lista de prompts
prompts = [
    "Explique o que é aprendizado de máquina.",
    "Dê um resumo da teoria da relatividade.",
    "Como funciona uma blockchain?",
    "Quais são os principais benefícios da energia solar?",
    "O que é o método científico?"
]

# Loop para gerar e imprimir respostas
for i, prompt in enumerate(prompts, start=1):
    print(f"\n🔹 Prompt {i}: {prompt}")
    resposta = chamada_LLM(prompt)
    print(f"🔸 Resposta:\n{resposta}")