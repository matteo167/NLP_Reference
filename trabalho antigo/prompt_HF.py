from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

LOCAL_MODEL_PATH = "./modelos/gpt2-medium"

def gerar_info_modelo(model_name="gpt2"):
    # Carrega tokenizer e modelo
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Imprime informações do modelo GPT-2
    print(f"Modelo carregado: {model_name}")
    print(f"Tamanho do vocabulário: {tokenizer.vocab_size}")
    
    # Para obter o número de camadas (layers), depende da arquitetura
    # GPT2 geralmente tem o atributo config.n_layer
    n_layers = getattr(model.config, "n_layer", "Desconhecido")
    print(f"Número de camadas (layers): {n_layers}")


def gerar_texto(prompt, model_name="gpt2", max_length=100, temperature=0.7):
    # Carrega tokenizer e modelo
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Tokeniza o prompt
    inputs = tokenizer(prompt, return_tensors="pt")

    # Gera texto continuando o prompt
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        temperature=temperature,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        num_return_sequences=1
    )

    # Decodifica e retorna o texto gerado
    texto_gerado = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return texto_gerado

if __name__ == "__main__":
    prompt = "Once upon a time"
    resultado = gerar_texto(prompt)
    print("Texto gerado:\n", resultado)
