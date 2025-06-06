from transformers import AutoModelForCausalLM, AutoTokenizer
import os

modelos = {
    "gpt2": "gpt2",
    "gpt2-medium": "gpt2-medium",
    "gpt2-xl": "gpt2-xl",
    "gpt-neo-1.3B": "EleutherAI/gpt-neo-1.3B",
#    "gpt-j-6B": "EleutherAI/gpt-j-6B",
#    "falcon-7b": "tiiuae/falcon-7b",
#    "mistral-7B": "mistralai/Mistral-7B-v0.1",
#    "llama2-7B": "meta-llama/Llama-2-7b-hf",
#    "dolly-v2-7B": "databricks/dolly-v2-7b",
#    "pythia-12B": "OpenAssistant/oasst-sft-1-pythia-12b"
}

for nome, model_name in modelos.items():
    local_dir = f"./modelos/{nome}"
    print(f"\n📥 Baixando modelo: {model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        os.makedirs(local_dir, exist_ok=True)
        tokenizer.save_pretrained(local_dir)
        model.save_pretrained(local_dir)
        print(f"✅ Modelo salvo em: {local_dir}")
    except Exception as e:
        print(f"❌ Erro ao baixar {model_name}: {e}")
