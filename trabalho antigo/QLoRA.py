import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from trl import SFTTrainer
from datasets import Dataset

# Caminhos e configurações
LOCAL_MODEL_PATH = "./modelos/phi-2"
CSV_PATH = "dados_sinteticos_en.csv"
OUTPUT_DIR = "./modelo_qlora"

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Usando o dispositivo:", device)

# Carrega modelo base e prepara para treinamento QLoRA
def carregar_modelo_e_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        load_in_4bit=True  # usa QLoRA (4-bit)
    )
    model = prepare_model_for_kbit_training(model)
    return model, tokenizer

# Prepara os dados a partir do CSV
def preparar_dados(csv_path):
    df = pd.read_csv(csv_path)
    df["text"] = "Question: " + df["Question"] + " Answer: " + df["Answer"]
    dataset = Dataset.from_pandas(df[["text"]])
    return dataset

# Treinamento do adaptador LoRA
def treinar_qlora():
    model, tokenizer = carregar_modelo_e_tokenizer(LOCAL_MODEL_PATH)
    dataset = preparar_dados(CSV_PATH)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],  # depende do modelo (ajuste se necessário)
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        logging_steps=10,
        learning_rate=2e-4,
        fp16=True,
        save_total_limit=2,
        save_strategy="epoch",
        evaluation_strategy="no",
        report_to="none"
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        tokenizer=tokenizer,
        args=training_args,
        dataset_text_field="text"
    )

    model.config.use_cache = False  # evita warnings do Trainer
    trainer.train()

    # Salvar adaptador LoRA
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Modelo LoRA salvo em {OUTPUT_DIR}")

if __name__ == "__main__":
    treinar_qlora()
