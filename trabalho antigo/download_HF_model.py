from huggingface_hub import snapshot_download
import os

modelos = {
    "phi-2": "microsoft/phi-2",
    "gemma-2b": "google/gemma-2b"
}

for nome_local, repo_id in modelos.items():
    local_dir = f"./modelos/{nome_local}"
    print(f"\n📥 Baixando {repo_id} para {local_dir}")
    
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            resume_download=True,
        )
        print(f"✅ Download concluído: {local_dir}")
    except Exception as e:
        print(f"❌ Erro ao baixar {repo_id}: {e}")
