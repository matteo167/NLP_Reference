import torch

print("PyTorch versão:", torch.__version__)
print("CUDA disponível:", torch.cuda.is_available())
print("Número de GPUs:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("Nome da GPU:", torch.cuda.get_device_name(0))
