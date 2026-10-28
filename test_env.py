import torch

print("PyTorch version:", torch.__version__)
print("PyTorch CUDA version:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("GPU memory (GiB):",
          round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))

    x = torch.randn(4096, 4096, device="cuda")
    y = x @ x
    torch.cuda.synchronize()

    print("Tensor device:", y.device)
    print("Tensor shape:", y.shape)
    print("GPU computation: SUCCESS")
else:
    print("GPU computation: FAILED")