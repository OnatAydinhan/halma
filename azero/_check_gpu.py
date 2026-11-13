import torch

print("torch version:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("cuda runtime:", torch.version.cuda)
    # quick tensor test
    x = torch.randn(4, 4, device="cuda")
    print("cuda matmul ok:", (x @ x.T).shape)
else:
    print("CPU only")
