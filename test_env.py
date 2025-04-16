import torch
import numpy as np
import librosa
import platform

print(f"Python ვერსია: {platform.python_version()}")
print(f"PyTorch ვერსია: {torch.__version__}")
print(f"NumPy ვერსია: {np.__version__}")
print(f"Librosa ვერსია: {librosa.__version__}")
print(f"MPS ხელმისაწვდომია: {torch.backends.mps.is_available()}")

# შევამოწმოთ M1 აქსელერაცია
if torch.backends.mps.is_available():
    device = torch.device("mps")
    x = torch.ones(1, device=device)
    print(f"ტენზორი M1 აქსელერაციაზე: {x}")
    print("M1 აქსელერაცია წარმატებით მუშაობს!")