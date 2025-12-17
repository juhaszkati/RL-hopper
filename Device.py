import torch

# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else  # for nvidia gpu
    "mps" if torch.backends.mps.is_available() else  # for apple mps
    "cpu"
)

print('Device: ', device)