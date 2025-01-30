import torch
print(torch.__version__)  # Should include '+cu118' or similar
print(torch.cuda.is_available())  # Should return True
print(torch.cuda.get_device_name(0))  # Should display your GPU name
