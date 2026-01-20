import torch

print(torch.cuda.memory_allocated() / 1e6, "MB allocated")
print(torch.cuda.memory_reserved() / 1e6, "MB reserved")


import torch

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device name:", torch.cuda.get_device_name(0))
    print("Memory total:", torch.cuda.get_device_properties(0).total_memory / 1e9, "GB")
    # x = torch.rand(2, 2).to("cuda")
    # print("Allocated after tensor:", torch.cuda.memory_allocated() / 1e6, "MB")
    # torch.cuda.empty_cache()