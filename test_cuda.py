
import torch
import sys

print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    try:
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        
        # Try to allocate some memory
        print("Attempting to allocate 1GB on GPU...")
        x = torch.ones(250 * 1024 * 1024, device='cuda') # ~1GB (float32 is 4 bytes)
        print("Allocation successful!")
        print(f"Memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"Memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        
    except Exception as e:
        print(f"ERROR during allocation: {e}")
else:
    print("CUDA is NOT available.")
