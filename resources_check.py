import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import psutil

# Function to check memory
def check_memory():
    vm = psutil.virtual_memory()
    print(f"Total memory: {vm.total / (1024 ** 3):.2f} GB")
    print(f"Available memory: {vm.available / (1024 ** 3):.2f} GB")
    print(f"Used memory: {vm.used / (1024 ** 3):.2f} GB")

    sm = psutil.swap_memory()
    print(f"Total swap: {sm.total / (1024 ** 3):.2f} GB")
    print(f"Used swap: {sm.used / (1024 ** 3):.2f} GB")
    print(f"Free swap: {sm.free / (1024 ** 3):.2f} GB")

# Check memory before loading model
check_memory()

# Paths to the local model directory
local_model_path = "tiiuae/falcon-7b"

# Load the tokenizer and model from the local directory
tokenizer = AutoTokenizer.from_pretrained(local_model_path, trust_remote_code=True, ignore_mismatched_sizes=True)
model = AutoModelForCausalLM.from_pretrained(local_model_path, trust_remote_code=True, torch_dtype=torch.float16, ignore_mismatched_sizes=True)

# Ensure the model runs on the GPU if available
model = model.to("cuda")

# Check memory after loading model
check_memory()

# Ensure the inputs are also in the correct format and moved to GPU
def forward_pass(input_text):
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
    with torch.no_grad():  # No need to calculate gradients for inference
        outputs = model(**inputs)
    return outputs.logits

# Example usage of the forward_pass function
input_text = "Your input text here"
logits = forward_pass(input_text)
print(logits)
