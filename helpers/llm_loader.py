from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
import torch


# Load LLM on startup
# Set up device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define quantization config for 4-bit inference
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True, 
    bnb_4bit_compute_dtype=torch.float16
)

# Select a model (ensure this is compatible with 4-bit inference)
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"  # Change if needed

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load model with correct quantization and device mapping
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map={"": 0} if device == "cuda" else {"": "cpu"},
    quantization_config=quantization_config
)

# Use transformers pipeline for inference
generator = lambda prompt, max_length: model.generate(
    **tokenizer(prompt, return_tensors="pt").to(device), 
    max_length=max_length
)

print("Model loaded successfully on", device)