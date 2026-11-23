
import torch
import transformers
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)

print("Transformers Version:", transformers.__version__)


# ==========================================
# TASK 1: SENTIMENT ANALYSIS
# ==========================================

classifier = pipeline('sentiment-analysis')

result = classifier("I love using Hugging Face models!")

print("\nTASK 1: SENTIMENT ANALYSIS")
print(result)


# ==========================================
# TASK 2: TEXT GENERATION
# ==========================================

generator = pipeline('text-generation', model='gpt2')

output = generator(
    "Once upon a time, in a land far away,",
    max_new_tokens=50,
    num_return_sequences=2
)

print("\nTASK 2: TEXT GENERATION")

for i, text in enumerate(output):
    print(f"Generated Text {i+1}: {text['generated_text']}")


# ==========================================
# TASK 3: TEXT SUMMARIZATION
# Using AutoTokenizer and AutoModelForSeq2SeqLM
# Compatible with Transformers v5
# ==========================================

model_name = "facebook/bart-large-cnn"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load pre-trained BART model
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Input text
text = '''Artificial intelligence is transforming industries worldwide.
From healthcare to finance, AI systems automate complex tasks.
Companies are investing billions to deploy AI at scale.'''

# Tokenize the input
inputs = tokenizer(
    text,
    return_tensors="pt",
    truncation=True
)

# Generate summary
with torch.no_grad():
    summary_ids = model.generate(
        **inputs,
        max_new_tokens=50,
        min_new_tokens=10,
        num_beams=4,
        do_sample=False
    )

# Convert tokens into readable text
summary = tokenizer.decode(
    summary_ids[0],
    skip_special_tokens=True
)

print("\nTASK 3: TEXT SUMMARIZATION")
print(summary)
