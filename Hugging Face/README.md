# Hugging Face Models Using Transformers

## Overview

This project demonstrates the use of pre-trained models from Hugging Face using the Transformers library in Python. It explores three Natural Language Processing (NLP) tasks: Sentiment Analysis, Text Generation, and Text Summarization.

The experiments are implemented in VS Code using the latest installed Transformers version (5.24.0).

## Technologies Used

- **Python 3.12**
- **Hugging Face Transformers**
- **PyTorch**
- **Hugging Face Hub**

## Installation

Install the required Python libraries:

```bash
python -m pip install --upgrade transformers torch huggingface_hub
```

## Experiments

### 1. Sentiment Analysis

**Objective:** Identify the sentiment expressed in a given text using a pre-trained sentiment classification model.

**Model:** `distilbert/distilbert-base-uncased-finetuned-sst-2-english`

**Input:**
```text
I love using Hugging Face models!
```

**Output:**
```python
[{'label': 'POSITIVE', 'score': 0.9992625117301941}]
```

**Result:** The model successfully classified the input sentence as POSITIVE with a confidence score of approximately 99.93%.

### 2. Text Generation

**Objective:** Generate new text based on a given input prompt using a pre-trained GPT-2 language model.

**Model:** `gpt2`

**Input Prompt:**
```text
Once upon a time, in a land far away,
```

**Parameters:**
- `max_new_tokens = 50`
- `num_return_sequences = 2`

**Output:**

**Generated Text 1:**
```text
Once upon a time, in a land far away, the king made his decision for the people he thought he should rule.
```

The subsequent generated passage begins with:
```text
The people of Haldor were divided into three distinct groups:
the Arian, the Arian-born, and the Arian-born-born...
```

**Generated Text 2:**
```text
Once upon a time, in a land far away, there would be no more darkness and no more fear. It would be like the night before. But then, a little later, they would see a light.
```

The subsequent generated passage continues:
```text
They would be like their parents. They would be like their children...
```

**Result:** GPT-2 generated two different text sequences from the same input prompt, demonstrating its ability to predict and generate contextually relevant text.

*Note: Generated text may vary between executions.*

### 3. Text Summarization

**Objective:** Generate a concise summary of a given paragraph using a pre-trained sequence-to-sequence model.

**Model:** `facebook/bart-large-cnn`

**Input Text:**
```text
Artificial intelligence is transforming industries worldwide.
From healthcare to finance, AI systems automate complex tasks.
Companies are investing billions to deploy AI at scale.
```

**Implementation:**

The summarization task uses:
- `AutoTokenizer` for text tokenization.
- `AutoModelForSeq2SeqLM` for loading the pre-trained BART model.
- `model.generate()` for generating the summary.
- `tokenizer.decode()` for converting generated tokens into readable text.

**Parameters:**
- `max_new_tokens = 50`
- `min_new_tokens = 10`
- `num_beams = 4`
- `do_sample = False`

**Output:**
```text
Artificial intelligence is transforming industries worldwide.
From healthcare to finance, AI systems automate complex tasks.
```

**Result:** The BART model successfully generated a shorter version of the original paragraph while retaining its central information.

## Transformers Version

```text
Transformers Version: 5.24.0
```

The implementation uses `AutoTokenizer` and `AutoModelForSeq2SeqLM` for summarization instead of the older `pipeline('summarization')` approach.

## Key Learnings

- Understanding how to use the Hugging Face Transformers library.
- Loading and using pre-trained NLP models.
- Performing sentiment classification.
- Generating text using GPT-2.
- Understanding tokenization and sequence generation.
- Performing text summarization using BART.
- Working with the AutoTokenizer and AutoModel APIs.

## Conclusion

This hands-on implementation demonstrates how pre-trained Hugging Face models can be used to perform different NLP tasks with Python. It provides practical exposure to transformer-based models, text processing, language generation, and summarization.