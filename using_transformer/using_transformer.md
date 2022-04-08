## Basic usage completed! 

Basic building blocks of a Transformer model: Tokenizer, model, postprocessing;

```python
from transformers import AutoTokenizer, AutoModel
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModel.from_pretrained(checkpoint)
sequences = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]
model_inputs = tokenizer(sequences, max_length=8, padding=True, truncation=True, return_tensors="pt")
outputs = model(**model_inputs)

# postprocessing
import torch
predictions = torch.nn.functional.softmax(outputs.ligits, dim=-1)
```

Tokenizer: word-based, character-based, subword-based

