# Prepare the model
from transformers import AutoModelForSequenceClassification, AutoTokenizer
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

# High dimensional vector: Batch Size, Sequence Length, Hidden Size
raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
outputs = model(**inputs)

# Postprocessing the output
print(outputs.logits)
import torch

predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
print(predictions)

# Model heads: Making sense out of numbers, using full model with model head instead of only model.