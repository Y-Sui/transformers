from transformers import AutoModelForSequenceClassification, AutoTokenizer

# checkpoint-distilbert
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"

# tokenizer & model
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

sequence = "I've been waiting for a HuggingFace course my whole life."
sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]

# pt refers to pytorch, truncation refers to cut off the raw sequence according to the max_length.
model_input = tokenizer(sequences, padding="max_length", max_length=8, truncation=True, return_tensors="pt")

output = model(**model_input) # ** refers to input a dict form data

print(output)