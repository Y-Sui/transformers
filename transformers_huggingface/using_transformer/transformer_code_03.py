# Tokenizer: All the preprocessing needs to be done as same as the pretrained model.
from transformers import AutoTokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# Once we have the tokenizer, we can directly pass our sentences to it and we’ll get back a dictionary that’s ready to
#   feed to our model! The only thing left to do is to convert the list of input IDs to tensors.
raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
print(inputs)
