import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

sequence = "I've been waiting for a HuggingFace course my whole life."

tokens = tokenizer.tokenize(sequence)
ids = tokenizer.convert_tokens_to_ids(tokens)

# input_ids = torch.tensor(ids)
input_ids = torch.tensor([ids]) # Transformer expect multiple sequences (batch*), so add [] (one dimension) instead.
print("Input IDs:", input_ids)

output = model(input_ids)
print("Logits:", output.logits)

# padding the inputs (to overcome the different size of the sentences)

model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

sequence1_ids = [[200, 200, 200]]
sequence2_ids = [[200, 200]]
batched_ids = [
    [200, 200, 200],
    [200, 200, tokenizer.pad_token_id], # tokenizer.pad_token_id refers to the padding the input.
]

print(model(torch.tensor(sequence1_ids)).logits) #tensor([[ 1.5694, -1.3895]], grad_fn=<AddmmBackward0>)
print(model(torch.tensor(sequence2_ids)).logits) #tensor([[ 0.5803, -0.4125]], grad_fn=<AddmmBackward0>)
print(model(torch.tensor(batched_ids)).logits) #tensor([[ 1.5694, -1.3895],[ 1.3374, -1.2163]], grad_fn=<AddmmBackward0>)

# The second output of the batch does not correspond to the sequence2, so it's necessary to mask the padding_id.

# Attention Mask
batched_ids = [
    [200, 200, 200],
    [200, 200, tokenizer.pad_token_id],
]

# Tell the model to ignore the 0 cells.
attention_mask = [
    [1, 1, 1],
    [1, 1, 0],
]

outputs = model(torch.tensor(batched_ids), attention_mask=torch.tensor(attention_mask)) # tensor([[ 1.5694, -1.3895],[ 0.5803, -0.4125]], grad_fn=<AddmmBackward0>)
print(outputs.logits)

# When dealing with the longer sequences, try to:

# 1. Use a model with a longer supported sequence length.
# 2. Truncate your sequences.

max_sequence_length = 128
sequence = sequence[:max_sequence_length]



