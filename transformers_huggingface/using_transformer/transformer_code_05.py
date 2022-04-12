from transformers import BertModel, BertConfig

# Load the Bert config
config = BertConfig()
# print(config)

# Train the Bert with the state config
model = BertModel(config)

# Or just reused the pretrained model, the cache is located at  ~/.cache/huggingface/transformers.
model_pre = BertModel.from_pretrained("bert-base-cased")

# Save the model
# model_pre.save_pretrained("bert_retrained")

sequences = ["Hello!", "Cool.", "Nice!"]

# Tokenizer convert the sequences to the encoded_sequences
encoded_sequences = [
    [101, 7592, 999, 102],
    [101, 4658, 1012, 102],
    [101, 3835, 999, 102],
]

import torch
model_inputs = torch.tensor(encoded_sequences)

output = model_pre(model_inputs)
print(output)


