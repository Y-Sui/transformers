import torch
from transformers import AdamW, AutoTokenizer, AutoModelForSequenceClassification

# Same as before
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
sequences = [
    "I've been waiting for a HuggingFace course my whole life.",
    "This course is amazing!",
]
batch = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")
# print(batch)

# This is new, It seems just add a labels key-value to the batch
batch["labels"] = torch.tensor([1, 1])

# print(batch)

optimizer = AdamW(model.parameters())
loss = model(**batch).loss # #这里的 loss 是直接根据 batch 中提供的 labels 来计算的，回忆：前面章节查看 model 的输出的时候，有loss这一项
loss.backward()
optimizer.step()