from transformers import AutoTokenizer, T5ForConditionalGeneration, pipeline

tokenizer = AutoTokenizer.from_pretrained("checkpoint")
model = T5ForConditionalGeneration.from_pretrained("checkpoint")
summarizer = pipeline(task="summarization", model=model, tokenizer=tokenizer)
x = summarizer("what are the languages spoken in the movies whose directors also directed [Son of Dracula]", max_length=12)

# use t5 in tf
# x = summarizer("An apple a day, keeps the doctor away", max_length=6, min_length=1)
print(x)