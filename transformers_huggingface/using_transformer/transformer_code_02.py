from transformers import pipeline

# Pipeline -> Tokenizer, Model, PostProcessing(Prediction)
classifier = pipeline("sentiment-analysis", model='distilbert-base-uncased-finetuned-sst-2-english')
output = classifier(
    [
        "I've been waiting for a HuggingFace course my whole life.",
        "I hate this so much!",
    ]
)
print(output)

