from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import random
import json

def show_random_elements(dataset, num_examples=10):
    assert num_examples <= len(dataset) # Can not pick more elements than there are in the dataset.
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset)-1)
        # print(dataset[pick])
        # print(dataset[pick]["utterance"]) # Only show the utterance
        picks.append(dataset[pick]["utterance"])
    return picks
def preprocess_dataset(dataset):
    picks = []
    for i in range(len(dataset)-1):
        picks.append(dataset[i]["utterance"])
    return picks

# Load the train data
dataset = json.load(open("webquestions_train.json"))
example = show_random_elements(dataset)
dataset = preprocess_dataset(dataset)

# Load the pre-trained model
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
# tokenized_input = tokenizer(dataset, is_split_into_words=False)
# tokens = tokenizer.convert_ids_to_tokens(tokenized_input["input_ids"])

model = AutoModelForTokenClassification.from_pretrained("distilbert-base-uncased")
nlp = pipeline("ner", model=model, tokenizer=tokenizer)
ner_results = nlp(example)
print(ner_results)


