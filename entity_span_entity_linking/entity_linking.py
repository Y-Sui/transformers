import spacy
import json, random # load the data

def show_random_elements(dataset, num_examples=10):
    assert num_examples <= len(dataset) # Can not pick more elements than there are in the dataset.
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset)-1)
        # print(dataset[pick])
        # print(dataset[pick]["utterance"]) # Only show the utterance
        picks.append(dataset[pick]["utterance"])
    return picks

# Load the train data
dataset = json.load(open("webquestions_train.json"))
example = show_random_elements(dataset)

# initialize language model
nlp = spacy.load("en_core_web_md")

# add pipeline (declared through entry_points in setup.py)
nlp.add_pipe("entityLinker", last=True)

for i in range(len(example)):
    doc = nlp(example[i])
    # returns all entities in the whole document
    all_linked_entities = doc._.linkedEntities
    # iterates over sentences and prints linked entities
    print("Example number - {}, raw_utterance - {}".format(i, example[i]))
    for sent in doc.sents:
        sent._.linkedEntities.pretty_print()
