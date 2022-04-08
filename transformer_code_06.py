# Tokenizer * is vital

# Word-based

# Character-based

# Subword-based tokenizer (word+character)
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

# from transformers import AutoTokenizer
#
# tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

output = tokenizer("Using a Transformer network is simple")
print(output)

# Save model
# tokenizer.save_pretrained("directory_on_my_computer")