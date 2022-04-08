from datasets import load_dataset

# MRPC is one of the GLUE datasets.
raw_datasets = load_dataset("glue", "mrpc")
print(raw_datasets)

raw_train_dataset = raw_datasets["train"]
# raw_train_dataset[0]
# raw_train_dataset.features

from transformers import AutoTokenizer

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# tokenizer (one disadvantage, large ram)
# tokenizer可以直接处理sequence pair
tokenized_dataset = tokenizer(
    raw_datasets["train"]["sentence1"],
    raw_datasets["train"]["sentence2"],
    padding=True,
    truncation=True,
)

# tokenizer (another way)
def tokenize_function(example):
    # 这里可以添加多种操作，不光是tokenize
    # 这个函数处理的对象，就是Dataset这种数据类型，通过features中的字段来选择要处理的数据
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

# Why Map?
# The results of the function are cached, so it won't take any time if we re-execute the code.
# It can apply multiprocessing to go faster than applying the function on each element of the dataset.
# It does not load the whole dataset into memory, saving the results as soon as one element is processed.

# Still no padding!因为如果使用了padding之后，就会全局统一对一个maxlen进行padding，这样无论在tokenize还是模型的训练上都不够高效

# dynamic padding
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

samples = tokenized_datasets["train"][:8]
samples = {k: v for k, v in samples.items() if k not in ["idx", "sentence1", "sentence2"]}
batch = data_collator(samples)