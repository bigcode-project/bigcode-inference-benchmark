import copy
import math
from typing import List
import random

dummy_input_sentences = [
    "DeepSpeed is a machine learning framework",
    "He is working on",
    "He has a",
    "He got all",
    "Everyone is happy and I can",
    "The new movie that got Oscar this year",
    "In the far far distance from our galaxy,",
    "Peace is the only way",
]


def get_dummy_batch(batch_size: int, input_sentences: List[str] = None) -> List[str]:
    if input_sentences is None:
        input_sentences = copy.deepcopy(dummy_input_sentences)

    if batch_size > len(input_sentences):
        input_sentences *= math.ceil(batch_size / len(input_sentences))
    input_sentences = input_sentences[:batch_size]

    return input_sentences

def get_dummy_batch_tokenizer(batch_size : int, tokenizer, max_input_length : int) -> List[str]:
    input_sentences = []
    for i in range(batch_size):
        sentence = [random.randint(0, tokenizer.vocab_size-1) for _ in range(max_input_length)]
        input_sentences.append(tokenizer.decode(sentence))
    return input_sentences