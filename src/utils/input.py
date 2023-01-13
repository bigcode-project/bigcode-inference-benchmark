import copy
import math
from typing import List


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


def get_dummy_batch(batch_size: int, max_input_length: int = -1) -> List[str]:
    if max_input_length == -1:
        input_sentences = copy.deepcopy(dummy_input_sentences)
    else:
        input_sentences = batch_size * ["Hello " * max_input_length]

    if batch_size > len(input_sentences):
        input_sentences *= math.ceil(batch_size / len(input_sentences))
    input_sentences = input_sentences[:batch_size]

    return input_sentences
