from typing import List, Union

import torch
from transformers import PreTrainedTokenizerBase


def safe_decode(tokenizer: PreTrainedTokenizerBase, outputs: Union[torch.Tensor, List[int]]) -> str:
    # Workaround to make SentencePiece .decode() keep leading spaces in a token
    fake_token = tokenizer("^")["input_ids"][0]
    outputs = outputs.tolist() if isinstance(outputs, torch.Tensor) else outputs
    result = tokenizer.decode([fake_token] + outputs)

    # We use .lstrip() since SentencePiece may add leading spaces, e.g. if the outputs are "</s>"
    return result.lstrip()[1:]
