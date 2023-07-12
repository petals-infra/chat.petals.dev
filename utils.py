import torch
from transformers import PreTrainedTokenizerBase


def safe_decode(tokenizer: PreTrainedTokenizerBase, outputs: torch.Tensor):
    # Workaround to make SentencePiece .decode() keep leading spaces in a token
    fake_token = tokenizer("^")["input_ids"][0]
    result = tokenizer.decode([fake_token] + outputs.tolist())

    # We use .lstrip() since SentencePiece may add leading spaces, e.g. if the outputs are "</s>"
    return result.lstrip()[1:]
