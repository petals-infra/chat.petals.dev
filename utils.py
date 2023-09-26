from typing import Dict, List, Tuple, Union

import hivemind
import torch
from petals import AutoDistributedModelForCausalLM
from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

import config
from data_structures import ModelInfo

logger = hivemind.get_logger(__file__)


def load_models() -> Dict[str, Tuple[PreTrainedModel, PreTrainedTokenizer, ModelInfo]]:
    models = {}
    for family in config.MODELS.values():
        for model_info in family:
            logger.info(f"Loading tokenizer for {model_info.repository}")
            tokenizer = AutoTokenizer.from_pretrained(model_info.repository, add_bos_token=False, use_fast=False)

            logger.info(
                f"Loading model {model_info.repository} with adapter {model_info.adapter} in {config.TORCH_DTYPE}"
            )
            # We set use_fast=False since LlamaTokenizerFast takes a long time to init
            model = AutoDistributedModelForCausalLM.from_pretrained(
                model_info.repository,
                active_adapter=model_info.adapter,
                torch_dtype=config.TORCH_DTYPE,
                initial_peers=config.INITIAL_PEERS,
                max_retries=3,
            )
            model = model.to(config.DEVICE)

            for key in [model_info.key] + list(model_info.aliases):
                models[key] = model, tokenizer, model_info
    return models


def safe_decode(tokenizer: PreTrainedTokenizer, outputs: Union[torch.Tensor, List[int]]) -> str:
    # Workaround to make SentencePiece .decode() keep leading spaces in a token
    fake_token = tokenizer("^")["input_ids"][0]
    outputs = outputs.tolist() if isinstance(outputs, torch.Tensor) else outputs
    result = tokenizer.decode([fake_token] + outputs)

    # We use .lstrip() since SentencePiece may add leading spaces, e.g. if the outputs are "</s>"
    return result.lstrip()[1:]
