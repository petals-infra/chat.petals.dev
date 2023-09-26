from traceback import format_exc

import hivemind
from flask import jsonify, request

import config
from app import app, models
from utils import safe_decode

logger = hivemind.get_logger(__file__)


@app.post("/api/v1/generate")
def http_api_generate():
    try:
        model_name = get_typed_arg("model", str)
        inputs = request.values.get("inputs")
        do_sample = get_typed_arg("do_sample", int, False)
        temperature = get_typed_arg("temperature", float)
        top_k = get_typed_arg("top_k", int)
        top_p = get_typed_arg("top_p", float)
        repetition_penalty = get_typed_arg("repetition_penalty", float)
        max_length = get_typed_arg("max_length", int)
        max_new_tokens = get_typed_arg("max_new_tokens", int)
        logger.info(f"generate(), {model_name=}, {inputs=}")

        model, tokenizer, backend_config = models[model_name]
        if not backend_config.public_api:
            raise ValueError(f"We do not provide public API for {model_name} due to license restrictions")

        if inputs is not None:
            inputs = tokenizer(inputs, return_tensors="pt")["input_ids"].to(config.DEVICE)
            n_input_tokens = inputs.shape[1]
        else:
            n_input_tokens = 0

        outputs = model.generate(
            inputs=inputs,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_length=max_length,
            max_new_tokens=max_new_tokens,
        )
        outputs = safe_decode(tokenizer, outputs[0, n_input_tokens:])
        logger.info(f"generate(), outputs={repr(outputs)}")

        return jsonify(ok=True, outputs=outputs)
    except Exception:
        return jsonify(ok=False, traceback=format_exc())


def get_typed_arg(name, expected_type, default=None):
    value = request.values.get(name)
    return expected_type(value) if value is not None else default
